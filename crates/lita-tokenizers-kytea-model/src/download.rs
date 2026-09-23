use reqwest::Client;
use reqwest::Error as ReqError;
use reqwest::IntoUrl;

use futures_core::Stream;

use tokio_util::io::StreamReader;

use bytes::Bytes;

use pin_project_lite::pin_project;

use async_compression::tokio::bufread::XzDecoder;

use std::pin::Pin;
use std::task::{Context, Poll};

const GITHUB_RAW: &str = "raw.githubusercontent.com";
const REPO: &str = "naughie/lita-tokenizers-kytea-model";
const BRANCH: &str = "main";
const PATH: &str = "/models/default.bin.xz";

const DEFAULT_URL: &str = const_format::formatcp!("https://{GITHUB_RAW}/{REPO}/{BRANCH}{PATH}");

pub struct Downloader<Fut> {
    inner: Fut,
}

pub fn download_model(
    client: &Client,
) -> Downloader<impl Future<Output = ModelStream<impl Stream<Item = Result<Bytes, ReqError>>>>> {
    download_model_with_url(client, DEFAULT_URL)
}

pub fn download_model_with_url<T: IntoUrl>(
    client: &Client,
    url: T,
) -> Downloader<impl Future<Output = ModelStream<impl Stream<Item = Result<Bytes, ReqError>>>>> {
    let res = download_impl(client, url);
    Downloader { inner: res }
}

async fn download_impl(
    client: &Client,
    url: impl IntoUrl,
) -> ModelStream<impl Stream<Item = Result<Bytes, ReqError>>> {
    let res = client.get(url).send().await;
    let res = res.and_then(|res| res.error_for_status());
    match res {
        Ok(res) => {
            let stream = res.bytes_stream();
            ModelStream::InProgress { inner: stream }
        }
        Err(e) => ModelStream::Err { inner: e },
    }
}

type ModelStreamReader<S> = XzDecoder<StreamReader<ModelStream<S>, Bytes>>;

pub enum Infallible {}
impl From<Infallible> for std::io::Error {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

pin_project! {
    #[project = _ModelStreamProj]
    pub enum ModelStream<S> {
        InProgress { #[pin] inner: S },
        Err { inner: ReqError },
    }
}

impl<S> Stream for ModelStream<S>
where
    S: Stream<Item = Result<Bytes, ReqError>>,
{
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.as_mut().project() {
            _ModelStreamProj::InProgress { inner } => match <_ as Stream>::poll_next(inner, cx) {
                Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
                Poll::Ready(Some(Err(e))) => {
                    self.set(Self::Err { inner: e });
                    Poll::Ready(None)
                }
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            },
            _ModelStreamProj::Err { .. } => Poll::Ready(None),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::InProgress { inner } => <_ as Stream>::size_hint(inner),
            Self::Err { .. } => (0, Some(0)),
        }
    }
}

impl<Fut, S> Downloader<Fut>
where
    Fut: Future<Output = ModelStream<S>>,
    S: Stream<Item = Result<Bytes, ReqError>>,
{
    pub async fn handle<H>(self, handler: H) -> Result<H::Output, H::Error>
    where
        H: Handler<ModelStreamReader<S>>,
        H::Error: From<ReqError>,
    {
        let mut rdr = match self.inner.await {
            ModelStream::InProgress { inner } => {
                let rdr = StreamReader::new(ModelStream::InProgress { inner });
                XzDecoder::new(rdr)
            }
            ModelStream::Err { inner } => {
                return Err(<H::Error as From<ReqError>>::from(inner));
            }
        };

        match handler.handle(&mut rdr).await {
            Ok(output) => {
                let model = rdr.into_inner().into_inner();
                if let ModelStream::Err { inner } = model {
                    Err(<H::Error as From<ReqError>>::from(inner))
                } else {
                    Ok(output)
                }
            }
            Err(e) => Err(e),
        }
    }
}

pub trait Handler<R>: Sized {
    type Output;
    type Error;

    fn handle(self, model: &mut R) -> impl Future<Output = Result<Self::Output, Self::Error>>;
}

pub use io::{CopyToIo, Error as CopyToIoError};
mod io {
    use super::Handler;

    use reqwest::Error as ReqError;

    use tokio::io::AsyncRead;
    use tokio::io::AsyncWrite;
    use tokio::io::AsyncWriteExt as _;

    use std::io::Error as IoError;

    pub struct CopyToIo<W> {
        to: W,
    }

    #[derive(Debug)]
    pub enum Error {
        Io(IoError),
        Network(ReqError),
    }

    impl From<ReqError> for Error {
        fn from(value: ReqError) -> Self {
            Self::Network(value)
        }
    }

    impl<W> CopyToIo<W> {
        pub fn new(wtr: W) -> Self {
            Self { to: wtr }
        }
    }

    impl<W: AsyncWrite + Unpin> CopyToIo<W> {
        pub async fn save<R>(mut self, model: &mut R) -> Result<(), Error>
        where
            R: AsyncRead + Unpin,
        {
            tokio::io::copy(model, &mut self.to)
                .await
                .map_err(Error::Io)?;
            self.to.shutdown().await.map_err(Error::Io)?;

            Ok(())
        }
    }

    impl<R, W> Handler<R> for CopyToIo<W>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        type Output = ();
        type Error = Error;

        async fn handle(self, model: &mut R) -> Result<Self::Output, Self::Error> {
            self.save(model).await
        }
    }
}

#[cfg(feature = "fs")]
pub use fs::{DEFAULT_PATH, Error as SaveToFileError, SaveToFile};
#[cfg(feature = "fs")]
mod fs {
    use super::Handler;
    use super::{CopyToIo, CopyToIoError};

    use reqwest::Error as ReqError;

    use tokio::fs::File;
    use tokio::fs::OpenOptions;
    use tokio::io::AsyncRead;
    use tokio::io::BufWriter;

    use std::io::Error as IoError;
    use std::path::Path;
    #[cfg(feature = "fs-temp")]
    use std::path::PathBuf;

    #[cfg(feature = "fs-temp")]
    use tempfile::TempDir;

    pub const DEFAULT_PATH: &str = "/usr/local/share/kytea/model.bin";

    pub struct SaveToFile<P> {
        path: P,
        opts: OpenOptions,
    }

    #[derive(Debug)]
    pub enum Error {
        OpenFile(IoError),
        CreateDir(IoError),
        CopyToFile(IoError),
        Network(ReqError),
    }

    impl From<ReqError> for Error {
        fn from(value: ReqError) -> Self {
            Self::Network(value)
        }
    }

    impl<P: AsRef<Path>> SaveToFile<P> {
        pub fn new(path: P) -> Self {
            let mut opts = File::options();
            opts.write(true).create(true).truncate(true).mode(0o644);

            Self { path, opts }
        }

        pub fn new_with(path: P, opts: OpenOptions) -> Self {
            Self { path, opts }
        }

        pub async fn save<R>(self, model: &mut R) -> Result<(), Error>
        where
            R: AsyncRead + Unpin,
        {
            let path = self.path.as_ref();
            if let Some(dir) = path.parent() {
                tokio::fs::create_dir_all(dir)
                    .await
                    .map_err(Error::CreateDir)?;
            }

            let file = self.opts.open(path).await.map_err(Error::OpenFile)?;
            let wtr = BufWriter::new(file);

            CopyToIo::new(wtr)
                .save(model)
                .await
                .map_err(|err| match err {
                    CopyToIoError::Io(e) => Error::CopyToFile(e),
                    CopyToIoError::Network(e) => Error::Network(e),
                })
        }
    }

    impl<R, P> Handler<R> for SaveToFile<P>
    where
        R: AsyncRead + Unpin,
        P: AsRef<Path>,
    {
        type Output = ();
        type Error = Error;

        async fn handle(self, model: &mut R) -> Result<Self::Output, Self::Error> {
            self.save(model).await
        }
    }

    #[cfg(feature = "fs-temp")]
    impl SaveToFile<PathBuf> {
        pub fn new_in(dir: &TempDir) -> Self {
            let path = dir.path().join("model.bin");
            Self::new(path)
        }

        pub fn new_in_with(dir: &TempDir, opts: OpenOptions) -> Self {
            let path = dir.path().join("model.bin");
            Self::new_with(path, opts)
        }
    }
}

pub use mem::{Error as SaveToVecError, SaveToVec};
mod mem {
    use super::Handler;

    use reqwest::Error as ReqError;

    use tokio::io::AsyncRead;
    use tokio::io::AsyncReadExt as _;

    use std::io::Error as IoError;

    pub struct SaveToVec;

    #[derive(Debug)]
    pub enum Error {
        Io(IoError),
        Network(ReqError),
    }

    impl Default for SaveToVec {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SaveToVec {
        pub fn new() -> Self {
            Self
        }

        pub async fn save<R>(self, model: &mut R) -> Result<Vec<u8>, Error>
        where
            R: AsyncRead + Unpin,
        {
            let mut buf = Vec::new();
            model.read_to_end(&mut buf).await.map_err(Error::Io)?;
            Ok(buf)
        }
    }

    impl<R> Handler<R> for SaveToVec
    where
        R: AsyncRead + Unpin,
    {
        type Output = Vec<u8>;
        type Error = Error;

        async fn handle(self, model: &mut R) -> Result<Self::Output, Self::Error> {
            self.save(model).await
        }
    }
}

mod errors {
    use super::{CopyToIoError, SaveToFileError, SaveToVecError};

    use std::error::Error as StdError;
    use std::fmt;

    fn fmt_network_err(e: &reqwest::Error, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "network error occurred while downloading the KyTea model: {e}",
        )
    }

    impl fmt::Display for CopyToIoError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Io(e) => write!(f, "failed to copy the KyTea model: {e}"),
                Self::Network(e) => fmt_network_err(e, f),
            }
        }
    }
    impl StdError for CopyToIoError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            match self {
                Self::Io(e) => Some(e),
                Self::Network(e) => Some(e),
            }
        }
    }

    impl fmt::Display for SaveToFileError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::OpenFile(e) => write!(f, "could not open file: {e}"),
                Self::CreateDir(e) => write!(f, "could not create directory: {e}"),
                Self::CopyToFile(e) => write!(f, "failed to copy the KyTea model to file: {e}"),
                Self::Network(e) => fmt_network_err(e, f),
            }
        }
    }
    impl StdError for SaveToFileError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            match self {
                Self::OpenFile(e) => Some(e),
                Self::CreateDir(e) => Some(e),
                Self::CopyToFile(e) => Some(e),
                Self::Network(e) => Some(e),
            }
        }
    }

    impl fmt::Display for SaveToVecError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Io(e) => write!(f, "failed to copy the KyTea model to memory: {e}"),
                Self::Network(e) => fmt_network_err(e, f),
            }
        }
    }
    impl StdError for SaveToVecError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            match self {
                Self::Io(e) => Some(e),
                Self::Network(e) => Some(e),
            }
        }
    }
}
