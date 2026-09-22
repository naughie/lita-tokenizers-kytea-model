use anyhow::Error;

use std::path::Path;
use std::path::PathBuf;

use clap::Parser;

async fn download(url: Option<&str>, dst: Option<&Path>) -> Result<(), Error> {
    use lita_tokenizers_kytea_model::download::{
        DEFAULT_PATH, SaveToFile, download_model, download_model_with_url,
    };

    let client = Default::default();

    let handler = if let Some(dst) = dst {
        SaveToFile::new(dst)
    } else {
        SaveToFile::new(Path::new(DEFAULT_PATH))
    };

    if let Some(url) = url {
        download_model_with_url(&client, url)
            .handle(handler)
            .await?;
    } else {
        download_model(&client).handle(handler).await?;
    };

    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// URL of the KyTea model [default: xxx]
    #[arg(short, long)]
    url: Option<String>,

    /// The output path of the KyTea model [default: /usr/local/share/kytea/model.bin]
    #[arg(short, long)]
    output: Option<PathBuf>,
}

async fn main_impl() -> Result<(), Error> {
    let args = Args::parse();

    download(args.url.as_deref(), args.output.as_deref()).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let signal = tokio::signal::ctrl_c();

    tokio::select! {
        res = main_impl() => res?,
        res = signal => res?,
    }

    Ok(())
}
