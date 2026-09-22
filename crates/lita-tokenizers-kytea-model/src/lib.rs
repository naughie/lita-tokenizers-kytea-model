#[cfg(feature = "download")]
pub mod download;
#[cfg(feature = "download")]
pub use download::{CopyToIo, SaveToFile, SaveToVec};
#[cfg(feature = "download")]
pub use download::{download_model, download_model_with_url};
