extern crate self as shogi_ai;

pub mod ai;
pub mod evaluation;
#[cfg(feature = "training-tools")]
pub mod halfkp_training;
pub mod move_ordering;
pub mod position_hash;
pub mod sennichite;
#[cfg(feature = "training-tools")]
pub mod training_data;
#[cfg(feature = "training-tools")]
pub mod training_tools;
pub mod usi_shogi;
pub mod utils;
