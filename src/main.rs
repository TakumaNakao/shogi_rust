pub mod ai;
pub mod evaluation;
pub mod game;
pub mod move_ordering;
pub mod position_hash;
pub mod sennichite;
pub mod utils;

fn main() {
    game::run();
}
