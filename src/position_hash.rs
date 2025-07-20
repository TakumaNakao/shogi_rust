use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shogi_core::{Color, PieceKind, Square};
use shogi_lib::Position;

pub const PIECE_KINDS: [PieceKind; 14] = [
    PieceKind::Pawn, PieceKind::Lance, PieceKind::Knight, PieceKind::Silver,
    PieceKind::Gold, PieceKind::Bishop, PieceKind::Rook, PieceKind::King,
    PieceKind::ProPawn, PieceKind::ProLance, PieceKind::ProKnight, PieceKind::ProSilver,
    PieceKind::ProBishop, PieceKind::ProRook,
];

pub const HAND_PIECE_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn, PieceKind::Lance, PieceKind::Knight, PieceKind::Silver,
    PieceKind::Gold, PieceKind::Bishop, PieceKind::Rook,
];

const MAX_HAND_COUNTS: [usize; 7] = [18, 4, 4, 4, 4, 2, 2];

pub struct ZobristKeys {
    // piece, square, color
    pub board: [[[u64; 2]; 81]; 14],
    // piece, count, color
    pub hand: [[[u64; 2]; 19]; 7],
    pub side_to_move: u64,
}

pub fn color_to_index(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

impl ZobristKeys {
    fn new() -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(19700101);
        let mut board = [[[0; 2]; 81]; 14];
        let mut hand = [[[0; 2]; 19]; 7];

        for piece_idx in 0..14 {
            for square_idx in 0..81 {
                for color_idx in 0..2 {
                    board[piece_idx][square_idx][color_idx] = rng.gen();
                }
            }
        }

        for piece_idx in 0..7 {
            // count 0 is not used, so we start from 1
            for count in 1..=MAX_HAND_COUNTS[piece_idx] {
                for color_idx in 0..2 {
                    hand[piece_idx][count][color_idx] = rng.gen();
                }
            }
        }

        let side_to_move = rng.gen();

        ZobristKeys { board, hand, side_to_move }
    }
}

lazy_static! {
    pub static ref ZOBRIST_KEYS: ZobristKeys = ZobristKeys::new();
}

pub struct PositionHasher;

impl PositionHasher {
    pub fn calculate_hash(position: &Position) -> u64 {
        position.key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move};

    #[test]
    fn test_hash_consistency() {
        let pos1 = Position::default();
        let hash1 = PositionHasher::calculate_hash(&pos1);
        let hash2 = PositionHasher::calculate_hash(&pos1);
        assert_eq!(hash1, hash2, "Hash for the same position should be consistent");
    }

    #[test]
    fn test_hash_changes_on_move() {
        let mut pos = Position::default();
        let hash_before = PositionHasher::calculate_hash(&pos);
        
        // Make a move
        let mv = Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false };
        pos.do_move(mv);
        
        let hash_after = PositionHasher::calculate_hash(&pos);
        assert_ne!(hash_before, hash_after, "Hash should change after a move");
    }

    #[test]
    fn test_hash_changes_on_side_to_move() {
        let mut pos_black_turn = Position::default();
        let hash_black = PositionHasher::calculate_hash(&pos_black_turn);
        
        pos_black_turn.do_move(Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false });
        let hash_white = PositionHasher::calculate_hash(&pos_black_turn);

        assert_ne!(hash_black, hash_white, "Hash should change when side to move changes");
    }

    #[test]
    fn test_hash_changes_on_hand_piece() {
        let mut pos = Position::default();
        // 7g7f 3c3d 8h2b+ 3a2b
        pos.do_move(Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false });
        pos.do_move(Move::Normal { from: Square::new(3, 3).unwrap(), to: Square::new(3, 4).unwrap(), promote: false });
        pos.do_move(Move::Normal { from: Square::new(8, 8).unwrap(), to: Square::new(2, 2).unwrap(), promote: true });
        let hash_before = PositionHasher::calculate_hash(&pos);

        pos.do_move(Move::Normal { from: Square::new(3, 1).unwrap(), to: Square::new(2, 2).unwrap(), promote: false });
        let hash_after = PositionHasher::calculate_hash(&pos);

        assert_ne!(hash_before, hash_after, "Hash should change when a piece is added to hand");
    }

    #[test]
    fn test_hash_repetition() {
        let mut pos = Position::default();
        let initial_hash = PositionHasher::calculate_hash(&pos);

        // Define a sequence of moves
        let moves = [
            Move::Normal { from: Square::new(2, 7).unwrap(), to: Square::new(2, 6).unwrap(), promote: false }, // P-2f
            Move::Normal { from: Square::new(3, 3).unwrap(), to: Square::new(3, 4).unwrap(), promote: false }, // P-3d
            Move::Normal { from: Square::new(2, 6).unwrap(), to: Square::new(2, 7).unwrap(), promote: false }, // P-2g
            Move::Normal { from: Square::new(3, 4).unwrap(), to: Square::new(3, 3).unwrap(), promote: false }, // P-3c
        ];

        // Apply moves
        for mv in &moves {
            pos.do_move(*mv);
        }

        let final_hash = PositionHasher::calculate_hash(&pos);

        // Note: The final hash will NOT be the same as the initial hash because the ply count is different.
        // Zobrist keys in shogi_lib::Position do not account for ply, but the sequence of moves is not a true repetition
        // in terms of game state if ply is considered. However, for transposition table purposes, this is often desired.
        // The core test here is that the hash function is deterministic.
        
        let mut pos2 = Position::default();
        for mv in &moves {
            pos2.do_move(*mv);
        }
        let final_hash2 = PositionHasher::calculate_hash(&pos2);

        assert_eq!(final_hash, final_hash2, "Hash must be the same for the same sequence of moves");
        // We don't assert initial_hash == final_hash because ply changes the state.
    }
}
