use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shogi_core::{Color, PieceKind, Position, Square};

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
        let mut hash = 0;

        // Board pieces
        for sq in Square::all() {
            if let Some(piece) = position.piece_at(sq) {
                let sq_idx = (sq.index() - 1) as usize; // Correctly map 1..=81 to 0..=80
                let piece_idx = PIECE_KINDS.iter().position(|&k| k == piece.piece_kind()).unwrap();
                let color_idx = color_to_index(piece.color());
                hash ^= ZOBRIST_KEYS.board[piece_idx][sq_idx][color_idx];
            }
        }

        // Hand pieces
        for color in [Color::Black, Color::White] {
            let hand = position.hand_of_a_player(color);
            for (piece_idx, &kind) in HAND_PIECE_KINDS.iter().enumerate() {
                if let Some(count) = hand.count(kind) {
                    if count > 0 {
                        // XOR with the key for the current count of this piece
                        hash ^= ZOBRIST_KEYS.hand[piece_idx][count as usize][color_to_index(color)];
                    }
                }
            }
        }

        // Side to move
        if position.side_to_move() == Color::Black {
            hash ^= ZOBRIST_KEYS.side_to_move;
        }

        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move, PartialPosition};

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
        pos.make_move(mv).unwrap();
        
        let hash_after = PositionHasher::calculate_hash(&pos);
        assert_ne!(hash_before, hash_after, "Hash should change after a move");
    }

    #[test]
    fn test_hash_changes_on_side_to_move() {
        let mut partial_pos = PartialPosition::startpos();
        let pos_black_turn = Position::arbitrary_position(partial_pos.clone());
        
        partial_pos.side_to_move_set(Color::White);
        let pos_white_turn = Position::arbitrary_position(partial_pos);

        let hash_black = PositionHasher::calculate_hash(&pos_black_turn);
        let hash_white = PositionHasher::calculate_hash(&pos_white_turn);

        assert_ne!(hash_black, hash_white, "Hash should change when side to move changes");
    }

    #[test]
    fn test_hash_changes_on_hand_piece() {
        let mut pos = Position::default();
        let hash_before = PositionHasher::calculate_hash(&pos);

        // Simulate capturing a pawn
        let mut partial_pos = pos.inner().clone();
        let black_hand = partial_pos.hand_of_a_player_mut(Color::Black);
        *black_hand = black_hand.added(PieceKind::Pawn).unwrap();
        let pos_after = Position::arbitrary_position(partial_pos);

        let hash_after = PositionHasher::calculate_hash(&pos_after);
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
            pos.make_move(*mv).unwrap();
        }

        let final_hash = PositionHasher::calculate_hash(&pos);

        assert_eq!(initial_hash, final_hash, "Hash must be the same after a sequence of moves that leads to the same position");
    }
}
