use shogi_core::{Color, PieceKind, Square};
use shogi_lib::Position;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpeningFeatures {
    pub rook: [u8; 2],
    pub king_zone: [u8; 2],
    pub pawn_advance_band: [u8; 2],
}

impl OpeningFeatures {
    pub fn stratum_id(self) -> u16 {
        self.rook[0] as u16
            | (self.rook[1] as u16) << 2
            | (self.king_zone[0] as u16) << 4
            | (self.king_zone[1] as u16) << 6
            | (self.pawn_advance_band[0] as u16) << 8
            | (self.pawn_advance_band[1] as u16) << 10
    }

    pub fn alias(self) -> String {
        format!(
            "b:{}-{}-p{}__w:{}-{}-p{}",
            rook_alias(self.rook[0]),
            king_alias(self.king_zone[0]),
            self.pawn_advance_band[0],
            rook_alias(self.rook[1]),
            king_alias(self.king_zone[1]),
            self.pawn_advance_band[1]
        )
    }
}

pub fn classify_opening(position: &Position) -> OpeningFeatures {
    let mut rook = [0u8; 2];
    let mut king_zone = [1u8; 2];
    let mut pawn_advances = [0usize; 2];
    for index in 1..=81 {
        let square = Square::from_u8(index).expect("board square");
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let side = usize::from(piece.color() == Color::White);
        let oriented = if piece.color() == Color::Black {
            square
        } else {
            square.flip()
        };
        match piece.piece_kind() {
            PieceKind::Rook => {
                rook[side] = match oriented.file() {
                    2 => 1,
                    5..=9 => 2,
                    _ => 3,
                };
            }
            PieceKind::King => {
                king_zone[side] = match oriented.file() {
                    1..=3 => 0,
                    4..=6 => 1,
                    _ => 2,
                };
            }
            PieceKind::Pawn if oriented.rank() < 7 => pawn_advances[side] += 1,
            _ => {}
        }
    }
    OpeningFeatures {
        rook,
        king_zone,
        pawn_advance_band: pawn_advances.map(|count| match count {
            0 => 0,
            1..=2 => 1,
            _ => 2,
        }),
    }
}

fn rook_alias(value: u8) -> &'static str {
    match value {
        0 => "none",
        1 => "home",
        2 => "ranging",
        _ => "shifted",
    }
}

fn king_alias(value: u8) -> &'static str {
    match value {
        0 => "left",
        1 => "center",
        _ => "right",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_has_stable_opening_stratum() {
        let features = classify_opening(&Position::default());
        assert_eq!([1, 1], features.rook);
        assert_eq!([1, 1], features.king_zone);
        assert_eq!([0, 0], features.pawn_advance_band);
        assert_eq!(85, features.stratum_id());
    }
}
