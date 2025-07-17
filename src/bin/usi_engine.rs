// main.rsやlib.rsでshogi_aiクレートが定義されていることを前提とします。
// 実際のクレート名に合わせて修正が必要な場合があります。
use shogi_ai::usi_shogi;

fn main() {
    usi_shogi::run_usi();
}
