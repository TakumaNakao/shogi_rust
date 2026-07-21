use shogi_ai::evaluation::{NNUE_NUM_FEATURES, NNUE_NUM_KING_BUCKETS};
use shogi_ai::utils::parse_usi_move_for_color;
use shogi_lib::Position;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(15);
const EXIT_TIMEOUT: Duration = Duration::from_secs(5);
const TERMINAL_SFEN: &str = "4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1";
const FORCED_MATE_POSITION: &str = "startpos moves \
    7g7f 8c8d 2g2f 4a3b 2f2e 8d8e 8h7g 3c3d 7i6h 2b7g+ 6h7g 3a2b \
    3i4h 2b3c 6i7h 7a7b 4g4f 5a4b 4h4g 9c9d 9g9f 6c6d 3g3f 1c1d \
    1g1f 7c7d 2i3g 8a7c 6g6f 7b6c 4g5f 6a6b 2h2i 6c5d 4i4h 8b8a \
    3g4e 3c4d 2e2d 2c2d 2i2d P*2c 2d2h B*1c P*2d 1c2d 4h4g 2a1c \
    B*3g 4d5e 5f5e 5d5e 1f1e 1d1e 1i1e S*1g 2h2g 1g1h 2g2h 4c4d \
    2h1h 4d4e 4f4e P*4f 4g5f 5e5f 5g5f G*2g S*8b 8a8b S*7a 2g1h \
    7a8b R*3i 5i6h 4f4g+ 3f3e";
static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TinyWeight {
    path: PathBuf,
}

impl TinyWeight {
    fn create() -> Self {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-usi-transcript-{}-{sequence}.binary",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create Tiny NNUE fixture");
        file.write_all(b"TNNUE001").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(NNUE_NUM_FEATURES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(NNUE_NUM_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();
        for _ in 0..NNUE_NUM_FEATURES + NNUE_NUM_KING_BUCKETS {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.5f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.25f32.to_le_bytes()).unwrap();
        file.flush().expect("flush Tiny NNUE fixture");
        Self { path }
    }
}

impl Drop for TinyWeight {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

struct EngineProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout_lines: Receiver<String>,
    stdout_reader: Option<thread::JoinHandle<()>>,
}

impl EngineProcess {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_usi_engine"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn USI engine");
        let stdin = child.stdin.take().expect("child stdin");
        let stdout = child.stdout.take().expect("child stdout");
        let (sender, stdout_lines) = mpsc::channel();
        let stdout_reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            stdout_lines,
            stdout_reader: Some(stdout_reader),
        }
    }

    fn send(&mut self, command: &str) {
        let stdin = self.stdin.as_mut().expect("engine stdin is open");
        writeln!(stdin, "{command}").expect("write USI command");
        stdin.flush().expect("flush USI command");
    }

    fn read_until(&self, expected: &str) -> Vec<String> {
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        let mut lines = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.stdout_lines.recv_timeout(remaining) {
                Ok(line) => {
                    let matches = line == expected || line.starts_with(expected);
                    lines.push(line);
                    if matches {
                        return lines;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!("timed out waiting for {expected:?}; received {lines:#?}")
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("engine exited while waiting for {expected:?}; received {lines:#?}")
                }
            }
        }
    }

    fn handshake(&mut self) -> Vec<String> {
        self.send("usi");
        self.read_until("usiok")
    }

    fn configure_evaluator(&mut self, weight: &TinyWeight) {
        self.send(&format!(
            "setoption name EvalFile value {}",
            weight.path.display()
        ));
        self.send("setoption name Threads value 1");
        self.send("setoption name MaxDepth value 100");
        self.send("isready");
        let lines = self.read_until("readyok");
        assert_eq!(vec!["readyok"], lines);
    }

    fn expect_bestmove(&self) -> (String, Vec<String>) {
        let lines = self.read_until("bestmove ");
        let bestmoves: Vec<_> = lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .collect();
        assert_eq!(1, bestmoves.len(), "received {lines:#?}");
        let bestmove = bestmoves[0]
            .strip_prefix("bestmove ")
            .expect("bestmove prefix")
            .to_string();
        (bestmove, lines)
    }

    fn quit(mut self) {
        self.send("quit");
        self.stdin.take();
        let deadline = Instant::now() + EXIT_TIMEOUT;
        loop {
            match self.child.try_wait().expect("poll USI engine") {
                Some(status) => {
                    assert!(status.success(), "USI engine exited with {status}");
                    break;
                }
                None if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                None => {
                    self.child.kill().ok();
                    panic!("USI engine did not exit after quit");
                }
            }
        }
        if let Some(reader) = self.stdout_reader.take() {
            reader.join().expect("join stdout reader");
        }
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            self.child.kill().ok();
            self.child.wait().ok();
        }
        if let Some(reader) = self.stdout_reader.take() {
            reader.join().ok();
        }
    }
}

fn assert_legal_startpos_move(bestmove: &str) {
    assert_ne!("resign", bestmove);
    let position = Position::default();
    let parsed = parse_usi_move_for_color(bestmove, position.side_to_move())
        .unwrap_or_else(|| panic!("invalid USI bestmove: {bestmove}"));
    assert!(
        position.legal_moves().contains(&parsed),
        "illegal startpos bestmove: {bestmove}"
    );
}

#[test]
fn usi_handshake_and_missing_evaluator_emit_one_response() {
    let mut engine = EngineProcess::spawn();
    let handshake = engine.handshake();
    assert!(handshake.iter().any(|line| line == "id name Shogi AI"));
    assert!(handshake.iter().any(|line| line == "id author Gemini"));
    assert!(handshake
        .iter()
        .any(|line| line.starts_with("option name EvalFile ")));
    assert!(handshake
        .iter()
        .any(|line| line.starts_with("option name Threads ")));

    engine.send("isready");
    assert_eq!(vec!["readyok"], engine.read_until("readyok"));
    engine.send("position startpos");
    engine.send("go depth 1");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_eq!("resign", bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("info string Error: Evaluation file is not set"))
            .count()
    );
    engine.quit();
}

#[test]
fn usi_normal_and_terminal_search_each_emit_one_bestmove() {
    let weight = TinyWeight::create();
    let mut engine = EngineProcess::spawn();
    engine.handshake();
    engine.configure_evaluator(&weight);

    engine.send("position startpos");
    engine.send("go depth 1");
    let (bestmove, first_lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&bestmove);
    assert_eq!(
        1,
        first_lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );

    let terminal =
        shogi_ai::utils::position_from_sfen_or_usi(TERMINAL_SFEN).expect("valid terminal fixture");
    assert!(terminal.legal_moves().is_empty());
    engine.send(&format!("position sfen {TERMINAL_SFEN}"));
    engine.send("go depth 1");
    let (bestmove, terminal_lines) = engine.expect_bestmove();
    assert_eq!("resign", bestmove);
    assert_eq!(
        1,
        terminal_lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );
    engine.quit();
}

#[test]
fn usi_immediate_stop_returns_one_legal_fallback_and_quits_cleanly() {
    let weight = TinyWeight::create();
    let mut engine = EngineProcess::spawn();
    engine.handshake();
    engine.configure_evaluator(&weight);

    engine.send("position startpos");
    engine.send("go infinite");
    engine.send("stop");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );
    engine.quit();
}

#[test]
fn usi_stop_before_and_during_search_each_return_one_legal_move() {
    let weight = TinyWeight::create();
    let mut engine = EngineProcess::spawn();
    engine.handshake();
    engine.configure_evaluator(&weight);

    engine.send("stop");
    engine.send("position startpos");
    engine.send("go depth 1");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );

    engine.send("go infinite");
    let progress = engine.read_until("info depth ");
    assert!(progress.iter().any(|line| line.starts_with("info depth ")));
    engine.send("stop");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );

    engine.send("usinewgame");
    engine.send("go depth 1");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );
    engine.quit();
}

#[test]
fn usi_forced_mate_keeps_the_known_mating_move() {
    let weight = TinyWeight::create();
    let mut engine = EngineProcess::spawn();
    engine.handshake();
    engine.configure_evaluator(&weight);

    engine.send(&format!("position {FORCED_MATE_POSITION}"));
    engine.send("go depth 3");
    let (bestmove, lines) = engine.expect_bestmove();
    assert_eq!("S*5g", bestmove);
    assert_eq!(
        1,
        lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );
    engine.quit();
}

#[test]
fn usi_replaces_search_jobs_serially_and_joins_on_quit() {
    let weight = TinyWeight::create();
    let mut engine = EngineProcess::spawn();
    engine.handshake();
    engine.configure_evaluator(&weight);
    engine.send("position startpos");

    engine.send("go infinite");
    engine.send("go depth 1");
    let (first, first_lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&first);
    assert_eq!(
        1,
        first_lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );
    let (second, second_lines) = engine.expect_bestmove();
    assert_legal_startpos_move(&second);
    assert_eq!(
        1,
        second_lines
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count()
    );

    engine.send("go infinite");
    engine.quit();
}
