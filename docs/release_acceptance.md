# Refactoring release acceptance

Phase 0--8のlocal gate後に、hosted runnerとWindows実機で行う最終確認である。code、weight、
benchmark条件の正本は[`refactoring_plan_v2.5.4.md`](refactoring_plan_v2.5.4.md)と
`benchmarks/baselines/v2.5.4_plus_master.json`にある。

## Candidate

- branch: `refactor/phase8-performance-cutover`
- implementation revision: `b6416e0`以降の文書commitを含むbranch head
- build: `cargo build --release --features halfkp64 --bin usi_engine`
- Linux reference SHA-256: `1705c3d88acd16b522b6e6e53d98d726a332086d020a526fe7c9611cbd17edc4`
- weight SHA-256: `91784d6f03f70580468a1509f968cbaaf598fbce084c22a739cb870344fb7c00`

OSやlinkerが異なるWindows binaryのhashはLinux referenceと一致しない。release workflowが記録した
artifact hashをそのOSの正本にする。

## Hosted CI

1. branchをremoteへpushし、Linux/Windows jobを実行する。
2. all-feature check、HalfKP-32/64 test、USI transcript、Clippy ratchet、inventory、fixed
   fingerprintがgreenであることを確認する。
3. failureが環境差でもgateを無効化せず、原因と再現commandをreportへ残す。

このworkspaceではremoteへのpushを行っていないため、hosted jobの結果は未記録である。

## Windows process smoke

PowerShellでrelease binaryとHalfKP-64 weightを同じmachineへ置き、次を確認する。

```powershell
@(
  "usi"
  "setoption name EvalFile value C:\path\to\policy_weights_halfkp64.binary"
  "isready"
  "usinewgame"
  "position startpos"
  "go depth 1"
  "quit"
) | .\usi_engine.exe
```

`usiok`、`readyok`、合法な`bestmove`が各一度だけ出て、processが終了すれば合格とする。

## ShogiHome

1. 新規engineとして`usi_engine.exe`を登録する。
2. HalfKP-64 weight pathを設定し、`isready`時にload errorがないことを確認する。
3. 平手初期局面からThreads=1で短い対局または検討を開始する。
4. `bestmove`が表示され、停止、連続検討、対局終了、engine終了がhangしないことを確認する。
5. Threads=0でも一局面を検索し、worker panicや二重`bestmove`がないことを確認する。

結果は日付、Windows version、CPU、binary/weight hash、ShogiHome versionを添えて`report/`へ
記録する。ここまで成功後にrelease tagを作る。tag作成、push、release公開は外部状態を変更するため、
release責任者が明示的に実行する。
