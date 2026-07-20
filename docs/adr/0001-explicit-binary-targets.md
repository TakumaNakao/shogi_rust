# ADR 0001: Explicit binary target profiles

- Status: accepted
- Date: 2026-07-21
- Decision revision: Phase 7 branch

## Context

Cargoの自動検出により、production engineをbuildするだけでも多数の研究targetが正式機能に
見え、追加fileが暗黙に配布対象になっていた。CLIの利用範囲と安定性も判別できなかった。

## Decision

`autobins = false`とし、全targetを`Cargo.toml`へ列挙する。`usi_engine`だけをdefaultの
production targetとし、他を`training-tools`、`benchmark-tools`、`research-tools`で
opt-inにする。既存target名は維持し、物理file移動は行わない。分類は
[`../binaries.md`](../binaries.md)を正本とする。

Phase 8 cutoverでsupported trainingとbenchmarkの実装を`src/training_tools/`と
`src/benchmark_tools/`へ移し、既存`src/bin` pathはthin compatibility entryとして維持した。
research targetはCLI/APIが安定していないため、引き続き単体binaryとして隔離する。

## Consequences

通常の`cargo build --release`はengineだけをbuildし、training/research固有dependencyも
解決しない。tool利用者は対応featureを指定する必要があるため、repository内のscript、
CI、release workflowを同時に更新する。将来CLIを改名するときはthin shimを最低1 release
維持する。
