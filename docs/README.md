# 開発ドキュメント索引

このディレクトリには、現在の実装説明、リファクタリング計画、評価関数学習、過去の調査記録が含まれる。
大規模な変更へ着手するときは、まず次の2文書を確認する。

1. [`refactoring_plan_v2.5.4.md`](refactoring_plan_v2.5.4.md)  
   目標アーキテクチャ、Phase、PR順、検証ゲート、完了条件を定義する実行計画。
2. [`refactoring_handoff_v2.5.4.md`](refactoring_handoff_v2.5.4.md)  
   v2.5.4以降も保存すべき探索、USI、HalfKP、教師データの契約。
3. [`toolchain_policy.md`](toolchain_policy.md)
   Rust stable、MSRV宣言条件、Clippy warning ratchet、release再現性の運用方針。
4. [`binaries.md`](binaries.md)
   production、training、benchmark、research binaryの完全なinventoryとbuild feature。

## 現行実装

- [`ai.md`](ai.md): 探索module
- [`evaluation.md`](evaluation.md): 評価module
- [`parallel_search.md`](parallel_search.md): 並列探索
- [`sennichite.md`](sennichite.md): 千日手
- [`usi_shogi.md`](usi_shogi.md): USI
- [`position_hash.md`](position_hash.md): position hash
- [`move_ordering.md`](move_ordering.md): 手順順序
- [`utils.md`](utils.md): 共通utility
- [`binaries.md`](binaries.md): binary inventoryと安定性区分
- [`artifact_policy.md`](artifact_policy.md): fixture、dataset、run、reportの配置とmanifest規約

## Architecture decision records

- [`adr/0001-explicit-binary-targets.md`](adr/0001-explicit-binary-targets.md)
- [`adr/0002-artifact-layout.md`](adr/0002-artifact-layout.md)

## HalfKP・学習

- [`halfkp64_training.md`](halfkp64_training.md)
- [`halfkp_performance_training_design.md`](halfkp_performance_training_design.md)
- [`halfkp_training_start_gate.md`](halfkp_training_start_gate.md)
- [`halfkp_kpp_projection_design.md`](halfkp_kpp_projection_design.md)
- [`kpp_learn.md`](kpp_learn.md)
- [`kpp_self_learn.md`](kpp_self_learn.md)
- [`mmto_tree_training.md`](mmto_tree_training.md)
- [`mmto_lite_validation.md`](mmto_lite_validation.md)

## 調査記録

- [`v2.5.2_parallel_mate_bug_investigation.md`](v2.5.2_parallel_mate_bug_investigation.md)
- [`../report/README.md`](../report/README.md): 性能、棋力、学習実験reportのmetadataと検索方法

## 文書の扱い

- 実装と一致している文書を`current`とする。
- 古い文書は削除せず、`historical`または`superseded`であることを追記する。
- format、CLI、crate境界、探索契約を変更するPRは、対応文書も同じPRで更新する。
- 将来計画は未実装の内容を「目標」「提案」「未決定」と明示する。
