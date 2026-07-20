# ADR 0002: Content-addressed artifacts and staged storage

- Status: accepted
- Date: 2026-07-21
- Decision revision: Phase 7 branch

## Context

入力棋譜、派生dataset、checkpoint、採否reportが`data/`以下に混在し、完了markerやpath名だけで
古い生成物を再利用するpipelineがあった。拡張子の一括ignoreはgolden fixtureまで隠す。

## Decision

固定fixture、baseline、raw input、derived data、run artifact、reviewed reportを役割別に分ける。
生成物の同一性はcontent hashとschema化manifestで判断する。詳細は
[`../artifact_policy.md`](../artifact_policy.md)を正本とする。既存pathは一括移動せず互換性を
保ち、新規runから段階移行する。

## Consequences

stale artifactの誤利用を検出でき、同じ入力と設定からstageを再構築できる。既存の大容量data
移動による運用停止は避けられる一方、移行期間は旧pathと標準pathが併存する。
