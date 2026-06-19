# Search Quality Fixed Sets

These SFEN sets are small regression/probe inputs for search experiments.

They are not training data. Use them before self-play or benchmark games to catch repeated failure modes cheaply.

## loss_in_check_low_reply.sfen

Positions exported from benchmark score/result mismatches. Most are in-check terminal-adjacent positions with few legal evasions.

Suggested probe:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/position_probe \
  --weights policy_weights_v2.1.0.binary \
  --positions data/search_quality/loss_in_check_low_reply.sfen \
  --depth 5 \
  --summary
```

## taildrop_root_rescue.sfen

Positions exported from largest tail evaluation drops and root rescue candidates.

Suggested probe:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/position_probe \
  --weights policy_weights_v2.1.0.binary \
  --positions data/search_quality/taildrop_root_rescue.sfen \
  --depth 4 \
  --root-top 8 \
  --root-search-limit 32 \
  --summary
```
