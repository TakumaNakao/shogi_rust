#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# font-kit can load fontconfig dynamically on build hosts without pkg-config.
export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

# shogi_lib has no accepted production warnings.
cargo clippy -p shogi_lib --lib -- -D warnings

# The root library contains the engine, evaluation and USI production paths.
# These lint classes are existing debt recorded on 2026-07-20 with Rust 1.95.0.
# Keep this list explicit: remove an allowance when its final occurrence is fixed.
cargo clippy -p shogi_ai --lib --features halfkp64 -- \
  -D warnings \
  -A clippy::collapsible_if \
  -A clippy::get_first \
  -A clippy::iter_nth_zero \
  -A clippy::large_enum_variant \
  -A clippy::needless_borrows_for_generic_args \
  -A clippy::needless_range_loop \
  -A clippy::new_without_default \
  -A clippy::redundant_pattern_matching \
  -A clippy::too_many_arguments \
  -A clippy::trim_split_whitespace \
  -A clippy::unwrap_or_default
