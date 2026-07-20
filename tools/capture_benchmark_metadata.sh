#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/capture_benchmark_metadata.sh LABEL=PATH... [-- COMMAND ARG...]

Example:
  tools/capture_benchmark_metadata.sh \
    binary=target/release/search_profile \
    weight=policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
    positions=taya36.sfen \
    -- target/release/search_profile --samples 16 --depth 5 --threads 1

The script writes JSON to stdout. It does not run COMMAND and does not modify
the repository.
EOF
}

json_string() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '"%s"' "$value"
}

sha256_file() {
  local path=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    printf 'sha256-unavailable'
  fi
}

file_size() {
  local path=$1
  if stat -c '%s' "$path" >/dev/null 2>&1; then
    stat -c '%s' "$path"
  elif stat -f '%z' "$path" >/dev/null 2>&1; then
    stat -f '%z' "$path"
  else
    wc -c <"$path" | tr -d ' '
  fi
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

file_specs=()
command_args=()
parsing_command=false
for argument in "$@"; do
  if [[ $parsing_command == true ]]; then
    command_args+=("$argument")
  elif [[ $argument == "--" ]]; then
    parsing_command=true
  else
    if [[ $argument != *=* ]]; then
      printf 'expected LABEL=PATH, got: %s\n' "$argument" >&2
      usage >&2
      exit 2
    fi
    label=${argument%%=*}
    path=${argument#*=}
    if [[ ! $label =~ ^[A-Za-z0-9_.-]+$ ]]; then
      printf 'invalid file label: %s\n' "$label" >&2
      exit 2
    fi
    if [[ ! -f $path ]]; then
      printf 'file does not exist: %s\n' "$path" >&2
      exit 2
    fi
    file_specs+=("$label=$path")
  fi
done

repo_root=$(git rev-parse --show-toplevel)
revision=$(git -C "$repo_root" rev-parse HEAD)
describe=$(git -C "$repo_root" describe --tags --always --dirty)
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
  dirty=true
else
  dirty=false
fi

rustc_version=$(rustc --version)
rustc_host=$(rustc -vV | sed -n 's/^host: //p')
cargo_version=$(cargo --version)
os=$(uname -s)
kernel=$(uname -r)
arch=$(uname -m)

cpu_model=unknown
cpu_features=unknown
if [[ -r /proc/cpuinfo ]]; then
  cpu_model=$(awk -F ': ' '/^model name/{print $2; exit}' /proc/cpuinfo)
  cpu_features=$(awk '/^flags/{sub(/^[^:]*:[[:space:]]*/, ""); print; exit}' /proc/cpuinfo)
elif command -v sysctl >/dev/null 2>&1; then
  cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf 'unknown')
  cpu_features=$(sysctl -n machdep.cpu.features 2>/dev/null || printf 'unknown')
fi

printf '{\n'
printf '  "schema_version": 1,\n'
printf '  "captured_at_utc": '
json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf ',\n'
printf '  "git": {\n'
printf '    "revision": '
json_string "$revision"
printf ',\n'
printf '    "describe": '
json_string "$describe"
printf ',\n'
printf '    "dirty": %s\n' "$dirty"
printf '  },\n'
printf '  "toolchain": {\n'
printf '    "rustc": '
json_string "$rustc_version"
printf ',\n'
printf '    "cargo": '
json_string "$cargo_version"
printf ',\n'
printf '    "host": '
json_string "$rustc_host"
printf '\n'
printf '  },\n'
printf '  "machine": {\n'
printf '    "os": '
json_string "$os"
printf ',\n'
printf '    "kernel": '
json_string "$kernel"
printf ',\n'
printf '    "arch": '
json_string "$arch"
printf ',\n'
printf '    "cpu_model": '
json_string "$cpu_model"
printf ',\n'
printf '    "cpu_features": '
json_string "$cpu_features"
printf '\n'
printf '  },\n'
printf '  "files": ['
if ((${#file_specs[@]} > 0)); then
  printf '\n'
fi
for index in "${!file_specs[@]}"; do
  spec=${file_specs[$index]}
  label=${spec%%=*}
  path=${spec#*=}
  printf '    {\n'
  printf '      "label": '
  json_string "$label"
  printf ',\n'
  printf '      "path": '
  json_string "$path"
  printf ',\n'
  printf '      "bytes": %s,\n' "$(file_size "$path")"
  printf '      "sha256": '
  json_string "$(sha256_file "$path")"
  printf '\n'
  if ((index + 1 == ${#file_specs[@]})); then
    printf '    }\n'
  else
    printf '    },\n'
  fi
done
printf '  ],\n'
printf '  "command": ['
if ((${#command_args[@]} > 0)); then
  printf '\n'
fi
for index in "${!command_args[@]}"; do
  printf '    '
  json_string "${command_args[$index]}"
  if ((index + 1 == ${#command_args[@]})); then
    printf '\n'
  else
    printf ',\n'
  fi
done
printf '  ]\n'
printf '}\n'
