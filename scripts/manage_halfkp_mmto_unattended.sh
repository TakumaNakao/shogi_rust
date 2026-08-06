#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SUPERVISOR="$ROOT_DIR/scripts/halfkp_mmto_unattended.py"
readonly DEFAULT_CONFIG="$ROOT_DIR/training/halfkp64-scratch-v1/unattended_mmto_20260806.json"
readonly UNIT="${HALFKP_MMTO_UNIT:-halfkp-mmto-unattended-20260806}"
readonly UNIT_FILE="$ROOT_DIR/scripts/systemd/halfkp-mmto-unattended-20260806.service"

CONFIG="${HALFKP_MMTO_CONFIG:-$DEFAULT_CONFIG}"
COMMAND=""

usage() {
  cat <<'EOF'
Usage:
  scripts/manage_halfkp_mmto_unattended.sh [--config PATH] COMMAND

Commands:
  validate  Validate the configuration and all run prerequisites.
  start     Validate and start the unattended supervisor as a user service.
  status    Show the user-service state and the supervisor status.
  logs      Show the latest user-service journal entries.
  stop      Request a clean stop, then stop the user service.
  resume    Clear a previous stop request and start from the last checkpoint.

Options:
  -c, --config PATH  Configuration file (default:
                     training/halfkp64-scratch-v1/unattended_mmto_20260806.json)
  -h, --help         Show this help.

Environment:
  HALFKP_MMTO_CONFIG  Alternative default configuration path.
  HALFKP_MMTO_UNIT    Alternative systemd unit name.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    -c|--config)
      (($# >= 2)) || die "$1 requires a path"
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    validate|start|status|logs|stop|resume)
      [[ -z "$COMMAND" ]] || die "only one command may be specified"
      COMMAND="$1"
      shift
      ;;
    *)
      die "unknown argument: $1 (use --help for usage)"
      ;;
  esac
done

[[ -n "$COMMAND" ]] || {
  usage >&2
  exit 2
}

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT_DIR/$CONFIG"
fi

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_common() {
  require_command python3
  [[ -f "$SUPERVISOR" ]] || die "supervisor not found: $SUPERVISOR"
  [[ -f "$CONFIG" ]] || die "configuration not found: $CONFIG"
}

require_systemd() {
  require_command systemctl
  require_command systemd-inhibit
  require_command loginctl
  systemctl --user show-environment >/dev/null 2>&1 ||
    die "the per-user systemd manager is unavailable"
}

require_persistent_service() {
  [[ -f "$UNIT_FILE" ]] || die "service unit not found: $UNIT_FILE"
  [[ "$(readlink -f "$ROOT_DIR")" == "$(readlink -f "$HOME/shogi_rust")" ]] ||
    die "persistent unit expects the repository at $HOME/shogi_rust"
  [[ "$(readlink -f "$CONFIG")" == "$(readlink -f "$DEFAULT_CONFIG")" ]] ||
    die "persistent start supports only the frozen default configuration"
  [[ "${UNIT%.service}" == "halfkp-mmto-unattended-20260806" ]] ||
    die "persistent start supports only unit halfkp-mmto-unattended-20260806"
  [[ "$(loginctl show-user "$USER" --property=Linger --value)" == "yes" ]] ||
    die "user lingering is disabled; run: loginctl enable-linger $USER"
}

service_state() {
  systemctl --user show "$UNIT" --property=ActiveState --value 2>/dev/null ||
    printf 'not-found\n'
}

service_is_running() {
  case "$(service_state)" in
    active|activating|reloading|deactivating)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

supervisor() {
  RUST_FONTCONFIG_DLOPEN=1 python3 -u "$SUPERVISOR" "$1" --config "$CONFIG"
}

do_validate() {
  require_common
  supervisor validate
}

do_start() {
  require_common
  require_systemd
  require_persistent_service
  if service_is_running; then
    die "$UNIT is already $(service_state); refusing a duplicate run"
  fi

  supervisor validate

  systemctl --user enable "$UNIT_FILE"
  systemctl --user daemon-reload
  systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
  systemctl --user start "$UNIT"

  printf 'unit: %s\nstate: %s\n' "$UNIT" "$(service_state)"
}

do_status() {
  require_command systemctl
  printf 'unit: %s\nstate: %s\n' "$UNIT" "$(service_state)"
  if [[ -f "$SUPERVISOR" && -f "$CONFIG" ]]; then
    supervisor status
  else
    printf 'warning: supervisor/config unavailable; service state only\n' >&2
  fi
}

do_logs() {
  require_command journalctl
  journalctl --user --unit="$UNIT" --no-pager --lines=200
}

do_stop() {
  require_command systemctl
  local request_failed=0
  if [[ ! -f "$SUPERVISOR" || ! -f "$CONFIG" ]] ||
     ! supervisor request-stop; then
    request_failed=1
    printf 'warning: could not write the clean-stop marker; stopping the unit anyway\n' >&2
  fi
  if service_is_running; then
    systemctl --user disable --now "$UNIT"
  else
    systemctl --user disable "$UNIT" >/dev/null 2>&1 || true
    if ((request_failed)); then
      printf '%s is not running; no stop marker could be recorded.\n' "$UNIT" >&2
    else
      printf '%s is not running; the stop request was recorded.\n' "$UNIT"
    fi
  fi
  return "$request_failed"
}

do_resume() {
  require_common
  require_systemd
  if service_is_running; then
    die "$UNIT is already $(service_state); refusing a duplicate run"
  fi
  supervisor clear-stop
  do_start
}

case "$COMMAND" in
  validate) do_validate ;;
  start) do_start ;;
  status) do_status ;;
  logs) do_logs ;;
  stop) do_stop ;;
  resume) do_resume ;;
esac
