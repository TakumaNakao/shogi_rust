#!/bin/sh
# Integration smoke for the bounded runner. It deliberately uses tiny limits.
set -eu

ROOT=${TMPDIR:-/tmp}/safe-search-smoke-$$
mkdir "$ROOT"
cleanup() {
    rm -rf "$ROOT"
}
trap cleanup EXIT HUP INT TERM

run() {
    name=$1
    shift
    scripts/safe_search_run.sh --lock "$ROOT/lock" --log "$ROOT/$name.log" "$@"
}

run normal -- sh -c 'sleep 0.3'
grep -q 'reason=completed' "$ROOT/normal.log"

run timeout --timeout 1 -- sh -c "sleep 30 & echo \$! > '$ROOT/timeout.pid'; wait" || true
grep -q 'reason=timeout' "$ROOT/timeout.log"
pid=$(cat "$ROOT/timeout.pid")
if kill -0 "$pid" 2>/dev/null; then
    echo "timeout smoke leaked descendant $pid" >&2
    exit 1
fi

run rss --max-rss-kb 1000 --timeout 5 -- sh -c 'sleep 5' || true
grep -q 'reason=rss_limit' "$ROOT/rss.log"

# A unique Perl allocation fixture exercises the VmSize watchdog independently
# of the inherited ulimit. The 200 MiB allocation exceeds a 64 MiB limit.
run vm --vm-kb 4194304 --vm-limit-kb 65536 --timeout 5 -- \
    perl -e '$x="x" x (200*1024*1024); sleep 5' || true
grep -q 'reason=vm_limit' "$ROOT/vm.log"

# The inherited ulimit can reject an allocation before the watchdog samples it;
# this must remain an explicit child-exit reason rather than `completed`.
run vm_child_exit --vm-kb 65536 --vm-limit-kb 4194304 --timeout 5 -- \
    perl -e '$x="x" x (200*1024*1024); sleep 1' || true
grep -q 'reason=child_exit_12' "$ROOT/vm_child_exit.log"

run vm_ok --vm-kb 4194304 --vm-limit-kb 524288 --timeout 5 -- \
    perl -e '$x="x" x (200*1024*1024); sleep 0.3'
grep -q 'reason=completed' "$ROOT/vm_ok.log"

run cpu --cpu-sec 1 --timeout 10 -- sh -c 'while :; do :; done' || true
grep -q 'reason=cpu_limit' "$ROOT/cpu.log"

scripts/safe_search_run.sh --lock "$ROOT/lock" --log "$ROOT/lock-first.log" -- sh -c 'sleep 1' &
first=$!
sleep 0.2
set +e
scripts/safe_search_run.sh --lock "$ROOT/lock" --log "$ROOT/lock-second.log" -- true
lock_status=$?
set -e
wait "$first"
[ "$lock_status" -eq 75 ]

echo "safe_search_run smoke: PASS"
