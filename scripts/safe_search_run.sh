#!/bin/sh
# Run one bounded search job and monitor the complete child process tree.
set -u

CPUS=${SAFE_SEARCH_CPUS:-0-3}
NICE_LEVEL=${SAFE_SEARCH_NICE:-10}
TIMEOUT_SEC=${SAFE_SEARCH_TIMEOUT_SEC:-300}
CPU_LIMIT_SEC=${CPU_LIMIT_SEC:-${SAFE_SEARCH_CPU_SEC:-600}}
MAX_RSS_KB=${MAX_RSS_KB:-${SAFE_SEARCH_MAX_RSS_KB:-1572864}}
VM_KB=${SAFE_SEARCH_VM_KB:-1572864}
VM_LIMIT_KB=${VM_LIMIT_KB:-${SAFE_SEARCH_VM_LIMIT_KB:-$VM_KB}}
LOCK_PATH=${SAFE_SEARCH_LOCK:-${TMPDIR:-/tmp}/shogi_ai_search.lock}
LOG_PATH=${SAFE_SEARCH_LOG:-search_quality/safe_search_run.log}

usage() {
    echo "usage: $0 [--cpus LIST] [--nice N] [--timeout SEC] [--cpu-sec SEC] [--max-rss-kb KB] [--vm-kb KB] [--vm-limit-kb KB] [--lock PATH] [--log PATH] -- command [args...]" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cpus) [ "$#" -ge 2 ] || usage; CPUS=$2; shift 2 ;;
        --nice) [ "$#" -ge 2 ] || usage; NICE_LEVEL=$2; shift 2 ;;
        --timeout) [ "$#" -ge 2 ] || usage; TIMEOUT_SEC=$2; shift 2 ;;
        --cpu-sec) [ "$#" -ge 2 ] || usage; CPU_LIMIT_SEC=$2; shift 2 ;;
        --max-rss-kb) [ "$#" -ge 2 ] || usage; MAX_RSS_KB=$2; shift 2 ;;
        --vm-kb) [ "$#" -ge 2 ] || usage; VM_KB=$2; shift 2 ;;
        --vm-limit-kb) [ "$#" -ge 2 ] || usage; VM_LIMIT_KB=$2; shift 2 ;;
        --lock) [ "$#" -ge 2 ] || usage; LOCK_PATH=$2; shift 2 ;;
        --log) [ "$#" -ge 2 ] || usage; LOG_PATH=$2; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[ "$#" -gt 0 ] || usage

mkdir -p "$(dirname "$LOG_PATH")"
lock_mode=mkdir
lock_dir=${LOCK_PATH}.d
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_PATH"
    if ! flock -n 9; then
        echo "safe_search_run: another job holds $LOCK_PATH" >&2
        exit 75
    fi
    lock_mode=flock
else
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "safe_search_run: another job holds $lock_dir" >&2
        exit 75
    fi
fi

child_pid=
reason=completed
max_rss=0
max_cpu=0
max_vmsize=0

tree_pids() {
    [ -n "$child_pid" ] || return 0
    ps -eo pid=,ppid= 2>/dev/null | awk -v root="$child_pid" '
        { parent[$1] = $2 }
        END {
            n = 1; list[1] = root; seen[root] = 1
            for (i = 1; i <= n; i++)
                for (pid in parent)
                    if (parent[pid] == list[i] && !seen[pid]) {
                        seen[pid] = 1; list[++n] = pid
                    }
            for (i = 1; i <= n; i++) print list[i]
        }'
}

cpu_to_seconds() {
    awk -v value="$1" 'BEGIN {
        n = split(value, a, ":")
        if (n == 3) printf "%.3f", a[1] * 3600 + a[2] * 60 + a[3]
        else if (n == 2) printf "%.3f", a[1] * 60 + a[2]
        else printf "%.3f", value + 0
    }'
}

tree_stats() {
    rss=0
    vsz=0
    cpu=0
    alive=0
    for pid in $(tree_pids); do
        line=$(ps -p "$pid" -o rss=,vsz=,cputime= 2>/dev/null || true)
        [ -n "$line" ] || continue
        set -- $line
        [ "$#" -ge 3 ] || continue
        rss=$((rss + ${1:-0}))
        vsz=$((vsz + ${2:-0}))
        cpu=$(awk -v a="$cpu" -v b="$(cpu_to_seconds "$3")" 'BEGIN { printf "%.3f", a + b }')
        alive=$((alive + 1))
    done
    TREE_RSS_KB=$rss
    TREE_VMSIZE_KB=$vsz
    TREE_CPU_SEC=$cpu
    TREE_ALIVE=$alive
}

kill_tree() {
    signal=$1
    pids=$(tree_pids)
    [ -n "$pids" ] || return 0
    for pid in $pids; do kill -"$signal" "$pid" 2>/dev/null || true; done
    if [ -n "$child_pid" ]; then kill -"$signal" -"$child_pid" 2>/dev/null || true; fi
}

cleanup() {
    if [ -n "$child_pid" ]; then kill_tree TERM; fi
    if [ "$lock_mode" = mkdir ]; then rmdir "$lock_dir" 2>/dev/null || true; fi
}
trap cleanup EXIT HUP INT TERM

{
    echo "safe_search_run start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cpus=$CPUS nice=$NICE_LEVEL timeout_sec=$TIMEOUT_SEC cpu_limit_sec=$CPU_LIMIT_SEC max_rss_kb=$MAX_RSS_KB vm_kb=$VM_KB vm_limit_kb=$VM_LIMIT_KB lock=$LOCK_PATH"
    echo "command=$*"
} >>"$LOG_PATH"

set +e
if command -v setsid >/dev/null 2>&1; then
    setsid sh -c '
        ulimit -v "$1" 2>/dev/null || true
        ulimit -t "$2" 2>/dev/null || true
        cpus=$3; nice_level=$4; shift 4
        if command -v taskset >/dev/null 2>&1; then
            exec taskset -c "$cpus" nice -n "$nice_level" "$@"
        else
            exec nice -n "$nice_level" "$@"
        fi
    ' sh "$VM_KB" "$CPU_LIMIT_SEC" "$CPUS" "$NICE_LEVEL" "$@" &
else
    (
        ulimit -v "$VM_KB" 2>/dev/null || true
        ulimit -t "$CPU_LIMIT_SEC" 2>/dev/null || true
        if command -v taskset >/dev/null 2>&1; then
            exec taskset -c "$CPUS" nice -n "$NICE_LEVEL" "$@"
        else
            exec nice -n "$NICE_LEVEL" "$@"
        fi
    ) &
fi
child_pid=$!
root_pgid=$(ps -p "$child_pid" -o pgid= 2>/dev/null | tr -d ' ')
start_epoch=$(date +%s)

while :; do
    tree_stats
    [ "$TREE_RSS_KB" -gt "$max_rss" ] && max_rss=$TREE_RSS_KB
    [ "$TREE_VMSIZE_KB" -gt "$max_vmsize" ] && max_vmsize=$TREE_VMSIZE_KB
    [ "$(awk -v a="$TREE_CPU_SEC" -v b="$max_cpu" 'BEGIN {print (a>b)?1:0}')" -eq 1 ] && max_cpu=$TREE_CPU_SEC
    now=$(date +%s)
    elapsed=$((now - start_epoch))
    if [ "$TREE_ALIVE" -eq 0 ]; then break; fi
    if [ "$TREE_RSS_KB" -gt "$MAX_RSS_KB" ]; then reason=rss_limit; kill_tree TERM; break; fi
    if [ "$TREE_VMSIZE_KB" -gt "$VM_LIMIT_KB" ]; then reason=vm_limit; kill_tree TERM; break; fi
    if [ "$(awk -v a="$TREE_CPU_SEC" -v b="$CPU_LIMIT_SEC" 'BEGIN {print (a>=b)?1:0}')" -eq 1 ]; then reason=cpu_limit; kill_tree TERM; break; fi
    if [ "$elapsed" -ge "$TIMEOUT_SEC" ]; then reason=timeout; kill_tree TERM; break; fi
    sleep 0.2
done

if [ "$reason" != completed ]; then
    sleep 0.3
    kill_tree KILL
fi
wait "$child_pid"
status=$?
set -e

# A child killed by its inherited RLIMIT_CPU may disappear before the polling
# loop can observe the final CPU sample. Preserve that limit in the reason.
if [ "$reason" = completed ] && [ "$status" -eq 137 ]; then
    reason=cpu_limit
    # RLIMIT_CPU can terminate the process between two polls; record the
    # enforced limit rather than falsely reporting zero observed CPU.
    max_cpu=$CPU_LIMIT_SEC
fi
if [ "$reason" = completed ] && [ "$status" -ne 0 ]; then reason=child_exit_$status; fi

{
    echo "safe_search_run end=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$status reason=$reason root_pid=$child_pid root_pgid=$root_pgid"
    echo "rss_limit_kb=$MAX_RSS_KB max_rss_kb=$max_rss vm_limit_kb=$VM_LIMIT_KB max_vmsize_kb=$max_vmsize cpu_limit_sec=$CPU_LIMIT_SEC max_cpu_sec=$max_cpu timeout_sec=$TIMEOUT_SEC"
} >>"$LOG_PATH"
exit "$status"
