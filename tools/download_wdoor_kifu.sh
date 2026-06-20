#!/usr/bin/env bash
set -euo pipefail

YEAR="${1:-2026}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/data/wdoor"
ARCHIVE_DIR="${DEST_DIR}/archive"
EXTRACT_ROOT="${DEST_DIR}/extract"
EXTRACT_DIR="${EXTRACT_ROOT}/${YEAR}"
ARCHIVE_PATH="${ARCHIVE_DIR}/wdoor${YEAR}.7z"
URL="https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor${YEAR}.7z"

mkdir -p "${ARCHIVE_DIR}" "${EXTRACT_ROOT}"

if [[ ! -s "${ARCHIVE_PATH}" ]]; then
  echo "Downloading ${URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "${ARCHIVE_PATH}.part" "${URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}.part" "${URL}"
  else
    echo "error: curl or wget is required" >&2
    exit 1
  fi
  mv "${ARCHIVE_PATH}.part" "${ARCHIVE_PATH}"
else
  echo "Archive already exists: ${ARCHIVE_PATH}"
fi

if ! command -v 7z >/dev/null 2>&1; then
  echo "error: 7z is required to extract ${ARCHIVE_PATH}" >&2
  echo "Install p7zip-full or p7zip, then rerun this script." >&2
  exit 1
fi

if [[ -d "${EXTRACT_DIR}" ]] && find "${EXTRACT_DIR}" -type f -name '*.csa' -print -quit | grep -q .; then
  echo "Extracted CSA files already exist: ${EXTRACT_DIR}"
else
  rm -rf "${EXTRACT_DIR}"
  mkdir -p "${EXTRACT_DIR}"
  echo "Extracting ${ARCHIVE_PATH} to ${EXTRACT_DIR}"
  7z x -y "-o${EXTRACT_DIR}" "${ARCHIVE_PATH}" >/dev/null
fi

CSA_COUNT="$(find "${EXTRACT_DIR}" -type f -name '*.csa' | wc -l)"
echo "Done."
echo "Archive: ${ARCHIVE_PATH}"
echo "Extracted: ${EXTRACT_DIR}"
echo "CSA files: ${CSA_COUNT}"
