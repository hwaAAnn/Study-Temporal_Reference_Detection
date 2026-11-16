#!/usr/bin/env bash
# Pod tarball for 12: the committed tree of this branch (LF line endings, as 06) plus data/12_titles.jsonl, which
# git ignores. Refuses to pack while scripts/ or src/ have uncommitted changes, so the pod runs committed code.
# usage (from this worktree): bash scripts/pack_12.sh <out.tar.gz>
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output .tar.gz}
[ -s data/12_titles.jsonl ] || { echo "run scripts/build_12_titles.py first"; exit 1; }
git diff --quiet HEAD -- scripts src pyproject.toml || { echo "uncommitted changes in scripts/ or src/"; exit 1; }
tmp=$(mktemp -d)
git -c core.autocrlf=false archive --format=tar HEAD > "$tmp/repo.tar"
tar -rf "$tmp/repo.tar" data/12_titles.jsonl
gzip -c "$tmp/repo.tar" > "$OUT"
rm -rf "$tmp"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) from $(git rev-parse --short HEAD)"
