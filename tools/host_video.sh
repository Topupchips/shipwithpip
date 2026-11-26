#!/usr/bin/env bash
# Usage: tools/host_video.sh path/to/reel.mp4
# Adds ONE video to the orphan branch `media` and prints its public URL.
# The branch is a single force-pushed commit (history never grows), but it KEEPS every video
# hosted in the last KEEP_DAYS days (default 7), so queued extras are not wiped by the next post.
set -e
f="$(realpath "$1")"; name="$(date -u +%Y%m%d-%H%M)-$(basename "$f")"
repo="$(cd "$(dirname "$0")/.." && pwd)"; tmp="$(mktemp -d)"; keep="${KEEP_DAYS:-7}"
origin="$(git -C "$repo" remote get-url origin)"
cd "$tmp"; git init -q; git checkout -q -b media
git remote add origin "$origin"
if git fetch -q origin media 2>/dev/null; then
  cutoff="$(date -u -d "-$keep days" +%Y%m%d)"
  for old in $(git ls-tree --name-only FETCH_HEAD); do
    d="${old%%-*}"
    if [[ "$d" =~ ^[0-9]{8}$ ]] && [ "$d" -ge "$cutoff" ]; then git show "FETCH_HEAD:$old" > "$old"; fi
  done
fi
cp "$f" "$name"; git add -A .
git -c user.name="shipwithpip-bot" -c user.email="bot@shipwithpip.invalid" commit -qm "media: $name (+ last $keep days)"
git push -q -f origin media
url="https://raw.githubusercontent.com/Topupchips/shipwithpip/media/$name"
for i in $(seq 1 12); do code=$(curl -sI -o /dev/null -w '%{http_code}' "$url"); [ "$code" = 200 ] && break; sleep 5; done
echo "$url"
