#!/usr/bin/env bash
# Usage: tools/host_video.sh path/to/reel.mp4
# Puts ONE video on the orphan branch `media` (force-pushed, so history never grows) and prints its public URL.
set -e
f="$(realpath "$1")"; name="$(date -u +%Y%m%d-%H%M)-$(basename "$f")"
repo="$(cd "$(dirname "$0")/.." && pwd)"; tmp="$(mktemp -d)"
cd "$tmp"; git init -q; git checkout -q -b media
cp "$f" "$name"; git add "$name"
git -c user.name="shipwithpip-bot" -c user.email="bot@shipwithpip.invalid" commit -qm "media: $name"
git remote add origin "$(git -C "$repo" remote get-url origin)"
git push -q -f origin media
url="https://raw.githubusercontent.com/Topupchips/shipwithpip/media/$name"
for i in $(seq 1 12); do code=$(curl -sI -o /dev/null -w '%{http_code}' "$url"); [ "$code" = 200 ] && break; sleep 5; done
echo "$url"
