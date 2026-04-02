#!/usr/bin/env bash
# Sync episode data and shownotes from the webserver.
# Usage: ./ingest/sync.sh

set -euo pipefail

LOCAL_DATA="data/inputs"
REMOTE_EPISODES="web:/home/pfh/code/tgn-whisperer/podcasts/tgn/"
REMOTE_SHOWNOTES="web:/home/pfh/code/tgn-whisperer/sites/tgn/docs/shownotes.md"

mkdir -p "$LOCAL_DATA"

echo "Syncing episodes..."
rsync -avz --delete \
    --include='*/' \
    --include='episode.md' \
    --include='episode.json' \
    --exclude='*' \
    "$REMOTE_EPISODES" "$LOCAL_DATA/"

echo ""
echo "Syncing shownotes..."
rsync -avz "$REMOTE_SHOWNOTES" "$LOCAL_DATA/shownotes.md"

echo ""
echo "Done. Episodes on disk:"
ls -d "$LOCAL_DATA"/[0-9]* | wc -l | xargs echo " "
