#!/bin/bash
# Usage: deploy-monitor.sh <prev_version>
# Polls until the live version number changes, then prints success.
prev_version=$1
attempts=0
while [ $attempts -lt 20 ]; do
  current_version=$(bash .claude/skills/itch-deploy/butler-version.sh)
  if [ -n "$current_version" ] && [ "$current_version" != "$prev_version" ]; then
    echo "✅ Version changed from $prev_version to $current_version — deployment live!"
    echo "https://guillaumegso.itch.io/pygame-linkx-test"
    exit 0
  fi
  attempts=$((attempts + 1))
  echo "Still on version $prev_version… attempt $attempts/20, waiting 30s"
  sleep 30
done
echo "⚠️ Timed out after 10min — check itch.io dashboard"
exit 0
