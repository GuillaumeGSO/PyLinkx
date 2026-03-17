---
description: Build the PyLinkx web package with pygbag and push it to itch.io via butler.
---

Deploy PyLinkx to itch.io. Follow these steps exactly, stopping and reporting any failure before continuing.

## Step 1 — Read configuration

Read `.itch.toml` at the project root. It must contain:

```toml
[itch]
user    = "<username>"
game    = "<game-slug>"
channel = "html5"
```

If the file is missing or any key is absent, stop and tell the user:
> `.itch.toml` is missing or incomplete. Create it at the project root with your itch.io username, game slug, and channel.

## Step 2 — Check butler is installed

Run: `which butler`

If butler is not found, stop and tell the user:
> `butler` is not installed or not on PATH.
> 1. Download it from https://itchio.itch.io/butler
> 2. Extract the binary to `/usr/local/bin/butler` and make it executable
> 3. Run `butler login` once to authenticate

## Step 3 — Build the web package

Run from the project root:
```bash
source venv/bin/activate && python -m pygbag --build src/main.py
```

Wait for completion. Output goes to `src/build/web/`. Report any errors and stop if the build fails.

## Step 4 — Verify build output

Check that `src/build/web/index.html` exists. If not, the build failed — report this and stop.

## Step 5 — Capture current version, then push

First capture the current version number (before the push):
```bash
prev_version=$(bash .claude/skills/itch-deploy/butler-version.sh)
echo "Current version before push: $prev_version"
```

Then push:
```bash
butler push src/build/web/ <user>/<game>:<channel>
```

## Step 6 — Monitor until live (background)

Run the following command using the Bash tool with `run_in_background: true` so Claude remains free for other work:

```bash
bash .claude/skills/itch-deploy/deploy-monitor.sh <prev_version captured above>
```

Tell the user the monitoring is running in the background and they will be notified when the build is live.

## Step 7 — Report success

When the background task completes, notify the user that the deployment is live and show the game URL: `https://<user>.itch.io/<game>`
