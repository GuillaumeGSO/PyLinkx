#!/usr/bin/env node
// ---------------------------------------------------------------------------
// Playwright-based web test harness for the PyLinkx pygbag build.
//
// Prerequisites (one-time):
//   npm install
//   npx playwright install chromium
//
// Usage:
//   # Start pygbag first (in another terminal or backgrounded):
//   uv run pygbag --port 8000 src/main.py &
//
//   # Run a scenario (starts game as Human vs Computer Hard):
//   node scripts/web_test.js --scenario vs-hard
//
//   # Manual key sequence with waits and mid-sequence screenshots:
//   node scripts/web_test.js --keys "ArrowDown,Enter,wait:3000,screenshot,ArrowDown,ArrowDown,Enter"
//
//   # Custom URL / output dir / load wait:
//   node scripts/web_test.js --scenario vs-hard --url http://localhost:8080 --out /tmp/shots --wait 20
//
// Key DSL (comma-separated):
//   KeyName     — any Playwright key name (Enter, ArrowDown, ArrowUp, Tab, etc.)
//   wait:N      — pause N milliseconds
//   screenshot  — take a screenshot at this point (auto-numbered)
//
// Screenshots are saved to ./screenshots/ by default.
// ---------------------------------------------------------------------------

const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

// -- Parse CLI args ---------------------------------------------------------
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : defaultVal;
}

const URL = getArg("url", "http://localhost:8000");
const OUT_DIR = getArg("out", path.join(process.cwd(), "screenshots"));
const WAIT_SECS = parseInt(getArg("wait", "40"), 10);
const SCENARIO = getArg("scenario", "vs-hard");
const KEYS_RAW = getArg("keys", "");

// -- Scenarios --------------------------------------------------------------
const SCENARIOS = {
  "vs-hard": [
    // Main menu → Human vs Computer
    "ArrowDown", "Enter",
    // Difficulty menu → Hard (Easy=0, Medium=1, Hard=2)
    "wait:2000", "ArrowDown", "ArrowDown", "Enter",
    // Wait for model to load, AI to play, then P2 turn
    "wait:15000", "screenshot",
    // P2: move left, screenshot
    "ArrowLeft", "ArrowLeft", "wait:500", "screenshot",
    // P2: move right, screenshot
    "ArrowRight", "ArrowRight", "ArrowRight", "wait:500", "screenshot",
    // P2: rotate (Up), screenshot
    "ArrowUp", "wait:500", "screenshot",
    // P2: flip (Enter), screenshot
    "Enter", "wait:500", "screenshot",
    // P2: cycle piece (Tab), screenshot
    "Tab", "wait:500", "screenshot",
  ],
  "vs-medium": [
    "ArrowDown", "Enter",
    "wait:2000", "ArrowDown", "Enter",
    "wait:5000", "screenshot",
  ],
  "vs-easy": [
    "ArrowDown", "Enter",
    "wait:2000", "Enter",
    "wait:5000", "screenshot",
  ],
  "2p": [
    // Main menu → 2 Human Players
    "Enter",
    "wait:1000", "screenshot",
  ],
  "menu": [
    // Just screenshot the menu (no keys)
    "screenshot",
  ],
};

// -- Build step list --------------------------------------------------------
function buildSteps() {
  if (SCENARIO) {
    const steps = SCENARIOS[SCENARIO];
    if (!steps) {
      console.error(`Unknown scenario: "${SCENARIO}". Available: ${Object.keys(SCENARIOS).join(", ")}`);
      process.exit(1);
    }
    return steps;
  }
  if (KEYS_RAW) {
    return KEYS_RAW.split(",").map((k) => k.trim());
  }
  return ["screenshot"]; // default: just screenshot after load
}

// -- Main -------------------------------------------------------------------
(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  // Clean old screenshots
  for (const f of fs.readdirSync(OUT_DIR)) {
    if (f.endsWith(".png")) fs.unlinkSync(path.join(OUT_DIR, f));
  }

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });

  // Collect browser console output
  const logs = [];
  page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => logs.push(`[PAGE_ERROR] ${err.message}`));

  console.log(`Navigating to ${URL} ...`);
  await page.goto(URL, { waitUntil: "networkidle", timeout: 60000 });

  // Poll until game is ready: wait for initial load, then click to dismiss
  // the "Ready to start ! Please click/touch page" splash, and keep clicking
  // until the pygame canvas is rendering (splash gone).
  console.log(`Waiting up to ${WAIT_SECS * 2}s for game to load ...`);
  const maxAttempts = WAIT_SECS * 2;  // one attempt per second
  let ready = false;
  for (let i = 0; i < maxAttempts; i++) {
    await page.waitForTimeout(1000);
    // Try clicking the canvas every few seconds
    if (i >= WAIT_SECS && i % 3 === 0) {
      await page.click("canvas", { force: true }).catch(() =>
        page.mouse.click(512, 384)
      );
    }
    // Check if the splash text is gone (pygbag sets a DOM element)
    const splashGone = await page.evaluate(() => {
      const el = document.getElementById("ume");
      return !el || el.style.display === "none" || el.offsetHeight === 0;
    }).catch(() => false);
    if (splashGone && i >= WAIT_SECS) {
      console.log(`Game ready after ${i}s`);
      ready = true;
      await page.waitForTimeout(2000);  // let the menu render
      break;
    }
  }
  if (!ready) {
    // Final click attempt
    console.log("Splash may still be up, clicking once more ...");
    await page.click("canvas", { force: true }).catch(() =>
      page.mouse.click(512, 384)
    );
    await page.waitForTimeout(3000);
  }

  // Screenshot after initial load
  let shotNum = 1;
  const initialShot = path.join(OUT_DIR, `${String(shotNum).padStart(2, "0")}_after_load.png`);
  await page.screenshot({ path: initialShot });
  console.log(`Screenshot: ${initialShot}`);
  shotNum++;

  // Execute steps
  const steps = buildSteps();
  for (const step of steps) {
    if (step.startsWith("wait:")) {
      const ms = parseInt(step.slice(5), 10);
      console.log(`Waiting ${ms}ms ...`);
      await page.waitForTimeout(ms);
    } else if (step === "screenshot") {
      const shotPath = path.join(OUT_DIR, `${String(shotNum).padStart(2, "0")}_step.png`);
      await page.screenshot({ path: shotPath });
      console.log(`Screenshot: ${shotPath}`);
      shotNum++;
    } else {
      // Keyboard input
      await page.keyboard.press(step);
      await page.waitForTimeout(300);
    }
  }

  // Final screenshot
  const finalShot = path.join(OUT_DIR, `${String(shotNum).padStart(2, "0")}_final.png`);
  await page.screenshot({ path: finalShot });
  console.log(`Screenshot: ${finalShot}`);

  // Print browser console output
  console.log("\n--- Browser Console Output ---");
  logs.forEach((l) => console.log(l));
  console.log("--- End Browser Console ---");

  await browser.close();
  console.log("\nDone.");
})();
