#!/usr/bin/env node
// ---------------------------------------------------------------------------
// Playwright-based web test harness for the PyLinkx pygbag build.
//
// Prerequisites (one-time):
//   npm install -g playwright
//   npx playwright install chromium
//
// Usage:
//   # Start pygbag first (in another terminal or backgrounded):
//   uv run pygbag --port 8000 src/main.py &
//
//   # Run a scenario (starts game as Human vs Computer Hard):
//   NODE_PATH=$(npm root -g) node scripts/web_test.js --scenario vs-hard
//
//   # Manual key sequence with waits and mid-sequence screenshots:
//   NODE_PATH=$(npm root -g) node scripts/web_test.js --keys "ArrowDown,Enter,wait:3000,screenshot,ArrowDown,ArrowDown,Enter"
//
//   # Custom URL / output dir / load wait:
//   NODE_PATH=$(npm root -g) node scripts/web_test.js --scenario vs-hard --url http://localhost:8080 --out /tmp/shots --wait 20
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
const WAIT_SECS = parseInt(getArg("wait", "15"), 10);
const SCENARIO = getArg("scenario", "");
const KEYS_RAW = getArg("keys", "");

// -- Scenarios --------------------------------------------------------------
const SCENARIOS = {
  "vs-hard": [
    // Main menu → Human vs Computer
    "ArrowDown", "Enter",
    // Difficulty menu → Hard (Easy=0, Medium=1, Hard=2)
    "wait:2000", "ArrowDown", "ArrowDown", "Enter",
    // Wait for model to load and game to start
    "wait:5000", "screenshot",
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

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });

  // Collect browser console output
  const logs = [];
  page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => logs.push(`[PAGE_ERROR] ${err.message}`));

  console.log(`Navigating to ${URL} ...`);
  await page.goto(URL, { waitUntil: "networkidle", timeout: 60000 });

  // Wait for pygbag to download Python + wheels
  console.log(`Waiting ${WAIT_SECS}s for game to load ...`);
  await page.waitForTimeout(WAIT_SECS * 1000);

  // Click canvas to dismiss "Ready to start ! Please click/touch page"
  console.log("Clicking canvas to unlock media gate ...");
  await page.click("canvas", { force: true }).catch(() => {
    console.log("No canvas found, clicking page center ...");
    return page.mouse.click(512, 384);
  });
  await page.waitForTimeout(5000);

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
