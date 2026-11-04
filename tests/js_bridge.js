// Reads JSON lines [prompt, answer] on stdin; writes one JSON line per input:
// LeanCheck.verify(prompt, answer) (true, false, or null when no domain claims
// the prompt). Lets the Python tests compare the Python checkers with
// web/checker.js on many cases with one Node start-up.
const path = require("path");
const readline = require("readline");
const { verify } = require(path.join(__dirname, "..", "web", "checker.js"));
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const out = [];
rl.on("line", (line) => {
  if (!line) return;
  const [prompt, answer] = JSON.parse(line);
  let v;
  try { v = verify(prompt, answer); } catch (e) { v = "error: " + e.message; }
  out.push(JSON.stringify(v));
  if (out.length >= 4096) { process.stdout.write(out.join("\n") + "\n"); out.length = 0; }
});
rl.on("close", () => { if (out.length) process.stdout.write(out.join("\n") + "\n"); });
