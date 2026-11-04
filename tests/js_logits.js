// Usage: node js_logits.js model.json '[ids...]'  -> JSON list of logits rows, one per fed token.
const fs = require("fs");
const path = require("path");
const { GPT } = require(path.join(__dirname, "..", "web", "gpt.js"));
const bundle = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ids = JSON.parse(process.argv[3]);
const model = new GPT(bundle);
const state = model.newState();
const rows = [];
for (const id of ids) rows.push(Array.from(model.step(state, id)));
process.stdout.write(JSON.stringify(rows));
