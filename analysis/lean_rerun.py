import json, sys, random, time
sys.path.insert(0, "/home/user/nanoGPT")
import lean_oracle
D = json.load(open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/decls.json"))
decls, stored = D["decls"], D["lean"]
LEAN = "/opt/lean/lean-4.34.0-linux/bin/lean"
order = list(range(len(decls))); random.Random(1).shuffle(order)
t = time.time()
v = lean_oracle.check_in_chunks([decls[i] for i in order], LEAN, chunk=1500)
new = [None] * len(decls)
for i, ok in zip(order, v): new[i] = ok
print(f"re-compiled {len(decls)} distinct declarations, shuffled, chunk 1500: {time.time()-t:.0f}s")
diff = [i for i in range(len(decls)) if new[i] != stored[i]]
print("verdicts differing from eval.json:", len(diff))
for i in diff[:20]: print("  ", stored[i], new[i], decls[i])
print("accepted:", sum(new), "stored accepted:", sum(stored))
# individually compile the FN declarations and a random sample of 60 accepted + 60 rejected, one Lean process each
rng = random.Random(2)
acc = [i for i in range(len(decls)) if stored[i]]; rej = [i for i in range(len(decls)) if not stored[i]]
fn = [decls.index(d) for d in dict.fromkeys(D["fn"])]
sample = fn + rng.sample(acc, 60) + rng.sample(rej, 60)
t = time.time()
bad = [i for i in sample if lean_oracle.check_declarations([decls[i]], LEAN)[0] != stored[i]]
print(f"individually compiled {len(sample)} (2 FN + 60 accepted + 60 rejected): {len(bad)} disagree with batch verdicts ({time.time()-t:.0f}s)")
for i in bad: print("  ", decls[i])
json.dump(new, open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/lean_rerun.json", "w"))
