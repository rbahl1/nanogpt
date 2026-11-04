import json, sys, collections, random, time
sys.path.insert(0, "/home/user/nanoGPT")
import formal, lean_oracle
E = json.load(open("/home/user/nanoGPT/results/eval.json"))
cks = E["checkpoints"]
print("checkpoints", len(cks), [c["label"] for c in cks])
print("eval config", {k: E["config"][k] for k in ("n","split","seed","lean","max_new_tokens","batch_size")})
recs = [(c["label"], i, r) for c in cks for i, r in enumerate(c["records"])]
print("records", len(recs))
# same prompts/order everywhere
p0 = [(r["domain"], r["prompt"]) for r in cks[0]["records"]]
print("same problem order for all:", all([(r["domain"], r["prompt"]) for r in c["records"]] == p0 for c in cks))
st = collections.Counter(r["status"] for _, _, r in recs)
print("status", st)
fin = [(l, i, r) for l, i, r in recs if r["status"] == "finished"]
decls = list(dict.fromkeys(r["prompt"] + " " + r["answer"] for _, _, r in fin))
print("distinct finished declarations", len(decls))
alld = list(dict.fromkeys(r["prompt"] + " " + r["answer"] for _, _, r in recs))
print("distinct (incl. unfinished)", len(alld))
# consistency of verdict per declaration
vd = collections.defaultdict(set); cd = collections.defaultdict(set)
for _, _, r in fin:
    vd[r["prompt"] + " " + r["answer"]].add(r["lean_ok"]); cd[r["prompt"] + " " + r["answer"]].add(r["checker_ok"])
print("declarations with inconsistent lean verdicts", sum(len(v) > 1 for v in vd.values()), "checker", sum(len(v) > 1 for v in cd.values()))
print("unfinished with lean_ok True:", sum(1 for _, _, r in recs if r["status"] != "finished" and r["lean_ok"]),
      "unfinished with checker_ok True:", sum(1 for _, _, r in recs if r["status"] != "finished" and r["checker_ok"]))
# re-run checker
t = time.time()
mism = [(l, i, r) for l, i, r in recs if r["checker_ok"] != (r["status"] == "finished" and formal.verify(r["prompt"], r["answer"]))]
print(f"checker verdict re-run mismatches: {len(mism)} ({time.time()-t:.0f}s)")
fp = [(l, i, r) for l, i, r in recs if r["checker_ok"] and not r["lean_ok"]]
fn = [(l, i, r) for l, i, r in recs if r["lean_ok"] and not r["checker_ok"]]
print("checker+ Lean-:", len(fp), " checker- Lean+:", len(fn))
for l, i, r in fn:
    print(f"  [{l} #{i} {r['domain']}] {r['prompt']} {r['answer']}")
# distinct declarations by verdict
dv = {d: next(iter(vd[d])) for d in decls}
dc = {d: next(iter(cd[d])) for d in decls}
print("distinct decls lean-accepted", sum(dv.values()), "checker-accepted", sum(dc.values()))
print("distinct by domain", collections.Counter(formal.domain_for(d.split(" := ")[0] + " :=").name if False else None for d in decls) if False else "")
bydom = collections.Counter()
for _, _, r in fin:
    pass
dd = {}
for _, _, r in fin:
    dd[r["prompt"] + " " + r["answer"]] = r["domain"]
print("distinct by domain", collections.Counter(dd.values()))
print("distinct lean-accepted by domain", collections.Counter(dd[d] for d in decls if dv[d]))
# plain-example filter
nonplain = [d for d in decls if not lean_oracle.is_plain_example(d)]
print("finished declarations rejected unread by the plain-example filter:", len(nonplain))
for d in nonplain[:10]: print("   ", d)
# answers per checkpoint by Lean, check summary results
for c in cks:
    a = c["results"]["all"]
    lc = sum(r["lean_ok"] for r in c["records"])
    assert lc == a["lean_correct"], c["label"]
json.dump({"decls": decls, "lean": [dv[d] for d in decls], "checker": [dc[d] for d in decls], "fn": [r["prompt"] + " " + r["answer"] for _, _, r in fn]},
          open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/decls.json", "w"), ensure_ascii=False)
