import json, sys, collections, re
sys.path.insert(0, "/home/user/nanoGPT")
import data, formal
from formal import nat as N, linear as L
E = json.load(open("/home/user/nanoGPT/results/eval.json"))
ck = {c["label"]: c for c in E["checkpoints"]}
recs = ck["sft"]["records"]
test = [(r["domain"], r["prompt"]) for r in recs]
# Train references (SFT sample) for frequency baselines
train = data.sample_problems(20000, "train", 2024, "all")  # first 5000/domain of SFT's stream
freq = {d: collections.Counter(p.reference for p in train if p.domain == d) for d in formal.DOMAINS}
print("== constant / heuristic answer baselines on the 2000 test problems (checker)")
for d in formal.DOMAINS:
    tp = [p for dd, p in test if dd == d]
    top = freq[d].most_common(10)
    # best single constant answer among the 20 most frequent train references
    best = max(((sum(formal.verify(p, a) for p in tp), a) for a, _ in freq[d].most_common(20)))
    # oracle best constant over all train references (upper bound for one fixed answer)
    print(f"{d:7} most frequent train reference {top[0][0]!r} ({100*top[0][1]/sum(freq[d].values()):.1f}% of train refs); "
          f"best of top-20 train refs as a constant answer: {best[1]!r} -> {100*best[0]/len(tp):.1f}%")
    if d in ("nat", "linear"):
        val = (lambda p: N.value_of(p)) if d == "nat" else (lambda p: L.solution_of(p))
        vals = collections.Counter(val(p) for p in tp)
        print(f"         test answer values: most common {vals.most_common(5)}; best constant value {100*vals.most_common(1)[0][1]/len(tp):.1f}%")
        nums = lambda p: [int(x) for x in re.findall(r"\d+", p[len(N.HEAD if d=='nat' else L.HEAD):])] if d == "nat" else [int(x) for x in re.findall(r"\d+", p[len(L.HEAD):-len(L.TAIL)])]
        h = {"first number": lambda p: nums(p)[0], "last number": lambda p: nums(p)[-1], "max": lambda p: max(nums(p)), "min": lambda p: min(nums(p))}
        print("         positional guesses: " + ", ".join(f"{k} {100*sum(f(p) == val(p) for p in tp)/len(tp):.1f}%" for k, f in h.items()))
# Memorisation-type stats for SFT: correct answers equal to the test reference / to any train reference
refs = {p.prompt: p.reference for p in data.sample_problems(2000, "test", 31337, unique=True)}
trainrefs = {d: set(freq[d]) for d in freq}
print("\n== SFT/arm greedy answers: Lean-correct, = test reference, answer string seen as a train reference (first 20k SFT stream)")
for l in ("sft", "rlhf_s1", "ppochecker_s1", "rft_s1", "sftplus_s1"):
    out = []
    for d in formal.DOMAINS:
        rs = [r for r in ck[l]["records"] if r["domain"] == d]
        ok = [r for r in rs if r["lean_ok"]]
        eqref = sum(r["answer"] == refs[r["prompt"]] for r in ok)
        seen = sum(r["answer"] in trainrefs[d] for r in ok)
        out.append(f"{d} {len(ok)} correct, {eqref} = ref, {seen} seen")
    print(f"{l:14} " + "; ".join(out))
