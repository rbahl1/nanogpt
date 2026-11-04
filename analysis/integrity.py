import json, sys, time, collections, itertools, re
sys.path.insert(0, "/home/user/nanoGPT")
import data, formal
from formal import prop as P, linear as L, nat as N
E = json.load(open("/home/user/nanoGPT/results/eval.json"))
recs = E["checkpoints"][1]["records"]
test = [(r["domain"], r["prompt"]) for r in recs]
tprompts = set(p for _, p in test)
tkeys = {data.split_key(p) for _, p in test}
print("test problems", len(test), collections.Counter(d for d, _ in test))
print("  split_of == test for all:", all(data.split_of(p) == "test" for _, p in test))
print("  distinct split keys:", len(tkeys), " distinct prompts:", len(tprompts))
print("  domain_for agrees with record domain:", all(formal.domain_for(p).name == d for d, p in test))
t = time.time()
regen = data.sample_problems(2000, "test", 31337, unique=True)
print(f"  regenerated with data.sample_problems(2000,'test',31337,unique=True): identical = {[(p.domain, p.prompt) for p in regen] == test} ({time.time()-t:.0f}s)")
# reference answers of the test problems compile per checker
print("  test references accepted by checker:", sum(formal.verify(p.prompt, p.reference) for p in regen), "/ 2000")
json.dump([p.prompt + " " + p.reference for p in regen], open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/test_refs.json", "w"), ensure_ascii=False)

# ---------- looser keys
def prop_loose(prompt):
    f = P.goal_of(prompt)
    if f is None: return None
    f = P._fold(f)
    atoms = P.atoms_in_order(f)
    def norm(g):
        tag = g[0]
        if tag in ("atom", "top", "bot"): return g
        if tag == "not": return ("not", norm(g[1]))
        if tag in ("and", "or"):
            parts = []
            def coll(h):
                if h[0] == tag: coll(h[1]); coll(h[2])
                else: parts.append(norm(h))
            coll(g)
            return (tag,) + tuple(sorted(set(parts), key=repr))
        if tag == "iff":
            return ("iff",) + tuple(sorted((norm(g[1]), norm(g[2])), key=repr))
        if tag == "imp":   # premises as a multiset: A → B → C ≡ B → A → C
            prem = []
            h = g
            while h[0] == "imp" and h[2] != P.BOT:
                prem.append(norm(h[1])); h = h[2]
            return ("imp", tuple(sorted(prem, key=repr)), norm(h))
        raise ValueError(tag)
    if len(atoms) > 7: return None
    best = None
    for perm in itertools.permutations(range(len(atoms))):
        m = {a: f"a{perm[i]}" for i, a in enumerate(atoms)}
        s = repr(norm(P._rename(f, m)))
        if best is None or s < best: best = s
    return best

def lin_norm(prompt):
    body = prompt[len(L.HEAD):len(prompt) - len(L.TAIL)]
    a, b = body.split("=")
    k1, m1, _, _ = L._parse_side(a); k2, m2, _, _ = L._parse_side(b)
    A, B = k1 - k2, m2 - m1
    if A < 0: A, B = -A, -B
    return (A, B)

def nat_value_shape(prompt):
    # value + multiset of literals + multiset of operators (loose: ignores structure)
    body = prompt[len(N.HEAD):len(prompt) - len(N.TAIL)]
    nums = tuple(sorted(int(x) for x in re.findall(r"\d+", body)))
    ops = tuple(sorted(re.findall(r"[-+*/%]", body)))
    return (N.value_of(prompt), nums, ops)

def shape(prompt):
    k = data.split_key(prompt)
    return re.sub(r"\b[A-Za-zα-ω_][A-Za-z0-9_']*\b", "I", re.sub(r"\d+", "0", k))

def loose(d, p):
    if d == "prop": return prop_loose(p)
    if d == "linear": return lin_norm(p)
    if d == "nat": return nat_value_shape(p)
    return None

t = time.time()
tloose = {d: collections.Counter(loose(d, p) for dd, p in test if dd == d) for d in ("prop", "nat", "linear")}
print(f"loose keys for test computed ({time.time()-t:.0f}s)")

def check_stream(name, prompts, expect, domains_of=None, loose_check=False):
    prompts = list(prompts)
    t = time.time()
    keys = [data.split_key(p) for p in prompts]
    bad_split = sum(data.split_of_key(k) != expect for k in keys)
    key_hits = sum(k in tkeys for k in keys)
    exact_hits = sum(p in tprompts for p in prompts)
    line = (f"{name:34} n={len(prompts):7d} distinct={len(set(prompts)):7d} split!={expect}: {bad_split}  "
            f"test-key collisions: {key_hits}  exact test prompts: {exact_hits}")
    out = {"n": len(prompts), "bad_split": bad_split, "key_hits": key_hits, "exact_hits": exact_hits}
    if loose_check:
        # how many TEST problems have a looser near-duplicate in this stream
        parts = []
        for d in ("prop", "nat", "linear"):
            sl = set(loose(d, p) for p in prompts if formal.domain_for(p).name == d)
            hit = sum(c for k, c in tloose[d].items() if k in sl)
            parts.append(f"{d} {hit}/500")
            out[f"loose_{d}"] = hit
        tshape = collections.Counter(shape(p) for dd, p in test if dd == "fol")
        ss = set(shape(p) for p in prompts if formal.domain_for(p).name == "fol")
        fh = sum(c for k, c in tshape.items() if k in ss)
        parts.append(f"fol(shape) {fh}/500")
        out["shape_fol"] = fh
        line += "\n" + " " * 36 + "test problems with a looser near-duplicate here: " + ", ".join(parts)
    print(line + f"  ({time.time()-t:.0f}s)", flush=True)
    return out

res = {}
val = data.sample_problems(75 * 4, "val", 0, "all", unique=True)
res["val"] = check_stream("val (VAL_SEED 0, 75/domain)", [p.prompt for p in val], "val")
t = time.time(); sft = data.sample_problems(80000, "train", 2024, "all"); print(f"  generated SFT 80k in {time.time()-t:.0f}s")
res["sft"] = check_stream("SFT 80k (seed 2024)", [p.prompt for p in sft], "train", loose_check=True)
print("  SFT references accepted by checker (first 4000):", sum(formal.verify(p.prompt, p.reference) for p in sft[:4000]), "/ 4000")
pref = [json.loads(l) for l in open("/home/user/nanoGPT/out/preferences.jsonl")]
res["pref"] = check_stream("RM comparisons (preferences.jsonl)", [r["prompt"] for r in pref], "train")
rm = data.sample_problems(6000, "train", 7, "all", unique=True)
res["rm"] = check_stream("RM prompt stream (6000, seed 7)", [p.prompt for p in rm], "train")
print("  preference prompts subset of RM stream:", set(r["prompt"] for r in pref) <= set(p.prompt for p in rm))
for s in (1, 2, 3):
    res[f"ppo{s}"] = check_stream(f"PPO prompt stream seed {s} (9600)", [p.prompt for p in data.sample_problems(150 * 64, "train", s, "all", unique=True)], "train")
    res[f"rft{s}"] = check_stream(f"RFT prompt stream seed {s} (2400)", [p.prompt for p in data.sample_problems(2 * 1200, "train", s, "all", unique=True)], "train")
for s in (1001, 1002, 1003):
    res[f"sftplus{s}"] = check_stream(f"sftplus refs seed {s} (80k)", [p.prompt for p in data.sample_problems(80000, "train", s, "all")], "train")
t = time.time(); pre = data.sample_problems(160000, "train", 1337, "all"); print(f"  generated pretrain 160k in {time.time()-t:.0f}s")
res["pretrain"] = check_stream("pretrain 160k docs (seed 1337)", [p.prompt for p in pre], "train", loose_check=True)
# Pretrain documents: check a sample of documents start with their prompt (doc construction)
import random
rng = random.Random("1337/documents")
docs = [formal.DOMAINS[p.domain].document(p, rng) for p in pre[:2000]]
print("  first 2000 pretrain documents start with their train prompt:", all(d.startswith(p.prompt + " ") for d, p in zip(docs, pre)))
json.dump(res, open("/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/integrity.json", "w"), indent=1)
