import json
import numpy as np
for name in ["rlhf_s1","rlhf_s2","rlhf_s3","ppochecker_s1","ppochecker_s2","ppochecker_s3"]:
    L = json.load(open(f"results/logs/{name}.json"))
    its = L["iterations"]
    acc = np.array([x["accuracy"] for x in its]); sc = np.array([x["score"] for x in its])
    rm = np.array([x["rm_score"] if x["rm_score"] is not None else np.nan for x in its])
    kl = np.array([x["kl"] for x in its]); tok = np.array([x["answer_tokens"] for x in its])
    t = np.arange(len(its))
    sl = lambda y: np.polyfit(t, y, 1)[0]*len(its)
    secs = sum(x["seconds"] for x in its)
    print(f"{name}: n_it={len(its)} acc slope over run {sl(acc)*100:+.2f}pt, first30 {acc[:30].mean()*100:.1f} last30 {acc[-30:].mean()*100:.1f}; "
          f"score first30 {sc[:30].mean():+.3f} last30 {sc[-30:].mean():+.3f}; rm raw first30 {np.nanmean(rm[:30]):+.3f} last30 {np.nanmean(rm[-30:]):+.3f}; "
          f"kl last30 {kl[-30:].mean():.3f} max {kl.max():.3f}; tokens {tok[:30].mean():.1f}->{tok[-30:].mean():.1f}; {secs/3600:.2f} h")
    print("   val evals:", [(e["iteration"], round(e["results"]["all"]["accuracy"]*100,1)) for e in L["evals"]])
    last = L["evals"][-1]["results"]; first = L["evals"][0]["results"]
    print("   val by domain first:", {k: round(v["accuracy"]*100,1) for k,v in first.items()}, " last:", {k: round(v["accuracy"]*100,1) for k,v in last.items()})
    print("   config lr", L["config"]["lr"], "kl", L["config"]["kl_coef"], "epochs", L["config"]["ppo_epochs"], "select_best", L["config"]["select_best"], "reward_mean/std", L["reward_mean"], L["reward_std"])
for name in ["rft_s1","rft_s2","rft_s3"]:
    L = json.load(open(f"results/logs/{name}.json"))
    print(name, [(r["round"], r["examples"], r["steps"], round(r["val"]["all"]["accuracy"]*100,1)) for r in L["rounds"]], "val0", round(L["evals"][0]["results"]["all"]["accuracy"]*100,1))
    print("   val by domain last:", {k: round(v["accuracy"]*100,1) for k,v in L["rounds"][-1]["val"].items()})
