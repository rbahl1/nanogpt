import sys, pickle, collections, json
D = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rft_s1_round1.pkl','rb'))
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
kept = collections.Counter(x for ks in D['kept'] for x in ks)
for dom in ['prop','fol','nat','linear']:
    keptd = set(x for (dd,_,_),ks in zip(D['problems'],D['kept']) for x in ks if dd==dom)
    row=[]
    for lab in ['sft','rft_s1','rft_s2','rft_s3','rlhf_s1','ppochecker_s1','sftplus_s1']:
        R=[r for r in cks[lab]['records'] if r['domain']==dom]
        inkept=[r for r in R if r['answer'] in keptd]
        row.append(f"{lab}: {len(inkept)/len(R)*100:.0f}% (acc {sum(r['lean_ok'] for r in inkept)/max(len(inkept),1)*100:.0f}%)")
    print(dom, 'share of test answers that are verbatim a round-1 RFT(s1) training answer:', ' | '.join(row))
# rft_s1 losses: arm answer is a training answer?
for dom in ['prop','fol']:
    keptd = set(x for (dd,_,_),ks in zip(D['problems'],D['kept']) for x in ks if dd==dom)
    L=[(a,b) for a,b in zip(sft, cks['rft_s1']['records']) if a['domain']==dom and a['lean_ok'] and not b['lean_ok']]
    G=[(a,b) for a,b in zip(sft, cks['rft_s1']['records']) if a['domain']==dom and b['lean_ok'] and not a['lean_ok']]
    print(dom, 'rft_s1 losses whose answer is a verbatim training answer:', sum(b['answer'] in keptd for a,b in L), '/', len(L),
          '; gains:', sum(b['answer'] in keptd for a,b in G), '/', len(G))
