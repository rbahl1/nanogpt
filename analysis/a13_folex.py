import sys, json, pickle, collections
z = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','rb'))
msgs = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/leanmsgs.pkl','rb'))
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
arm=sys.argv[1]; dom=sys.argv[2]; nmax=int(sys.argv[3])
# problems lost in all 3 seeds
for i,a in enumerate(sft):
    if a['domain']!=dom or not a['lean_ok']: continue
    bs=[cks[f'{arm}_s{s}']['records'][i] for s in (1,2,3)]
    if sum(not b['lean_ok'] for b in bs)>=3:
        nmax-=1
        if nmax<0: break
        print(a['prompt']); print(f"   SFT  ({z.get((a['prompt'],a['answer']),0):+.2f}): {a['answer']}")
        for s,b in zip((1,2,3),bs):
            m=msgs.get(b['prompt']+' '+b['answer'],['?'])
            print(f"   s{s}   ({z.get((b['prompt'],b['answer']),float('nan')):+.2f}): {b['answer']}   [{m[0].splitlines()[0][:60] if m else ''}]")
