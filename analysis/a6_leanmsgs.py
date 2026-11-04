import json, sys, os, subprocess, tempfile, collections, pickle
sys.path.insert(0, '/home/user/nanoGPT')
import lean_oracle
LEAN='/opt/lean/lean-4.34.0-linux/bin/lean'
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
labs = ['sft','pretrain'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
decls=set()
for lab in labs:
    for r in cks[lab]['records']:
        if r['domain'] in ('prop','fol') and not r['lean_ok']:
            decls.add(r['prompt']+' '+r['answer'])
decls=sorted(d for d in decls if lean_oracle.is_plain_example(d))
print(len(decls))
def compile_msgs(decls):
    lines=list(lean_oracle._SENTINEL); spans=[]
    for dcl in decls:
        start=len(lines)+1; lines.append('section'); lines.extend(dcl.split('\n')); lines.append('end'); spans.append((start,len(lines)))
    with tempfile.TemporaryDirectory() as tmp:
        p=os.path.join(tmp,'Check.lean'); open(p,'w').write('\n'.join(lines)+'\n')
        proc=subprocess.run([LEAN,'-DmaxErrors=1000000','--json',p],capture_output=True,text=True,timeout=1800)
    msgs=collections.defaultdict(list); stray=[]
    for line in proc.stdout.splitlines():
        try: m=json.loads(line)
        except: continue
        if m.get('severity')!='error': continue
        row=m['pos']['line']
        owner=max((i for i,s in enumerate(spans) if s[0]<=row), default=None)
        if owner is None: stray.append(m['data'])
        else: msgs[owner].append(m['data'])
    return msgs, stray
out={}
CH=1500
for i in range(0,len(decls),CH):
    chunk=decls[i:i+CH]
    msgs,stray=compile_msgs(chunk)
    print(i, len(msgs), 'stray', len(stray), stray[:2], flush=True)
    for j,dcl in enumerate(chunk): out[dcl]=msgs.get(j,[])
pickle.dump(out, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/leanmsgs.pkl','wb'))
