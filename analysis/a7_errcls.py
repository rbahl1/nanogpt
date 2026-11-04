import json, sys, collections, pickle, re
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
msgs = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/leanmsgs.pkl','rb'))
allfirst=collections.Counter()
def cls(dcl):
    ms = msgs.get(dcl)
    if ms is None: return 'not-plain'
    if not ms: return 'none?'
    m = ms[0].split('\n')[0]
    if 'unexpected' in m or 'expected' in m and 'type' not in m: return 'parse error'
    if m.startswith('type mismatch') : return 'type mismatch'
    if m.startswith('Application type mismatch') or m.startswith('application type mismatch'): return 'application type mismatch'
    if m.startswith('Unknown identifier') or m.startswith('unknown identifier'): return 'unknown identifier'
    if 'Function expected' in m or 'function expected' in m: return 'function expected'
    if 'Invalid field' in m or 'invalid field' in m or 'Invalid projection' in m or 'invalid projection' in m: return 'invalid field/projection'
    if 'unsolved goals' in m: return 'unsolved goals'
    if 'Invalid ⟨...⟩' in m or 'invalid constructor' in m.lower() or 'anonymous constructor' in m: return 'anonymous constructor'
    return m[:50]
groups = {'pretrain':['pretrain'],'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
for dom in ['prop','fol']:
    print('==',dom)
    for g,labs in groups.items():
        c=collections.Counter(); n=0; nw=0
        for lab in labs:
            for r in cks[lab]['records']:
                if r['domain']!=dom: continue
                n+=1
                if r['lean_ok']: continue
                nw+=1
                k = 'cut off' if r['status']!='finished' else cls(r['prompt']+' '+r['answer'])
                c[k]+=1
        print(f'{g:10s} wrong={nw/n*100:.1f}% of answers; error mix (% of wrong):', ', '.join(f'{k}:{v/nw*100:.1f}' for k,v in c.most_common(9)))
# print raw first lines most common
fl=collections.Counter()
for dcl,ms in msgs.items():
    if ms: fl[re.sub(r'\d+','N',ms[0].split('\n')[0])[:70]]+=1
print(fl.most_common(25))
