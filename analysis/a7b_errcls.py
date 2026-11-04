import json, sys, collections, pickle, re
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
msgs = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/leanmsgs.pkl','rb'))
def cls(r):
    if r['status']!='finished': return 'cut off/malformed'
    ms = msgs.get(r['prompt']+' '+r['answer'])
    if not ms: return 'not compiled'
    m = ms[0].split('\n')[0]
    if 'unexpected' in m or m.startswith('expected'): return 'syntax'
    if 'nknown identifier' in m: return 'unknown name'
    if 'Function expected' in m: return 'too many args'
    if 'type mismatch' in m.lower(): return 'type mismatch'
    if 'projection' in m.lower() or 'field' in m.lower(): return 'bad projection/field'
    if '⟨...⟩' in m: return 'bad ⟨⟩'
    if m.startswith('Tactic'): return 'tactic failed'
    return 'other'
cats=['type mismatch','too many args','bad projection/field','bad ⟨⟩','tactic failed','unknown name','syntax','cut off/malformed','other']
groups = {'pretrain':['pretrain'],'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
for dom in ['prop','fol']:
    print('==',dom, '(% of all answers, per seed avg)')
    print(f'{"":10s}', ' '.join(f'{c[:14]:>14s}' for c in cats), '  wrong')
    for g,labs in groups.items():
        c=collections.Counter(); n=0
        for lab in labs:
            for r in cks[lab]['records']:
                if r['domain']!=dom: continue
                n+=1
                if not r['lean_ok']: c[cls(r)]+=1
        print(f'{g:10s}', ' '.join(f'{c[k]/n*100:14.1f}' for k in cats), f'{sum(c.values())/n*100:7.1f}')
# For losses only: error class of arm answer on problems SFT solved
print('\nError class of the ARM answer on its LOSSES (SFT right, arm wrong), counts summed over seeds')
for dom in ['prop','fol']:
    for g,labs in groups.items():
        if g in('sft','pretrain'): continue
        c=collections.Counter()
        for lab in labs:
            for a,b in zip(cks['sft']['records'], cks[lab]['records']):
                if a['domain']==dom and a['lean_ok'] and not b['lean_ok']: c[cls(b)]+=1
        print(dom, g, dict(c.most_common()))
