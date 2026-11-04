import sys, json, re, collections
sys.path.insert(0,'/home/user/nanoGPT')
from formal import nat, linear
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
def v(a):
    m=re.match(r'^(-?\d+) :=',a); return int(m.group(1)) if m else None
print('nat: SFT right, rft_s1 says 0 (value != 0):')
n=0
for i,a in enumerate(sft):
    if a['domain']=='nat' and a['lean_ok']:
        b=[cks[f'rft_s{s}']['records'][i]['answer'] for s in (1,2,3)]; r=[cks[f'rlhf_s{s}']['records'][i]['answer'] for s in (1,2,3)]
        if v(b[0])==0:
            print(' ', a['prompt'], '| SFT', a['answer'], '| rft', [x.split(' :=')[0] for x in b], '| rlhf', [x.split(' :=')[0] for x in r]); n+=1
            if n>8: break
print('nat: SFT wrong, rft_s1 right with 0:')
n=0
for i,a in enumerate(sft):
    if a['domain']=='nat' and not a['lean_ok']:
        b=cks['rft_s1']['records'][i]
        if b['lean_ok'] and v(b['answer'])==0:
            print(' ', a['prompt'], '| SFT', a['answer'], '| rft_s1', b['answer']); n+=1
            if n>4: break
# fraction of rft_s1 nat '0' answers on problems containing '-' 
R=[r for r in cks['rft_s1']['records'] if r['domain']=='nat']
z=[r for r in R if v(r['answer'])==0]
print('rft_s1 nat zero answers', len(z), 'with "-" in prompt', sum(' - ' in r['prompt'] for r in z), '; correct', sum(r['lean_ok'] for r in z))
for lab in ['sft','rlhf_s1','rlhf_s3','ppochecker_s3','rft_s1','sftplus_s1']:
    R=[r for r in cks[lab]['records'] if r['domain']=='nat']
    z=[r for r in R if v(r['answer'])==0]
    sub=[r for r in R if ' - ' in r['prompt']]
    print(f'{lab:14s} answers 0 on {len(z)}/500 ({sum(r["lean_ok"] for r in z)} right); on prompts with "-": {sum(v(r["answer"])==0 for r in sub)}/{len(sub)}; without "-": {sum(v(r["answer"])==0 for r in R if " - " not in r["prompt"])}/{500-len(sub)}')
print('\nlinear: SFT right, rft_s2/s3 wrong with their attractor values')
n=0
for i,a in enumerate(sft):
    if a['domain']=='linear' and a['lean_ok']:
        b=[cks[f'rft_s{s}']['records'][i] for s in (1,2,3)]
        if not b[1]['lean_ok'] and not b[2]['lean_ok']:
            print(' ', a['prompt'], '| SFT', a['answer'].split(' :=')[0], '| rft s1/s2/s3', [x['answer'].split(' :=')[0] for x in b], '| rlhf', [cks[f'rlhf_s{s}']['records'][i]['answer'].split(' :=')[0] for s in (1,2,3)]); n+=1
            if n>10: break
