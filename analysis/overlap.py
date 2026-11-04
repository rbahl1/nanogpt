import sys, os, json
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import data, formal, rlhf
from config import RLHFConfig, parse_config
pairs=[json.loads(l) for l in open('out/preferences.jsonl')]
rmp={p['prompt'] for p in pairs}; rmk={data.split_key(p) for p in rmp}
print('RM prompts', len(rmp))
for r in ['rlhf_s1','rlhf_s2','rlhf_s3']:
    d=json.load(open(f'results/logs/{r}.json'))
    cfg=RLHFConfig(**d['config'])
    sch=rlhf.rollout_problems(cfg, formal.resolve(cfg.domains))
    # check against logged examples
    ok=all(sch[i][j].prompt==e['prompt'] for i,it in enumerate(d['iterations']) for j,e in enumerate(it['examples']))
    allp=[p for it in sch for p in it]
    keys=[data.split_key(p.prompt) for p in allp]
    ov=sum(p.prompt in rmp for p in allp); ovk=sum(k in rmk for k in keys)
    print(r, 'schedule matches logged examples:', ok, 'n', len(allp), 'exact prompt in RM data', ov, 'same split key', ovk)
