import json, sys, collections
sys.path.insert(0, '/home/user/nanoGPT')
import data
probs = data.sample_problems(6000, 'train', 7, 'all', unique=True)
ref = {}
for p in probs:
    ref.setdefault(p.prompt, p.reference)
rows = [json.loads(l) for l in open('/home/user/nanoGPT/out/preferences.jsonl')]
print('rows', len(rows), 'prompts in probs', len(ref), 'distinct prompts in prefs', len({r['prompt'] for r in rows}))
missing = sum(1 for r in rows if r['prompt'] not in ref)
print('rows with prompt not in regenerated', missing)
ch = sum(1 for r in rows if r['chosen'] == ref.get(r['prompt']))
rj = sum(1 for r in rows if r['rejected'] == ref.get(r['prompt']))
print('chosen==ref', ch, ch/len(rows), 'rejected==ref', rj, 'include ref', ch+rj)
dom = collections.Counter()
for r in rows:
    if r['chosen'] != ref.get(r['prompt']) and r['rejected'] != ref.get(r['prompt']):
        dom[r['domain']] += 1
print('sample-vs-sample by domain', dom, sum(dom.values()))
