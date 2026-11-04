import sys, collections, time
sys.path.insert(0, '/home/user/nanoGPT')
import data
t=time.time()
a = data.sample_problems(80000, 'train', 2024, 'all')
b = data.sample_problems(80000, 'train', 1001, 'all')
ka = {data.split_key(p.prompt) for p in a}
tot=collections.Counter(); hit=collections.Counter()
for p in b:
    tot[p.domain]+=1
    if data.split_key(p.prompt) in ka: hit[p.domain]+=1
for d in tot: print(d, round(100*hit[d]/tot[d],1))
print('all', round(100*sum(hit.values())/sum(tot.values()),1), time.time()-t)
