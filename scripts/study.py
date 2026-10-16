"""Separate prompt-style PPO pilot; existing infrastructure, isolated outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from math_rl.provenance import sha256,write_json
STYLES=dict(neutral='Work through the problem carefully.',
 confident='Express confidence in your reasoning while working through the problem. Check calculations and correct mistakes when needed.',
 critical='Examine your reasoning critically while working through the problem. Check calculations and correct mistakes when needed.')
SEEDS=[301,302,303,304,305]
def read(p): return json.loads(p.read_text())
def prepare(out):
 from datasets import Dataset
 from transformers import AutoTokenizer
 from math_rl.prompts import make_prompt,encode_completion
 from math_rl.large_probe import family_key,digest
 if (out/'manifest.json').exists(): raise ValueError('Already prepared')
 asset_manifest=read(ROOT/'inputs/assets.json')
 for relative,checksum in asset_manifest['files'].items():
  if sha256(ROOT/relative)!=checksum: raise ValueError(f'Imported asset changed: {relative}')
 rows=read(ROOT/'inputs/questions.json')['questions']
 tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen-math',local_files_only=True)
 selected=[];seen=set()
 for split,n in dict(train=512,val=64,test=300).items():
  pool=sorted((q for q in rows if q['split']==split),key=lambda q:digest(['confidence-v1',q['id']]))
  group=[]
  for q in pool:
   f=family_key(q['question'])
   if f in seen or any(len(encode_completion(tok,v+'\n\n'+q['question'])[1])>512 for v in STYLES.values()): continue
   seen.add(f);group.append(q)
   if len(group)==n: break
  if len(group)!=n: raise ValueError(f'Insufficient distinct {split} questions: {len(group)}/{n}')
  selected+=group
 hashes={}
 for style,instruction in STYLES.items():
  data=out/style/'data';data.mkdir(parents=True,exist_ok=True)
  for split in ('train','val','test'):
   records=[dict(data_source='numina_numeric',prompt=make_prompt(instruction+'\n\n'+q['question']),ability='math',
    reward_model=dict(style='rule',ground_truth=q['ground_truth']),
    extra_info=dict(index=i,prompt_id=q['id'],split=split,level=q['level'])) for i,q in enumerate(selected) if q['split']==split]
   p=data/f'{split}.parquet';Dataset.from_list(records).to_parquet(str(p));hashes[str(p.relative_to(out))]=sha256(p)
 write_json(out/'questions.json',selected)
 write_json(out/'manifest.json',dict(steps=20,seeds=SEEDS,styles=STYLES,data_hashes=hashes,
  input_assets=asset_manifest, input_manifest_sha256=sha256(ROOT/'inputs/assets.json'),
  code_hashes={str(p.relative_to(ROOT)):sha256(p) for pattern in ('src/**/*.py','scripts/*.py','scripts/*.sh','configs/*.yaml','requirements*.txt','pyproject.toml') for p in ROOT.glob(pattern)},
  scope='Prompted behavior; inspected questions; five seeds; exploratory pilot, no guaranteed significance.'))
def launch(out,style,seed,steps,smoke=False):
 import math_comparison as comparison
 m=read(out/'manifest.json')
 for p,h in m['input_assets']['files'].items():
  if sha256(ROOT/p)!=h: raise ValueError(f'Imported asset changed: {p}')
 for p,h in m['data_hashes'].items():
  if sha256(out/p)!=h: raise ValueError('Data changed')
 for p,h in m['code_hashes'].items():
  if sha256(ROOT/p)!=h: raise ValueError('Study code changed after preparation')
 case=('smoke' if smoke else 'ppo')+f'-seed{seed}'
 cmd=comparison.command(out/style,'ppo',seed,steps,ppo_profile='more_fitting',evaluation_split='val' if smoke else 'test',case_name=case)
 cmd.append(f'experiment.neutral_test_file={out / "neutral/data" / ("val.parquet" if smoke else "test.parquet")}')
 env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),RAY_ADDRESS='local',MATH_RL_GPU='1')
 env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
 with tempfile.TemporaryDirectory(prefix='cp-',dir='/tmp') as tmp:
  env['RAY_TMPDIR']=tmp
  with (out/f'{style}-{case}.log').open('w') as log: subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def interval(x):
 rng=np.random.default_rng(7)
 draws=[x[rng.integers(len(x),size=len(x))][:,rng.integers(x.shape[1],size=x.shape[1])].mean()*100 for _ in range(5000)]
 return dict(mean_pp=float(x.mean()*100),interval_95_pp=np.quantile(draws,[.025,.975]).tolist(),interval_97_5_pp=np.quantile(draws,[.0125,.9875]).tolist())
def report(out):
 m=read(out/'manifest.json');results={};gains={};expected=[(q['id'],q['ground_truth']) for q in read(out/'questions.json') if q['split']=='test']
 from math_rl.ppo_baseline import audit_health
 health={};cost={}
 for style in STYLES:
  gains[style]={}
  for evaluation,suffix in [('style',''),('neutral','-neutral')]:
   before=[];after=[];tokens=[];capped=[];statuses={}
   for seed in m['seeds']:
    case=out/style/f'ppo-seed{seed}';record=read(case/'result.json')
    if record['updates']!=m['steps'] or record['answers']!=64*m['steps']: raise ValueError('Incomplete budget')
    cost[f'{style}/{seed}']=record['training_gpu_hours']
    health[f'{style}/{seed}']=audit_health(case/'ppo-critic-audit',m['steps'])
    pair=[]
    for stage in ('base','final'):
     rows=[json.loads(l) for l in (case/f'{stage}{suffix}-test.jsonl').read_text().splitlines()]
     if [(r['id'],r['ground_truth']) for r in rows]!=expected: raise ValueError('Unmatched evaluation questions')
     if any(r['score']!=int(r['verifier_status']=='correct') for r in rows): raise ValueError('Score/status mismatch')
     pair.append([r['score'] for r in rows])
     if stage=='final':
      tokens.extend(r['response_tokens'] for r in rows);capped.extend(r['capped'] for r in rows)
      for r in rows: statuses[r['verifier_status']]=statuses.get(r['verifier_status'],0)+1
    before.append(pair[0]);after.append(pair[1])
   before,after=np.array(before),np.array(after);gains[style][evaluation]=after-before
   results[f'{style}/{evaluation}']=dict(base_accuracy=float(before.mean()),final_accuracy=float(after.mean()),gain=interval(after-before),mean_final_tokens=float(np.mean(tokens)),truncation=float(np.mean(capped)),statuses=statuses)
 contrasts={f'{s}-minus-neutral/{e}':interval(gains[s][e]-gains['neutral'][e]) for s in ('confident','critical') for e in ('style','neutral')}
 summary=dict(results=results,contrasts=contrasts,critic_health=health,training_gpu_hours=cost,
  primary='Neutral-prompt learning gains; 97.5% intervals for two planned contrasts.',scope=m['scope'])
 write_json(out/'summary.json',summary)
 (out/'report.txt').write_text('Confidence/self-criticism PPO pilot\n'+json.dumps(summary,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);
 p.add_argument('--stage',choices=['prepare','smoke','run','report'],required=True);p.add_argument('--task',type=int,default=0)
 a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 if a.stage=='prepare': prepare(a.out)
 elif a.stage=='report': report(a.out)
 elif a.stage=='smoke':
  for s in STYLES: launch(a.out,s,19,2,True)
  write_json(a.out/'smoke.json',dict(passed=True))
 else:
  m=read(a.out/'manifest.json');i=a.task//3;style=list(STYLES)[(a.task%3+i)%3];launch(a.out,style,m['seeds'][i],m['steps'])
if __name__=='__main__': main()
