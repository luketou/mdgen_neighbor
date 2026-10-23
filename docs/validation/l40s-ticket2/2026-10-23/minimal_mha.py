import argparse,json,time,traceback,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from mdgen.model.mha import MultiheadAttention
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--require-gpu',action='store_true');a=p.parse_args()
runs=[]
for _ in range(2):
 torch.manual_seed(137);m=MultiheadAttention(8,2,add_bias_kv=True,use_rotary_embeddings=False).cuda();x=torch.randn(3,1,8,device='cuda');start=time.monotonic()
 try:
  with torch.autocast('cuda',dtype=torch.bfloat16): y,_=m(x,x,x,need_weights=False)
  torch.cuda.synchronize();r={'passed':True}
 except Exception as e:r={'passed':False,'error':type(e).__name__,'traceback':traceback.format_exc()}
 r['seconds']=time.monotonic()-start;runs.append(r)
x={'passed':all(r['passed'] for r in runs),'status':'failed','runs':runs}
if x['passed']:x['status']='passed'
a.output.write_text(json.dumps(x,indent=2));print(json.dumps(x));raise SystemExit(0 if x['passed'] else 1)
