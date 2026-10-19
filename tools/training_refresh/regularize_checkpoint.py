"""Limit an update toward a clean prior; preserve unchanged tensors exactly."""
import argparse,shutil,json
from pathlib import Path
import torch

def regularize(prior,trained,output,weight):
 assert 0<=weight<=1
 a=torch.load(prior,map_location='cpu',weights_only=False);b=torch.load(trained,map_location='cpu',weights_only=False)
 assert a['state_dict'].keys()==b['state_dict'].keys()
 for k,x in b['state_dict'].items():
  y=a['state_dict'][k];assert x.shape==y.shape
  if 'spk_embed' in k:assert torch.equal(x,y)
  if x.is_floating_point() and not torch.equal(x,y):b['state_dict'][k]=x*weight+y*(1-weight)
  else:b['state_dict'][k]=y.clone()
 b.pop('optimizer_states',None);b.pop('lr_schedulers',None);output.parent.mkdir(parents=True,exist_ok=True);torch.save(b,output)
 for n in ['config.yaml','spk_map.json','lang_map.json','dictionary-ja.txt']:shutil.copy2(trained.parent/n,output.parent/n)
 output.with_suffix('.regularization.json').write_text(json.dumps(dict(prior=str(prior),trained=str(trained),trained_weight=weight,unchanged_tensors_preserved_exactly=True,speaker_embeddings_identical=True),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('trained',type=Path);p.add_argument('output',type=Path);p.add_argument('--weight',type=float,required=True);a=p.parse_args();regularize(a.prior,a.trained,a.output,a.weight)
