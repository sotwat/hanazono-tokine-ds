"""Learn individual accepted breath trajectories instead of averaging their spectra."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
from torch import nn


class SelectedBreaths(nn.Module):
    def __init__(self, durations):
        super().__init__()
        self.register_buffer('durations', torch.tensor(durations, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(11 + len(durations), 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))

    def forward(self, position, duration, identity=None):
        p = position.clamp(0, 1)
        if identity is None:
            # Pick the nearest recorded duration, keeping one identity per AP.
            identity = torch.argmin(abs(duration[..., None] - self.durations), dim=-1)
        duration = self.durations[identity]
        style = (identity[..., None] == torch.arange(len(self.durations), device=p.device)).float()
        time = torch.stack([p, p*p, p*p*p, duration.clamp(.08, 1),
                            torch.sin(p*np.pi), torch.cos(p*np.pi),
                            torch.sin(p*np.pi*2), torch.cos(p*np.pi*2),
                            torch.sin(p*np.pi*3), torch.cos(p*np.pi*3), torch.ones_like(p)], -1)
        return self.net(torch.cat([time, style], -1))


def train(dataset, output):
    sys.path.insert(0, '/Users/aaa/Documents/DiffSinger')
    from utils.binarizer_utils import get_mel_torch
    torch.set_num_threads(4)
    torch.manual_seed(2741)
    rows = json.loads((dataset / 'candidates.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    durations, clips, parts = [], [], []
    for identity, row in enumerate(rows):
        path = dataset / 'clips' / f'{row["name"]}.wav'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        y, sr = sf.read(path, dtype='float32')
        gain = min(15., 10**(-33/20)/(np.sqrt(np.mean(y*y)) + 1e-9))
        mel = get_mel_torch(y*gain, sr, device='cpu')
        mel = np.stack([mel[max(0, i-1):min(len(mel), i+2)].mean(0) for i in range(len(mel))])
        duration = len(y)/sr
        durations.append(duration)
        # Hold out alternating individual frames for interpolation checks.
        validation = np.arange(len(mel)) % 5 == 2
        parts.append((np.linspace(0, 1, len(mel), dtype='float32'),
                      np.full(len(mel), duration, np.float32),
                      np.full(len(mel), identity, np.int64), mel.astype('float32'), validation))
        clips.append(dict(name=row['name'],sha256=row['sha256'],duration=duration,gain=float(gain),frames=len(mel)))
    arrays = [np.concatenate([v[i] for v in parts]) for i in range(5)]
    values = [torch.from_numpy(v).to('mps') for v in arrays[:4]]
    vp, vd, vi, vy = [v[arrays[4]] for v in values]
    p, d, identity, y = [v[~arrays[4]] for v in values]
    scale = y.std(0).clamp(min=.8)
    model = SelectedBreaths(durations).to('mps')
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.copy_(y.mean(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    history, best, best_state = [], float('inf'), None
    for step in range(2401):
        optimizer.zero_grad()
        loss = (((model(p,d,identity)-y)/scale)**2).mean()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                validation = float(((((model(vp,vd,vi)-vy)/scale)**2).mean()).cpu())
            history.append(dict(step=step,training=float(loss.detach().cpu()),validation=validation))
            print(history[-1], flush=True)
            if validation < best:
                best, best_state = validation, copy.deepcopy(model.cpu().state_dict())
                model.to('mps')
    baseline = float(((((y.mean(0)[None]-vy)/scale)**2).mean()).cpu())
    model.cpu().load_state_dict(best_state)
    torch.save(dict(kind='selected-breaths',durations=durations,state_dict=best_state), output/'decoder.pt')
    result = dict(clips=clips,history=history,best_validation=best,constant_baseline=baseline,
                  training_frames=len(y),validation_frames=len(vy),seed=2741,
                  validation_scope='held-out frames within every accepted clip; not unseen recordings',
                  method='identity-conditioned learned mel trajectories, nearest-duration selection')
    (output/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('best',best,'constant baseline',baseline,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    train(args.dataset,args.output)
