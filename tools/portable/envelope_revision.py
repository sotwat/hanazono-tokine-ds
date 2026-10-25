"""Remove duplicate fixed-spectrum vowel repair and decouple neural onset timing."""
import argparse
from pathlib import Path
import torch
import onnx
from build_pair import append_model
from ts_affricate import dependencies
from vowel_balance import VowelBalance

class Restore(torch.nn.Module):
    """Use a 6-frame neural crossfade without changing the existing release gate."""

    def forward(self, short, back, eligible, weight, gate, baseline, durations):
        left = back.long().clamp(0, short.shape[1] - 1)
        right = (left + 1).clamp(max=short.shape[1] - 1)
        a = (back - left).unsqueeze(2)
        restored = short.gather(1, left.unsqueeze(2).expand(-1, -1, 128)) * (1 - a) + short.gather(1, right.unsqueeze(2).expand(-1, -1, 128)) * a
        starts = durations.cumsum(1) - durations
        ends = starts + durations
        t = torch.arange(back.shape[1])[None, None, :]
        inside = (t >= starts[:, :, None]) & (t < ends[:, :, None])
        off = ((t - starts[:, :, None]) * inside).sum(1).float().unsqueeze(2)
        attack = (off / 6).clamp(0, 1)
        attack = attack.square() * (3 - 2 * attack)
        onset = torch.where(off < 20, torch.maximum(weight, attack * gate), weight)
        w = (eligible * onset).clamp(0, 1)
        return torch.where(w > 0, baseline * (1 - w) + restored * w, baseline)

def build(bank, out):
    out.mkdir(parents=True, exist_ok=False)
    m = onnx.load(bank / 'tokine_ds_clarity_hybrid.tokine.onnx')
    if any((o.startswith('er_') for n in m.graph.node for o in n.output)):
        raise ValueError('Envelope revision already installed')
    outputs = {o for n in m.graph.node for o in n.output}
    for prefix in ('original_', 'donor_'):
        for suffix in ('gv_base', 'gv_anchor', 'gv_eligible', 'ss_gate'):
            if prefix + suffix not in outputs:
                raise ValueError('Unsupported acoustic graph: ' + prefix + suffix)
    # Repeated vowels are already continuous in the neural model input.
    for p in ['original_', 'donor_']:
        for n in m.graph.node:
            if p + 'gv_base' in n.output:
                n.op_type = 'Identity'
                del n.input[:]
                n.input.append(p + 'gv_base_raw')
                del n.attribute[:]
            if p + 'gv_anchor' in n.output:
                n.op_type = 'Identity'
                del n.input[:]
                n.input.append(p + 'gv_localanchor')
                del n.attribute[:]
        name = p + 'gv_eligible'
        for n in m.graph.node:
            if name in n.output:
                n.output[0] = name + '_before_repeat'
        match = p + 'er_same'
        notmatch = p + 'er_different'
        nodes = [onnx.helper.make_node('Equal', ['original_su_tokens3' if p == 'original_' else 'donor_su_tokens3', p + 'su_vv_prev3'], [match]), onnx.helper.make_node('Not', [match], [notmatch]), onnx.helper.make_node('And', [name + '_before_repeat', notmatch], [name])]
        old = list(m.graph.node)
        i = next((i for (i, n) in enumerate(old) if name + '_before_repeat' in n.output)) + 1
        del m.graph.node[:]
        m.graph.node.extend(old[:i] + nodes + old[i:])
    layer = out / 'restore-layer.onnx'
    args = (torch.randn(1, 64, 128), torch.linspace(0, 63, 96)[None], torch.ones(1, 96, 1), torch.ones(1, 96, 1), torch.ones(1, 96, 1), torch.randn(1, 96, 128), torch.tensor([[8, 80, 8]]))
    torch.onnx.export(Restore(), args, layer, input_names=['short', 'back', 'eligible', 'weight', 'gate', 'baseline', 'durations'], output_names=['restored'], opset_version=18, dynamo=False, dynamic_axes={k: {1: 'C' if k == 'short' else 'N' if k == 'durations' else 'F'} for k in ['short', 'back', 'eligible', 'weight', 'gate', 'baseline', 'durations', 'restored']})
    nodes = [n for n in m.graph.node if not any((o.startswith('tc_restore_') or o in ['tc_original_mel', 'tc_donor_mel'] for o in n.output))]
    constants = list(m.graph.initializer)
    for p in ['original_', 'donor_']:
        append_model(nodes, constants, layer, 'er_restore_' + p, dict(short='tc_' + p + 'su_raw', back='tc_back', eligible='tc_eligible', weight=p + 'ss_weight', gate=p + 'ss_gate', baseline=p + 'mel', durations='durations', restored='tc_' + p + 'mel'))
    known = {v.name for v in m.graph.input} | {v.name for v in constants} | {''}
    ordered = []
    while nodes:
        ready = [n for n in nodes if dependencies(n) <= known]
        if not ready:
            raise ValueError('dependency')
        for n in ready:
            ordered.append(n)
            known.update(n.output)
            nodes.remove(n)
    del m.graph.node[:]
    m.graph.node.extend(ordered)
    del m.graph.initializer[:]
    m.graph.initializer.extend(constants)
    onnx.checker.check_model(m)
    onnx.save(m, out / 'acoustic.onnx')
    layer = out / 'balance-layer.onnx'
    torch.onnx.export(VowelBalance(0.75, 0.9, 6.0, 12.0), (torch.randn(1, 64 * 512) * 0.05, torch.zeros(64, 128)), layer, input_names=['wave', 'control'], output_names=['balanced'], opset_version=18, dynamo=False, dynamic_axes={'wave': {1: 'S'}, 'control': {0: 'F'}, 'balanced': {1: 'S'}})
    m = onnx.load(next((bank / 'dsvocoder').glob('*.onnx')))
    old = list(m.graph.node)
    indices = [i for (i, n) in enumerate(old) if any((o.startswith('lvb_') or o == 'vb_balanced' for o in n.output))]
    assert indices == list(range(min(indices), max(indices) + 1))
    nodes = []
    append_model(nodes, m.graph.initializer, layer, 'evb_', dict(wave='pb_input_wave', control='rp_control', balanced='vb_balanced'))
    nodes = old[:min(indices)] + nodes + old[max(indices) + 1:]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, out / 'vocoder.onnx')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.bank, args.output)
