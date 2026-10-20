"""Bound excessive consonant anticipation without moving vowel anchors."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

# Hop 512 / 44.1 kHz. These are ceilings, not fixed phoneme durations.
CAPS = {4:6,5:6,6:7,8:6,10:7,11:6,12:7,13:7,15:7,16:6,
        17:6,18:6,19:6,21:6,22:4,23:4,24:9,25:9,26:6,27:7,
        29:5,30:5,31:9}


def update(source, destination):
    model = onnx.load(source)
    if any(x.name == 'ct_caps' for x in model.graph.initializer):
        raise ValueError('Consonant timing already installed')
    for node in model.graph.node:
        for i,name in enumerate(node.output):
            if name == 'ph_dur_pred': node.output[i] = 'ct_original'
    limits=np.zeros(32,np.float32)
    for token,cap in CAPS.items(): limits[token]=cap
    for name,value,dtype in [('caps',limits,np.float32),('zero',0,np.float32),
                             ('budget',9,np.float32),('vowel_ids',[2,3,9,14,20,28],np.int64)]:
        model.graph.initializer.append(nh.from_array(np.array(value,dtype),'ct_'+name))
    nodes=[]
    def n(op,ins,out,**kw): nodes.append(h.make_node(op,ins,['ct_'+out],**kw))
    def group_sum(source,name):
        n('Unsqueeze',[source,'lt_axis2'],name+'3')
        n('MatMul',['lt_membership','ct_'+name+'3'],name+'sum3')
        n('Squeeze',['ct_'+name+'sum3','lt_axis2'],name)
    group_sum('ct_original','total')
    n('Max',['lt_length','lt_epsilon'],'length_safe')
    n('Div',['ct_total','ct_length_safe'],'scale')
    n('Where',['onset_first_group','lt_onef','ct_scale'],'normalization')
    n('Gather',['ct_caps','lt_tokeni'],'cap',axis=0)
    n('Mul',['ct_cap','ct_normalization'],'raw_cap')
    n('Greater',['ct_cap','ct_zero'],'consonant')
    n('Min',['ct_original','ct_raw_cap'],'bounded')
    n('Where',['ct_consonant','ct_bounded','ct_zero'],'leads')
    group_sum('ct_leads','lead_sum')
    n('Mul',['ct_budget','ct_normalization'],'raw_budget')
    n('Max',['ct_lead_sum','lt_epsilon'],'lead_safe')
    n('Div',['ct_raw_budget','ct_lead_safe'],'cluster_scale0')
    n('Min',['ct_cluster_scale0','lt_onef'],'cluster_scale')
    n('Mul',['ct_bounded','ct_cluster_scale'],'final')
    n('Where',['ct_consonant','ct_final','ct_original'],'limited')
    n('Sub',['ct_original','ct_limited'],'saved')
    group_sum('ct_saved','saved_sum')
    n('Unsqueeze',['lt_tokeni','lt_axis2'],'tokens3')
    n('Equal',['ct_tokens3','ct_vowel_ids'],'vowel_match')
    n('Cast',['ct_vowel_match'],'vowel_matchf',to=onnx.TensorProto.FLOAT)
    n('ReduceMax',['ct_vowel_matchf','lt_axis2'],'vowel',keepdims=0)
    n('Mul',['ct_original','ct_vowel'],'vowels')
    group_sum('ct_vowels','vowel_sum')
    n('Max',['ct_vowel_sum','lt_epsilon'],'vowel_safe')
    n('Div',['ct_vowels','ct_vowel_safe'],'vowel_ratio')
    n('Mul',['ct_saved_sum','ct_vowel_ratio'],'vowel_extra')
    n('Where',['onset_first_group','ct_zero','ct_vowel_extra'],'extra')
    n('Add',['ct_limited','ct_extra'],'corrected')
    n('Greater',['ct_vowel_sum','ct_zero'],'has_vowel')
    n('Or',['ct_has_vowel','onset_first_group'],'eligible0')
    n('Greater',['lt_length','ct_zero'],'positive_length')
    n('And',['ct_eligible0','ct_positive_length'],'eligible')
    nodes.append(h.make_node('Where',['ct_eligible','ct_corrected','ct_original'],['ph_dur_pred']))
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();update(a.source,a.destination)
