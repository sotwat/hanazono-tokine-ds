"""Keep consonant leads independent of long-note duration inside the voicebank."""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,compose


def rename_output(model,old,new):
    for node in model.graph.node:
        for i,x in enumerate(node.input):
            if x==old:node.input[i]=new
        for i,x in enumerate(node.output):
            if x==old:node.output[i]=new
    for value in model.graph.output:
        if value.name==old:value.name=new


def update(bank,output):
    output.mkdir(parents=True,exist_ok=False)
    lm=onnx.load(bank/'dsdur/tokine_song_variance_v1.linguistic.onnx')
    dm=onnx.load(bank/'dsdur/tokine_song_variance_v1.dur.onnx')
    lm=onnx.version_converter.convert_version(lm,18)
    dm=onnx.version_converter.convert_version(dm,18)
    lc=compose.add_prefix(lm,'ltc_',rename_inputs=False)
    dc=compose.add_prefix(dm,'ltc_',rename_inputs=False)
    rename_output(lm,'encoder_out','lt_original_encoder')
    rename_output(dm,'ph_dur_pred','lt_original_prediction')
    nodes=[];constants=[]
    def c(k,v,dtype=np.int64):constants.append(nh.from_array(np.array(v,dtype),'lt_'+k))
    def n(op,ins,out,**kw):nodes.append(h.make_node(op,ins,['lt_'+out],**kw))
    for k,v in [('zero',0),('one',1),('axis0',[0]),('axis1',[1]),('axis2',[2]),('idx1',1),('cap',43)]:c(k,v)
    n('Min',['word_dur','lt_cap'],'canonical_word_dur')
    for node in lc.graph.node:
        for i,x in enumerate(node.input):
            if x=='word_dur':node.input[i]='lt_canonical_word_dur'
    nodes.extend(lc.graph.node)
    n('CumSum',['word_div','lt_one'],'word_ends')
    n('Shape',['tokens'],'shape');n('Gather',['lt_shape','lt_idx1'],'count',axis=0)
    n('Range',['lt_zero','lt_count','lt_one'],'range')
    n('Unsqueeze',['lt_range','lt_axis1'],'column')
    n('GreaterOrEqual',['lt_column','lt_word_ends'],'after')
    n('Cast',['lt_after'],'afteri',to=onnx.TensorProto.INT64)
    n('ReduceSum',['lt_afteri','lt_axis1'],'groups',keepdims=0)
    n('Gather',['word_dur','lt_groups'],'length',axis=1)
    n('Unsqueeze',['lt_groups','lt_axis0'],'groups2')
    for k,source in [('token','tokens'),('group','lt_groups2'),('length','lt_length')]:
        n('Cast',[source],k+'f',to=onnx.TensorProto.FLOAT)
        n('Unsqueeze',['lt_'+k+'f','lt_axis2'],k+'3')
    nodes.append(h.make_node('Concat',['lt_original_encoder','ltc_encoder_out','lt_token3','lt_group3','lt_length3'],['encoder_out'],axis=2))
    lm.graph.node.extend(nodes);lm.graph.initializer.extend(lc.graph.initializer);lm.graph.initializer.extend(constants)
    lm.graph.output[0].name='encoder_out';lm.graph.output[0].type.tensor_type.shape.dim[2].dim_value=259
    onnx.checker.check_model(lm);onnx.save(lm,output/'tokine_song_variance_v1.linguistic.onnx')

    nodes=[];constants=[]
    for k,v in [('zero',0),('one',1),('axis1',[1]),('axis2',[2]),('start',[0]),('stop',[128]),('canonstart',[128]),('canonstop',[256]),('tokstart',[256]),('tokstop',[257]),('groupstart',[257]),('groupstop',[258]),('lenstart',[258]),('lenstop',[259])]:c(k,v)
    c('vowels',[1,2,3,9,14,20,28],np.float32)
    for k,v in [('reference',43.),('long',40.),('epsilon',.0001),('fraction',.4),('onef',1.)]:c(k,v,np.float32)
    mins=np.full(32,2,np.float32);maxs=np.full(32,12,np.float32)
    for i in [4,5,8,11,16,17,21,26]:mins[i]=3;maxs[i]=10
    for i in [6,10,12,13,15,24,25,27,31]:mins[i]=4;maxs[i]=16
    for i in [18,19,29,30]:mins[i]=3;maxs[i]=12
    for i in [22,23]:mins[i]=2;maxs[i]=8
    maxs[7]=16;mins[7]=3
    c('minimum',mins,np.float32);c('maximum',maxs,np.float32)
    for k,a,b in [('original','start','stop'),('canonical','canonstart','canonstop'),('token','tokstart','tokstop'),('group','groupstart','groupstop'),('length','lenstart','lenstop')]:
        n('Slice',['encoder_out','lt_'+a,'lt_'+b,'lt_axis2'],k+'3')
        if k not in ['original','canonical']:n('Squeeze',['lt_'+k+'3','lt_axis2'],k)
    for node in dm.graph.node:
        for i,x in enumerate(node.input):
            if x=='encoder_out':node.input[i]='lt_original3'
    for node in dc.graph.node:
        for i,x in enumerate(node.input):
            if x=='encoder_out':node.input[i]='lt_canonical3'
    nodes.extend(dm.graph.node);nodes.extend(dc.graph.node)
    n('Transpose',['lt_group3'],'grouprow',perm=[0,2,1]);n('Equal',['lt_group3','lt_grouprow'],'samegroup')
    n('Cast',['lt_samegroup'],'membership',to=onnx.TensorProto.FLOAT)
    n('Cast',['lt_token'],'tokeni',to=onnx.TensorProto.INT64)
    n('Equal',['lt_token3','lt_vowels'],'vowelmatch')
    n('Cast',['lt_vowelmatch'],'vowelmatchf',to=onnx.TensorProto.FLOAT)
    n('ReduceMax',['lt_vowelmatchf','lt_axis2'],'vowel',keepdims=0)
    n('Sub',['lt_onef','lt_vowel'],'consonant')
    def groupsum(source,key):
        n('Unsqueeze',[source,'lt_axis2'],key+'3')
        n('MatMul',['lt_membership','lt_'+key+'3'],key+'sum3')
        n('Squeeze',['lt_'+key+'sum3','lt_axis2'],key+'sum')
        return 'lt_'+key+'sum'
    total=groupsum('ltc_ph_dur_pred','pred')
    n('Max',[total,'lt_epsilon'],'predsafe')
    n('Div',['ltc_ph_dur_pred','lt_predsafe'],'ratio')
    n('Mul',['lt_ratio','lt_reference'],'reference_duration')
    for k in ['minimum','maximum']:n('Gather',['lt_'+k,'lt_tokeni'],k+'2',axis=0)
    n('Max',['lt_reference_duration','lt_minimum2'],'bounded0');n('Min',['lt_bounded0','lt_maximum2'],'bounded')
    n('Mul',['lt_bounded','lt_consonant'],'consonant_desired')
    csum=groupsum('lt_consonant_desired','desired')
    n('Max',[csum,'lt_epsilon'],'csafe');n('Mul',['lt_length','lt_fraction'],'cbudget')
    n('Div',['lt_cbudget','lt_csafe'],'cscale0');n('Min',['lt_cscale0','lt_onef'],'cscale')
    n('Mul',['lt_consonant_desired','lt_cscale'],'cfinal')
    actual_c=groupsum('lt_cfinal','actual')
    n('Sub',['lt_length',actual_c],'vbudget')
    n('Mul',['ltc_ph_dur_pred','lt_vowel'],'vpred')
    vsum=groupsum('lt_vpred','vpred');n('Max',[vsum,'lt_epsilon'],'vsafe')
    n('Div',['lt_vpred','lt_vsafe'],'vratio');n('Mul',['lt_vratio','lt_vbudget'],'vfinal')
    n('Add',['lt_vfinal','lt_cfinal'],'corrected')
    n('GreaterOrEqual',['lt_length','lt_long'],'longgroup');n('Greater',[vsum,'lt_epsilon'],'hasvowel')
    n('And',['lt_longgroup','lt_hasvowel'],'eligible0')
    n('Equal',['lt_tokeni','lt_one'],'silence');n('Cast',['lt_silence'],'silencef',to=onnx.TensorProto.FLOAT)
    silence=groupsum('lt_silencef','silence')
    n('Less',[silence,'lt_onef'],'nosilence');n('And',['lt_eligible0','lt_nosilence'],'eligible')
    nodes.append(h.make_node('Where',['lt_eligible','lt_corrected','lt_original_prediction'],['ph_dur_pred']))
    del dm.graph.node[:];dm.graph.node.extend(nodes);dm.graph.initializer.extend(dc.graph.initializer);dm.graph.initializer.extend(constants)
    dm.graph.input[0].type.tensor_type.shape.dim[2].dim_value=259;dm.graph.output[0].name='ph_dur_pred'
    onnx.checker.check_model(dm);onnx.save(dm,output/'tokine_song_variance_v1.dur.onnx')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args();update(a.bank,a.output)
