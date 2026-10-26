"""Exercise context scope, boundary preservation and finite corrections."""
import torch
from ni_tsu_stability import Context,Correction

def check():
    count=0
    for previous in [1,19,27]:
        for current in [3,14,28]:
            for duration in [19,20,40]:
                for midi in [60,68,69,73,74,80]:
                    tokens=torch.tensor([[previous,current]])
                    d=torch.tensor([[8,duration]])
                    f=torch.full((1,8+duration),440*2**((midi-69)/12))
                    mask=Context()(tokens,d,f)
                    assert torch.isfinite(mask).all() and mask.min()>=0 and mask.max()<=1
                    assert not mask[:,:8].any()
                    if duration<20 or not ((previous==19 and current==14) or (previous==27 and current==28)):
                        assert not mask.any()
                    count+=1
    c=torch.zeros(40,128);c[:,2]=40
    value=torch.rand(1,40,1025)+.01
    for spectrum in [False,True]:
        layer=Correction(spectrum)
        assert torch.equal(layer(value,c),value)
        c[:,10 if spectrum else 9]=1
        y=layer(value,c)
        assert torch.isfinite(y).all() and (y>0).all()
        assert torch.equal(y[:,0],value[:,0]) and torch.equal(y[:,-1],value[:,-1])
        c.zero_();c[:,2]=40
    print(f'{count} context cases and correction boundaries passed')
if __name__=='__main__':check()
