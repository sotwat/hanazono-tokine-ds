"""Check bounded gains, exact bypass, boundaries and dynamic lengths."""
import sys,json,itertools
import numpy as np,onnxruntime as ort
s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider'])
checks=0
for lengths,frequency in itertools.product([(1,1),(5,5),(6,8),(12,20),(24,58),(100,160)],[65.406,110,220,440,880,1760]):
 for weak in [0,.001,.005,.01,.03,.06,.1]:
  for vowel in [False,True]:
   lens=[4,lengths[0],3,lengths[1],4];starts=np.r_[0,np.cumsum(lens)];frames=int(starts[-1]);c=np.zeros((frames,128),np.float32);x=np.zeros((1,frames*512),np.float32)
   for j,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
    c[a:b,1]=a;c[a:b,2]=b;c[a:b,3]=frequency;c[a:b,6]=vowel and j in [1,3]
    t=np.arange((b-a)*512);x[0,a*512:b*512]=(weak if j==1 else .1)*np.sin(2*np.pi*frequency*t/44100)
   y=s.run(None,{'wave':x,'control':c})[0]
   assert y.shape==x.shape and np.isfinite(y).all()
   assert np.max(abs(y))<=np.max(abs(x))*10**(4/20)+1e-6
   for j in [0,2,4]:assert np.array_equal(x[:,starts[j]*512:starts[j+1]*512],y[:,starts[j]*512:starts[j+1]*512])
   if not vowel or min(lengths)<6 or weak<=.01:assert np.array_equal(x,y)
   for a,b in zip(starts[:-1],starts[1:]):
    assert x[0,a*512]==y[0,a*512] and x[0,b*512-1]==y[0,b*512-1]
   if vowel and min(lengths)>=6 and .01<weak<.06:
    a,b=starts[1:3]*512;mid=(a+b)//2;assert np.sqrt(np.mean(y[0,mid:mid+128]**2))>np.sqrt(np.mean(x[0,mid:mid+128]**2))*1.01
   checks+=1
print(json.dumps({'checks':checks,'passed':True}))
