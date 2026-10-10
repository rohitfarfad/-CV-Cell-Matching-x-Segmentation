"""Weighted tiles and rotation averaging for saved scalar-output U-Nets."""
import numpy as np
import torch
from torch.nn import functional as F
from cell_matching import normalize
from unet_cells import device

@torch.inference_mode()
def dense_prediction(image,model,tile=192,augment=False):
    image=normalize(image);h,w=image.shape
    padded=np.pad(image,((0,max(0,tile-h)),(0,max(0,tile-w))),mode='reflect')
    stride=tile//2;ph,pw=padded.shape
    ys=sorted(set(list(range(0,ph-tile+1,stride))+[ph-tile]))
    xs=sorted(set(list(range(0,pw-tile+1,stride))+[pw-tile]))
    positions=[(y,x) for y in ys for x in xs]
    channels=model.head.out_channels
    assert not augment or channels==2, "Scalar TTA cannot rotate star rays without channel permutation"
    result=np.zeros((channels,ph,pw),np.float32);count=np.zeros((ph,pw),np.float32)
    weight=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32)
    model.eval()
    for start in range(0,len(positions),8):
        batch=positions[start:start+8]
        patches=torch.from_numpy(np.stack([padded[y:y+tile,x:x+tile] for y,x in batch])[:,None]).to(device)
        values=[]
        # Ray directions require permutation under rotation; use TTA only for scalar-output models.
        for k in range(4 if augment else 1):
            logits=torch.rot90(model(torch.rot90(patches,k,(-2,-1))),-k,(-2,-1))
            value=logits.sigmoid() if channels==2 else torch.cat([logits[:,:1].sigmoid(),F.softplus(logits[:,1:])*12],1)
            values.append(value)
        output=torch.stack(values).mean(0).cpu().numpy()
        for (y,x),value in zip(batch,output):
            result[:,y:y+tile,x:x+tile]+=value*weight;count[y:y+tile,x:x+tile]+=weight
    assert (count>0).all()
    return (result/count)[:,:h,:w]

