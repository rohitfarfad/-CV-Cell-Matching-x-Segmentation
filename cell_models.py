"""Distance-map and StarDist-style U-Nets; automatic native-size instance inference."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch,tifffile
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed,relabel_sequential
from cell_matching import normalize,decode_rle,mask_quality
from unet_cells import SmallUNet,device

PATCH,BATCH,EPOCHS,STEPS,RAYS = 192,8,60,40,32


def truth_labels(row, modality, shape):
    labels=np.zeros(shape,np.int32)
    for i,rle in enumerate(json.loads(getattr(row,modality+'_instances')).values(),1):
        labels[decode_rle(rle,shape)]=i
    return labels


def distance_target(labels):
    distance=np.zeros(labels.shape,np.float32)
    for i,slices in enumerate(ndi.find_objects(labels),1):
        if slices is not None:
            mask=labels[slices]==i
            d=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
            distance[slices][mask]=d[mask]/max(d.max(),1)
    return distance


def load_records(rows,modality,root='Data'):
    records=[]
    for row in rows.itertuples(index=False):
        image=tifffile.imread(Path(root,'training',*row.sample_id.split('__'),modality+'.tif'))
        labels=truth_labels(row,modality,image.shape)
        points=np.array(ndi.center_of_mass(labels>0,labels,np.arange(1,labels.max()+1)))
        background=np.argwhere((normalize(image)>.3)&~ndi.binary_dilation(labels>0,iterations=3))
        records.append((normalize(image),labels,distance_target(labels),points,background))
    return records


class CellUNet(SmallUNet):
    def __init__(self,kind='distance'):
        super().__init__()
        self.kind=kind
        self.head=nn.Conv2d(16,1+RAYS if kind=='stardist' else 2,1)


def sample_batch(records,rng,kind):
    xs,ys=[],[]
    for _ in range(BATCH):
        image,labels,distance,centers,background=records[rng.integers(len(records))]
        h,w=image.shape
        choice=rng.random()
        points=centers if choice<.6 else background if choice<.8 else []
        y,x=points[rng.integers(len(points))] if len(points) else [rng.integers(h),rng.integers(w)]
        y,x=int(np.clip(y-PATCH//2,0,h-PATCH)),int(np.clip(x-PATCH//2,0,w-PATCH))
        a=image[y:y+PATCH,x:x+PATCH]
        labels=labels[y:y+PATCH,x:x+PATCH];d=distance[y:y+PATCH,x:x+PATCH]
        k=int(rng.integers(4))
        a,labels,d=[np.rot90(v,k) for v in [a,labels,d]]
        if rng.random()<.5:
            a,labels,d=[v[:,::-1] for v in [a,labels,d]]
        a=np.clip(a**rng.uniform(.7,1.4)*rng.uniform(.8,1.2)+rng.normal(0,.015,a.shape),0,1)
        if kind=='stardist':
            from stardist.geometry import star_dist
            target=np.concatenate([d[None],star_dist(labels.copy(),RAYS).transpose(2,0,1)/12])
        else:
            target=np.stack([labels>0,d])
        xs.append(a[None].astype(np.float32));ys.append(target.astype(np.float32).copy())
    return torch.from_numpy(np.stack(xs)).to(device),torch.from_numpy(np.stack(ys)).to(device)


def loss_function(logits,target,kind):
    if kind=='stardist':
        probability=target[:,:1]
        regression=(F.softplus(logits[:,1:])-target[:,1:]).abs()*probability
        return F.binary_cross_entropy_with_logits(logits[:,:1],probability)+regression.mean()/probability.mean().clamp_min(.01)
    p=logits[:,0].sigmoid();truth=target[:,0]
    dice=(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
    boundary=F.max_pool2d(truth[:,None],3,1,1)+F.max_pool2d(-truth[:,None],3,1,1)
    bce=(F.binary_cross_entropy_with_logits(logits[:,:1],target[:,:1],reduction='none')*(1+2*boundary)).mean()
    distance=F.mse_loss(logits[:,1].sigmoid(),target[:,1],reduction='none')
    return bce+1-dice+(distance*(1+4*truth)).mean()


from unet_inference import dense_prediction


def make_instances(prediction,kind,limits,threshold=.5,peak=.25):
    if kind=='stardist':
        from stardist.nms import non_maximum_suppression
        from stardist.geometry import polygons_to_label
        points,prob,distance=non_maximum_suppression(prediction[1:].transpose(1,2,0),prediction[0],b=0,prob_thresh=threshold,nms_thresh=.3)
        labels=polygons_to_label(distance,points,prediction.shape[1:],prob=prob)
    else:
        foreground=ndi.binary_fill_holes(prediction[0]>threshold)
        smooth=ndi.gaussian_filter(prediction[1],.7)
        peaks=peak_local_max(smooth,labels=foreground.astype(np.uint8),threshold_abs=peak,min_distance=3,exclude_border=False)
        markers=np.zeros(foreground.shape,np.int32)
        markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
        labels=watershed(-smooth,markers,mask=foreground)
    areas=np.bincount(labels.ravel());keep=(areas>=limits[0])&(areas<=limits[1]);keep[0]=False
    labels[~keep[labels]]=0
    return relabel_sequential(labels)[0].astype(np.int32)


def train_model(rows,modality,directory,kind,root='Data'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=load_records(rows,modality,root)
    areas=np.concatenate([np.bincount(record[1].ravel())[1:] for record in records])
    areas=areas[areas>0];limits=[max(5,int(np.quantile(areas,.002))),int(np.quantile(areas,.998))]
    torch.manual_seed(11);rng=np.random.default_rng(11)
    model=CellUNet(kind).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=3e-5)
    history=[];start=time.time()
    for epoch in range(EPOCHS):
        model.train();losses=[]
        for _ in range(STEPS):
            x,y=sample_batch(records,rng,kind);optimizer.zero_grad(set_to_none=True)
            loss=loss_function(model(x),y,kind);assert torch.isfinite(loss)
            loss.backward();optimizer.step();losses.append(loss.item())
        scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-start))
        pd.DataFrame(history).to_csv(directory/f'{modality}_history.csv',index=False)
        if (epoch+1)%10==0: print(kind,directory.name,modality,epoch+1,history[-1],flush=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},directory/f'{modality}.pt')
    candidates=[dict(threshold=t,peak=p) for t in [.35,.5,.65] for p in ([.2,.35] if kind=='distance' else [.25])]
    scores=np.zeros(len(candidates))
    for row,record in zip(rows.itertuples(index=False),records):
        # Native image normalization is the same as deployment; avoid normalizing an already scaled array.
        image=tifffile.imread(Path(root,'training',*row.sample_id.split('__'),modality+'.tif'))
        prediction=dense_prediction(image,model)
        truth=json.loads(getattr(row,modality+'_instances'))
        for i,setting in enumerate(candidates): scores[i]+=mask_quality(make_instances(prediction,kind,limits,**setting),truth)[0]/len(rows)
    settings=candidates[int(scores.argmax())]
    meta=dict(kind=kind,limits=limits,settings=settings,training_ids=rows.sample_id.tolist(),epochs=EPOCHS,steps=STEPS,patch=PATCH,seed=11)
    (directory/f'{modality}.json').write_text(json.dumps(meta,indent=2))
    pd.DataFrame([dict(**s,pq=float(q)) for s,q in zip(candidates,scores)]).to_csv(directory/f'{modality}_calibration.csv',index=False)
    print('CALIBRATION',kind,modality,settings,float(scores.max()),flush=True)
    return model,meta


def load_model(directory,modality):
    directory=Path(directory);metadata=json.loads((directory/f'{modality}.json').read_text())
    model=CellUNet(metadata['kind']).to(device)
    model.load_state_dict(torch.load(directory/f'{modality}.pt',map_location='cpu',weights_only=True))
    return model.eval(),metadata
