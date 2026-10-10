"""Compact pretrained Cellpose 3 comparison; run with .venv-cellpose/bin/python."""
from pathlib import Path
import os,json,logging
os.environ['CELLPOSE_LOCAL_MODELS_PATH']=str(Path('outputs/q5/advanced/pretrained').resolve())
os.environ['MPLCONFIGDIR']='/tmp/cv2-mpl'
import numpy as np,pandas as pd,tifffile,torch
from scipy import ndimage as ndi
from torch.nn import functional as F
from skimage.segmentation import relabel_sequential
from cellpose import models,train,dynamics
from cell_models import truth_labels
from cell_matching import mask_quality,read_pair

torch.set_num_threads(4)
DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def boundary_dice_loss(labels,prediction,device):
    target=torch.from_numpy(labels[:,0]>.5).to(device).float()
    flows=5*torch.from_numpy(labels[:,1:]).to(device)
    boundary=F.max_pool2d(target[:,None],3,1,1)+F.max_pool2d(-target[:,None],3,1,1)
    bce=F.binary_cross_entropy_with_logits(prediction[:,2:3],target[:,None],reduction='none')
    probability=prediction[:,2].sigmoid()
    dice=(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    return F.mse_loss(prediction[:,:2],flows)/2+(bce*(1+3*boundary)).mean()+1-dice


def training_patches(rows,modality):
    images,masks,areas=[],[],[];rng=np.random.default_rng(19)
    for row in rows.itertuples(index=False):
        image=read_pair('Data/training',row.sample_id)[0 if modality=='invivo' else 1]
        labels=truth_labels(row,modality,image.shape)
        centers=np.array(ndi.center_of_mass(labels>0,labels,np.arange(1,labels.max()+1)))
        counts=np.bincount(labels.ravel())[1:];areas.extend(counts[counts>0])
        background=np.argwhere((image>np.percentile(image,95))&~ndi.binary_dilation(labels>0,iterations=4))
        points=centers[rng.choice(len(centers),min(4,len(centers)),replace=False)]
        if len(background):points=np.vstack([points,background[rng.integers(len(background))]])
        for y,x in points:
            y,x=int(np.clip(y-96,0,image.shape[0]-192)),int(np.clip(x-96,0,image.shape[1]-192))
            images.append(image[y:y+192,x:x+192].copy())
            masks.append(relabel_sequential(labels[y:y+192,x:x+192])[0].astype(np.int32))
    return images,masks,np.asarray(areas)


def predict_flows(image,model,metadata):
    _,flows,_=model.eval(image,channels=[0,0],diameter=metadata['diameter'],compute_masks=False,
                        normalize={'percentile':[1,99.8],'tile_norm_blocksize':metadata.get('tile_norm_blocksize',0)},batch_size=16,tile_overlap=.2)
    return flows[1],flows[2]


def instances_from_flows(flows,metadata,threshold=None):
    threshold=metadata.get('threshold',0.) if threshold is None else threshold
    labels=dynamics.compute_masks(*flows,niter=int(200*metadata['diameter']/30),
            cellprob_threshold=threshold,flow_threshold=.4,min_size=metadata['limits'][0],device=torch.device('cpu'))
    areas=np.bincount(labels.ravel());keep=(areas>=metadata['limits'][0])&(areas<=metadata['limits'][1]);keep[0]=False
    labels[~keep[labels]]=0
    return relabel_sequential(labels)[0].astype(np.int32)


def fit(rows,modality,directory,boundary_loss=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(19);np.random.seed(19)
    images,masks,areas=training_patches(rows,modality)
    metadata=dict(diameter=float(np.median(2*np.sqrt(areas/np.pi))),
                  limits=[max(5,int(np.quantile(areas,.002))),int(np.quantile(areas,.998))],
                  training_ids=rows.sample_id.tolist(),epochs=100,seed=19,pretrained='cyto3',cellpose_version='3.1.1.3',boundary_loss=boundary_loss)
    model=models.CellposeModel(model_type='cyto3',device=DEVICE,gpu=DEVICE.type!='cpu')
    original_loss=train._loss_fn_seg
    if boundary_loss:train._loss_fn_seg=boundary_dice_loss
    try:
        path,losses,_=train.train_seg(model.net,train_data=images,train_labels=masks,channels=[0,0],normalize={'percentile':[1,99.8]},
                     n_epochs=100,nimg_per_epoch=64,batch_size=8,learning_rate=.1,weight_decay=1e-4,SGD=True,
                     min_train_masks=0,save_path=str(directory),model_name=modality,save_every=100)
    finally:
        train._loss_fn_seg=original_loss
    assert np.isfinite(losses).all()
    metadata['weights']=str(Path(path).relative_to(directory))
    pd.DataFrame({'epoch':np.arange(1,len(losses)+1),'loss':losses}).to_csv(directory/f'{modality}_history.csv',index=False)
    candidates=[-.5,0.,.5];scores=np.zeros(3)
    calibration=rows.iloc[np.linspace(0,len(rows)-1,min(8,len(rows))).astype(int)]
    metadata['calibration_ids']=calibration.sample_id.tolist()
    for row in calibration.itertuples(index=False):
        image=read_pair('Data/training',row.sample_id)[0 if modality=='invivo' else 1]
        flows=predict_flows(image,model,metadata);truth=json.loads(getattr(row,modality+'_instances'))
        for i,threshold in enumerate(candidates):scores[i]+=mask_quality(instances_from_flows(flows,metadata,threshold),truth)[0]/len(calibration)
    metadata['threshold']=candidates[int(scores.argmax())]
    (directory/f'{modality}.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame({'threshold':candidates,'pq':scores}).to_csv(directory/f'{modality}_calibration.csv',index=False)
    print('CELLPOSE CALIBRATION',directory.name,modality,metadata['threshold'],scores,flush=True)
    return model,metadata


def load_model(directory,modality):
    directory=Path(directory);metadata=json.loads((directory/f'{modality}.json').read_text())
    model=models.CellposeModel(pretrained_model=str(directory/metadata['weights']),device=DEVICE,gpu=DEVICE.type!='cpu')
    return model,metadata


def experiment(output='outputs/q5/advanced'):
    out=Path(output);rows=pd.read_csv('Data/training/train_ground_truth.csv');subjects=rows.sample_id.str.split('__').str[0]
    records=[]
    for mouse in sorted(subjects.unique()):
        training,excluded=rows[subjects!=mouse],rows[subjects==mouse]
        for modality in ['invivo','exvivo']:
            model,metadata=fit(training,modality,out/'cellpose'/mouse)
            for row in excluded.itertuples(index=False):
                image=read_pair('Data/training',row.sample_id)[0 if modality=='invivo' else 1]
                labels=instances_from_flows(predict_flows(image,model,metadata),metadata)
                directory=out/'masks/cellpose'/row.sample_id;directory.mkdir(parents=True,exist_ok=True)
                tifffile.imwrite(directory/f'{modality}.tif',labels,compression='zlib')
                records.append(dict(method='cellpose',sample_id=row.sample_id,modality=modality,pq=mask_quality(labels,json.loads(getattr(row,modality+'_instances')))[0]))
                pd.DataFrame(records).to_csv(out/'cellpose_comparison.csv',index=False)
            print('CELLPOSE EXCLUDED',mouse,modality,pd.DataFrame(records).query('modality==@modality').pq.mean(),flush=True)
            del model
    print('CELLPOSE EXPERIMENT COMPLETE',flush=True)


def normalization_experiment(output='outputs/q5/advanced',final=False,modalities=('invivo','exvivo')):
    out=Path(output);rows=pd.read_csv('Data/training/train_ground_truth.csv');subjects=rows.sample_id.str.split('__').str[0]
    records=[]
    for mouse in (['final'] if final else sorted(subjects.unique())):
        training,excluded=(rows,rows.iloc[:0]) if final else (rows[subjects!=mouse],rows[subjects==mouse])
        for modality in modalities:
            model,metadata=load_model(out/'cellpose'/mouse,modality)
            candidates=[dict(tile_norm_blocksize=b,threshold=t) for b in [0,128] for t in [-.5,0.,.5,1.,2.]]
            scores=np.zeros(len(candidates));calibration=training[training.sample_id.isin(metadata['calibration_ids'])]
            for row in calibration.itertuples(index=False):
                image=read_pair('Data/training',row.sample_id)[0 if modality=='invivo' else 1]
                truth=json.loads(getattr(row,modality+'_instances'))
                for block in [0,128]:
                    flows=predict_flows(image,model,metadata|dict(tile_norm_blocksize=block))
                    for i,setting in enumerate(candidates):
                        if setting['tile_norm_blocksize']==block:
                            scores[i]+=mask_quality(instances_from_flows(flows,metadata,setting['threshold']),truth)[0]/len(calibration)
            metadata.update(candidates[int(scores.argmax())])
            directory=out/'cellpose_tile'/mouse;directory.mkdir(parents=True,exist_ok=True)
            metadata['weights']=os.path.relpath(out/'cellpose'/mouse/'models'/modality,directory)
            (directory/f'{modality}.json').write_text(json.dumps(metadata,indent=2))
            pd.DataFrame([dict(**s,pq=q) for s,q in zip(candidates,scores)]).to_csv(directory/f'{modality}_calibration.csv',index=False)
            for row in excluded.itertuples(index=False):
                image=read_pair('Data/training',row.sample_id)[0 if modality=='invivo' else 1]
                labels=instances_from_flows(predict_flows(image,model,metadata),metadata)
                folder=out/'masks/cellpose_tile'/row.sample_id;folder.mkdir(parents=True,exist_ok=True)
                tifffile.imwrite(folder/f'{modality}.tif',labels,compression='zlib')
                records.append(dict(method='cellpose_tile',sample_id=row.sample_id,modality=modality,pq=mask_quality(labels,json.loads(getattr(row,modality+'_instances')))[0]))
                pd.DataFrame(records).to_csv(out/'cellpose_tile_comparison.csv',index=False)
            print('CELLPOSE NORMALIZATION',mouse,modality,metadata['tile_norm_blocksize'],metadata['threshold'],float(scores.max()),flush=True)
            del model


def dice_experiment():
    out=Path('outputs/q5/advanced');rows=pd.read_csv('Data/training/train_ground_truth.csv');subjects=rows.sample_id.str.split('__').str[0]
    records=[]
    for mouse in sorted(subjects.unique()):
        training,excluded=rows[subjects!=mouse],rows[subjects==mouse]
        model,metadata=fit(training,'exvivo',out/'cellpose_dice'/mouse,boundary_loss=True)
        for row in excluded.itertuples(index=False):
            image=read_pair('Data/training',row.sample_id)[1]
            labels=instances_from_flows(predict_flows(image,model,metadata),metadata)
            folder=out/'masks/cellpose_dice'/row.sample_id;folder.mkdir(parents=True,exist_ok=True)
            tifffile.imwrite(folder/'exvivo.tif',labels,compression='zlib')
            records.append(dict(method='cellpose_dice',sample_id=row.sample_id,modality='exvivo',pq=mask_quality(labels,json.loads(row.exvivo_instances))[0]))
            pd.DataFrame(records).to_csv(out/'cellpose_dice_comparison.csv',index=False)
        print('CELLPOSE DICE',mouse,pd.DataFrame(records).pq.mean(),flush=True)
        del model


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['experiment','normalization','dice','fit','fit_tile','fit_dice','predict'],default='experiment')
    parser.add_argument('--modality',choices=['invivo','exvivo'],default='invivo')
    parser.add_argument('--models',default='outputs/q5/advanced/cellpose/final')
    parser.add_argument('--output',default='outputs/q5/advanced/inference')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    if args.stage=='experiment':experiment()
    elif args.stage=='normalization':normalization_experiment()
    elif args.stage=='dice':dice_experiment()
    elif args.stage in ['fit','fit_tile','fit_dice']:
        fit(pd.read_csv('Data/training/train_ground_truth.csv'),args.modality,args.models,boundary_loss=args.stage=='fit_dice')
        if args.stage=='fit_tile':normalization_experiment(final=True,modalities=[args.modality])
    else:
        model,metadata=load_model(args.models,args.modality)
        sample=pd.read_csv('Data/sample_submission.csv')
        for sample_id in sample.sample_id:
            image=read_pair('Data/hidden_test',sample_id)[0 if args.modality=='invivo' else 1]
            labels=instances_from_flows(predict_flows(image,model,metadata),metadata)
            directory=Path(args.output,sample_id);directory.mkdir(parents=True,exist_ok=True)
            tifffile.imwrite(directory/f'{args.modality}.tif',labels,compression='zlib')
            print(sample_id,args.modality,int(labels.max()),flush=True)
