"""Fixed pretrained SAM pilot on one released region per mouse; no tuning by test ID."""
from pathlib import Path
import os,json,time
os.environ['CELLPOSE_LOCAL_MODELS_PATH']=str(Path('outputs/q5/advanced/pretrained').resolve())
os.environ['MPLCONFIGDIR']='/tmp/cv2-mpl'
import pandas as pd,numpy as np,torch,tifffile
from skimage.segmentation import relabel_sequential
from cellpose import models
from cell_matching import read_pair,mask_quality
model=models.CellposeModel(pretrained_model='cpsam_v2',gpu=True,device=torch.device('mps'),use_bfloat16=True)
rows=pd.read_csv('Data/training/train_ground_truth.csv');subjects=rows.sample_id.str.split('__').str[0]
records=[]
for mouse in sorted(subjects.unique()):
 row=next(rows[subjects==mouse].itertuples(index=False))
 metadata=json.load(open(Path('outputs/q5/unet_complete',mouse,'experiment.json')))
 for modality,image in zip(['invivo','exvivo'],read_pair('Data/training',row.sample_id)):
  start=time.time();labels,_,_=model.eval(image,batch_size=4,normalize={'percentile':[1,99.8]},cellprob_threshold=0.,flow_threshold=.4)
  limits=metadata['area_limits'][modality];areas=np.bincount(labels.ravel());keep=(areas>=limits[0])&(areas<=limits[1]);keep[0]=False;labels[~keep[labels]]=0
  labels=relabel_sequential(labels)[0].astype(np.int32)
  directory=Path('outputs/q5/advanced/masks/cpsam_pilot',row.sample_id);directory.mkdir(parents=True,exist_ok=True);tifffile.imwrite(directory/f'{modality}.tif',labels,compression='zlib')
  records.append(dict(sample_id=row.sample_id,modality=modality,pq=mask_quality(labels,json.loads(getattr(row,modality+'_instances')))[0],seconds=time.time()-start,cells=int(labels.max())))
  pd.DataFrame(records).to_csv('outputs/q5/advanced/cpsam_pilot.csv',index=False);print(records[-1],flush=True)
