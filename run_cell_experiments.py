"""Evaluate saved U-Net inference variants by excluded mouse; run on the GPU."""
from pathlib import Path
import json
import pandas as pd,tifffile,numpy as np
from unet_inference import dense_prediction
from unet_cells import load_models,instances_from_probabilities
from cell_matching import read_pair,mask_quality
out=Path('outputs/q5/advanced');rows=pd.read_csv('Data/training/train_ground_truth.csv');subjects=rows.sample_id.str.split('__').str[0]
records=[]
for mouse in sorted(subjects.unique()):
 train,excluded=rows[subjects!=mouse],rows[subjects==mouse]
 old_models,oldmeta=load_models(Path('outputs/q5/unet_complete')/mouse)
 for row in excluded.itertuples(index=False):
  for modality,image in zip(['invivo','exvivo'],read_pair('Data/training',row.sample_id)):
   for name,tile,tta in [('context128',128,False),('context128_tta',128,True),('context256_tta',256,True)]:
    directory=out/'masks'/name/row.sample_id;directory.mkdir(parents=True,exist_ok=True)
    existing=directory/f'{modality}.tif'
    if existing.exists():mask=tifffile.imread(existing)
    else:
     prob=dense_prediction(image,old_models[modality],tile=tile,augment=tta)
     mask=instances_from_probabilities(prob,oldmeta['area_limits'][modality],**oldmeta['postprocessing'][modality])
    tifffile.imwrite(directory/f'{modality}.tif',mask,compression='zlib')
    records.append(dict(method=name,sample_id=row.sample_id,modality=modality,pq=mask_quality(mask,json.loads(getattr(row,modality+'_instances')))[0]))
    pd.DataFrame(records).to_csv(out/'unet_inference_comparison.csv',index=False)
 print('CONTEXT COMPLETE',mouse,flush=True)
 del old_models
