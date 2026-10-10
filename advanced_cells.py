"""Reproduce improved Q5 predictions using existing U-Net weights; no training."""
from pathlib import Path
import json
import numpy as np,pandas as pd,tifffile
from unet_inference import dense_prediction
from cell_alignment import match_prediction
from cell_matching import read_pair,instances,validate_submission
from unet_cells import load_models,instances_from_probabilities

OUT=Path('outputs/q5/advanced')


def predict_masks(choice):
    sample=pd.read_csv('Data/sample_submission.csv')
    models,metadata=load_models('outputs/q5/unet_complete/final')
    for modality,method in choice['segmentation'].items():
        assert method in ['context128','context128_tta','context256_tta']
        for sample_id in sample.sample_id:
            image=read_pair('Data/hidden_test',sample_id)[0 if modality=='invivo' else 1]
            prediction=dense_prediction(image,models[modality],tile=128 if method.startswith('context128') else 256,augment=method.endswith('_tta'))
            labels=instances_from_probabilities(prediction,metadata['area_limits'][modality],**metadata['postprocessing'][modality])
            directory=OUT/'inference'/sample_id;directory.mkdir(parents=True,exist_ok=True)
            tifffile.imwrite(directory/f'{modality}.tif',labels,compression='zlib')
            print('SEGMENTED',sample_id,modality,int(labels.max()),flush=True)
    (OUT/'inference_config.json').write_text(json.dumps(choice['segmentation']))


def write_submission(output='Kaggle Submissions/unet_improved.csv',reuse_masks=False):
    choice=json.loads((OUT/'selected.json').read_text())
    config=OUT/'inference_config.json'
    ready=reuse_masks and config.exists() and json.loads(config.read_text())==choice['segmentation']
    sample_ids=pd.read_csv('Data/sample_submission.csv').sample_id
    ready=ready and all((OUT/'inference'/sid/f'{m}.tif').exists() for sid in sample_ids for m in ['invivo','exvivo'])
    if not ready:predict_masks(choice)
    sample=pd.read_csv('Data/sample_submission.csv',dtype=str,keep_default_na=False)
    settings={key:choice[key] for key in ['gate','cutoff','deform','structure']}
    rows=[];diagnostics=[]
    for sample_id in sample.sample_id:
        images=read_pair('Data/hidden_test',sample_id)
        labels=[tifffile.imread(OUT/'inference'/sample_id/f'{m}.tif') for m in ['invivo','exvivo']]
        pairs,matrix,statistics=match_prediction(images,labels,**settings)
        row=dict(sample_id=sample_id)
        for modality,prefix,mask in zip(['invivo','exvivo'],['IV','EX'],labels):
            row[modality+'_instances']=json.dumps(instances(mask,prefix),separators=(',',':'))
        row['match_pairs']=json.dumps([[f'IV_{i:06d}',f'EX_{j:06d}'] for i,j in pairs])
        rows.append(row);diagnostics.append(dict(sample_id=sample_id,invivo=int(labels[0].max()),exvivo=int(labels[1].max()),pairs=len(pairs),**statistics))
        print('MATCHED',sample_id,len(pairs),statistics,flush=True)
    frame=pd.DataFrame(rows,columns=sample.columns)
    validate_submission(frame,sample,'Data/hidden_test')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
    assert pd.read_csv(output,dtype=str,keep_default_na=False).equals(frame)
    pd.DataFrame(diagnostics).to_csv(OUT/'submission_diagnostics.csv',index=False)
    return frame


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Kaggle Submissions/unet_improved.csv')
    args=parser.parse_args()
    write_submission(args.output)
