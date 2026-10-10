"""Partial-overlap affine/CPD registration and structural one-to-one cell matching."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from cell_matching import centers,register_cells,register,match_cells,alignment_support


def overlap_statistics(source,target,matrix):
    warped=source@matrix[:,:2].T+matrix[:,2]
    distance,nearest=cKDTree(target).query(warped)
    _,reverse=cKDTree(warped).query(target)
    mutual=reverse[nearest]==np.arange(len(source))
    # Normalize by target cells inside the warped source field, rather than all ex-vivo cells.
    hull=cv2.convexHull(warped.astype(np.float32))
    inside=np.array([cv2.pointPolygonTest(hull,tuple(map(float,p)),True)>=-8 for p in target])
    denominator=max(1,min(len(source),int(inside.sum())))
    support=float(np.exp(-.5*(distance[mutual]/3)**2).sum())
    return support/denominator,support,int(((distance<4)&mutual).sum())


def initialize(images,labels,coarse=None):
    a,b=[centers(mask) for mask in labels]
    if min(len(a),len(b))<6:return None,dict(confidence=0.,support=0.,close=0)
    hypotheses=[]
    matrix,_=register_cells(labels)
    if matrix is not None:hypotheses.append(matrix)
    matrix,_=register(*images) if coarse is None else coarse
    if matrix is not None:hypotheses.append(matrix)
    best=None;quality=-np.inf;statistics=None
    for matrix in hypotheses:
        _,fitted=match_cells(*labels,matrix,guard=True)
        confidence,support,close=overlap_statistics(a,b,fitted)
        # Require actual support and favor repeatable inliers over sparse accidental overlap.
        rank=support*np.sqrt(max(confidence,0))
        if close>=6 and rank>quality:
            best,quality,statistics=fitted,rank,dict(confidence=confidence,support=support,close=close)
    return best,statistics or dict(confidence=0.,support=0.,close=0)


def deform_points(source,target,matrix):
    from pycpd import DeformableRegistration
    warped=source@matrix[:,:2].T+matrix[:,2]
    distance,nearest=cKDTree(target).query(warped)
    _,reverse=cKDTree(warped).query(target)
    keep=(distance<8)&(reverse[nearest]==np.arange(len(source)))
    if keep.sum()<6:return warped
    # Mutual local anchors avoid fitting the unmatched parts of the large ex-vivo field.
    registration=DeformableRegistration(X=target[nearest[keep]],Y=warped[keep],
                    alpha=2,beta=25,w=.1,max_iterations=40,tolerance=.001)
    registration.register()
    kernel=np.exp(-np.sum((warped[:,None]-warped[None,keep])**2,axis=2)/(2*registration.beta**2))
    result=warped+kernel@registration.W
    if not np.isfinite(result).all() or np.quantile(np.linalg.norm(result-warped,axis=1),.95)>12:return warped
    old=alignment_support(warped,target,np.eye(2,3))
    new=alignment_support(result,target,np.eye(2,3))
    return result if new>old else warped


def assign(labels,warped,matrix,gate=6,structure=0,geometry=None):
    a,b,areas=geometry if geometry is not None else (*[centers(mask) for mask in labels],[np.bincount(mask.ravel())[1:] for mask in labels])
    if min(len(a),len(b))==0:return []
    distance=np.linalg.norm(warped[:,None]-b[None],axis=2)
    ratio=areas[0][:,None]*abs(np.linalg.det(matrix[:,:2]))/np.maximum(areas[1],1)
    cost=distance+abs(np.log(np.maximum(ratio,1e-6)))
    if structure:
        d,j=cKDTree(b).query(warped);_,reverse=cKDTree(warped).query(b)
        anchor=np.flatnonzero((d<3)&(reverse[j]==np.arange(len(a))))
        if len(anchor)>=3:
            # Distances to the same local anchor correspondences provide a structural cost.
            _,neighbors=cKDTree(warped[anchor]).query(warped,k=3)
            p=warped[anchor[neighbors]];q=b[j[anchor[neighbors]]]
            source_dist=np.linalg.norm(warped[:,None]-p,axis=2)
            target_dist=np.linalg.norm(b[None,:,None]-q[:,None],axis=3)
            discrepancy=np.median(abs(source_dist[:,None]-target_dist),axis=2)
            cost+=structure*np.minimum(discrepancy,gate)/3
    costs=np.c_[np.where(distance<gate,cost,1e6),np.full((len(a),len(a)),gate)]
    i,j=linear_sum_assignment(costs)
    return [(int(x+1),int(y+1)) for x,y in zip(i,j) if y<len(b) and cost[x,y]<gate]


def match_prediction(images,labels,gate=6,cutoff=.2,deform=False,structure=0,initial=None):
    matrix,statistics=initialize(images,labels) if initial is None else initial
    if matrix is None or statistics['confidence']<cutoff:return [],matrix,statistics
    a,b=[centers(mask) for mask in labels]
    warped=deform_points(a,b,matrix) if deform else a@matrix[:,:2].T+matrix[:,2]
    return assign(labels,warped,matrix,gate,structure),matrix,statistics


def compare_methods(root='outputs/q5/advanced'):
    from pathlib import Path
    import json,itertools
    import pandas as pd,tifffile
    from cell_matching import read_pair,mask_quality,summarize
    root=Path(root);rows=pd.read_csv('Data/training/train_ground_truth.csv')
    segmentation=pd.read_csv(root/'unet_inference_comparison.csv')
    assert all(n==47 for n in segmentation.groupby(['method','modality']).size())
    table=segmentation.groupby(['method','modality']).pq.mean().unstack()
    table.to_csv(root/'segmentation_summary.csv')
    best={modality:table[modality].idxmax() for modality in ['invivo','exvivo']}
    combinations={name:dict(invivo=name,exvivo=name) for name in table.index}
    combinations['hybrid']=best
    results=pd.read_csv(root/'matching_comparison.csv').to_dict('records') if (root/'matching_comparison.csv').exists() else []
    completed={(r['method'],r['sample_id']) for r in results}
    cache_path=root/'image_alignment.json'
    coarse=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for name,methods in combinations.items():
        for row in rows.itertuples(index=False):
            if (name,row.sample_id) in completed:continue
            chosen=methods
            if name=='hybrid':
                mouse=row.sample_id.split('__')[0]
                training=segmentation[segmentation.sample_id.str.split('__').str[0]!=mouse]
                means=training.groupby(['method','modality']).pq.mean().unstack()
                chosen={m:means[m].idxmax() for m in ['invivo','exvivo']}
            if name=='hybrid':
                reused=[r for r in results if r['method']!='hybrid' and r['sample_id']==row.sample_id and all(r['source_'+m]==chosen[m] for m in ['invivo','exvivo'])]
                if reused:
                    results.extend([dict(r,method='hybrid') for r in reused])
                    pd.DataFrame(results).to_csv(root/'matching_comparison.csv',index=False)
                    continue
            labels=[tifffile.imread(root/'masks'/chosen[m]/row.sample_id/f'{m}.tif') for m in ['invivo','exvivo']]
            images=read_pair('Data/training',row.sample_id)
            if row.sample_id not in coarse:
                image_matrix,response=register(*images)
                coarse[row.sample_id]=[image_matrix.tolist() if image_matrix is not None else None,float(response)]
                cache_path.write_text(json.dumps(coarse))
            image_matrix,response=coarse[row.sample_id]
            image_matrix=np.array(image_matrix) if image_matrix is not None else None
            matrix,statistics=initialize(images,labels,coarse=(image_matrix,response))
            a,b=[centers(mask) for mask in labels]
            warped=a@matrix[:,:2].T+matrix[:,2] if matrix is not None else None
            deformed=deform_points(a,b,matrix) if matrix is not None else None
            pqs,maps=zip(*(mask_quality(mask,json.loads(getattr(row,m+'_instances'))) for mask,m in zip(labels,['invivo','exvivo'])))
            verified={tuple(pair) for pair in json.loads(row.match_pairs)}
            recoverable=sum(i in maps[0].values() and j in maps[1].values() for i,j in verified)
            geometry=(a,b,[np.bincount(mask.ravel())[1:] for mask in labels])
            for deform,structure,gate,cutoff in itertools.product([False,True],[0,1],[4,6,8],[.15,.25]):
                pairs=assign(labels,deformed if deform else warped,matrix,gate,structure,geometry) if matrix is not None and statistics['confidence']>=cutoff else []
                results.append(dict(method=name,sample_id=row.sample_id,source_invivo=chosen['invivo'],source_exvivo=chosen['exvivo'],deform=deform,structure=structure,gate=gate,cutoff=cutoff,
                                    **statistics,pq_invivo=pqs[0],pq_exvivo=pqs[1],correct_pairs=sum((maps[0].get(i),maps[1].get(j)) in verified for i,j in pairs),recoverable_pairs=recoverable,predicted_pairs=len(pairs),verified_pairs=len(verified)))
            pd.DataFrame(results).to_csv(root/'matching_comparison.csv',index=False)
            print('MATCHING',name,row.sample_id,statistics,flush=True)
    frame=pd.DataFrame(results);columns=['method','deform','structure','gate','cutoff']
    summary=pd.DataFrame([dict(zip(columns,key),**summarize(group)) for key,group in frame.groupby(columns)])
    summary=summary.sort_values('score',ascending=False)
    summary.to_csv(root/'matching_summary.csv',index=False)
    choice=summary.iloc[0].to_dict();choice['deform']=bool(choice['deform']);choice['structure']=int(choice['structure']);choice['gate']=int(choice['gate'])
    choice['segmentation']=combinations[choice['method']]
    (root/'selected.json').write_text(json.dumps(choice,indent=2))
    mask=np.logical_and.reduce([frame[key].to_numpy()==choice[key] for key in columns])
    selected=frame[mask].copy();selected.to_csv(root/'selected_per_region.csv',index=False)
    folds=pd.DataFrame([dict(mouse=mouse,**summarize(group)) for mouse,group in selected.groupby(selected.sample_id.str.split('__').str[0])])
    folds.to_csv(root/'selected_per_mouse.csv',index=False)
    # Retrospective selection on other mice; earlier design/OOF model choices preclude an untouched or fully nested claim.
    subjects=frame.sample_id.str.split('__').str[0];grouped=[]
    for mouse in sorted(subjects.unique()):
        training=frame[subjects!=mouse]
        choices=pd.DataFrame([dict(zip(columns,key),**summarize(g)) for key,g in training.groupby(columns)]).sort_values('score',ascending=False)
        setting=choices.iloc[0].to_dict()
        keep=(subjects==mouse)&np.logical_and.reduce([frame[key].to_numpy()==setting[key] for key in columns])
        grouped.append(frame[keep])
    grouped=pd.concat(grouped,ignore_index=True);grouped.to_csv(root/'grouped_per_region.csv',index=False)
    (root/'grouped_summary.json').write_text(json.dumps(summarize(grouped),indent=2))
    print('SELECTED',choice,flush=True)
    print('GROUPED CONFIGURATION SELECTION',summarize(grouped),flush=True)
    return choice


if __name__=='__main__':
    compare_methods()
