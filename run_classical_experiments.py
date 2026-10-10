"""Small classical mask pilot, complete retained comparison, then mouse-grouped selection."""
from pathlib import Path
import itertools,json
import numpy as np,pandas as pd,tifffile
from classical_cells import OUT,segment_cells,mutual_information
from cell_matching import read_pair,mask_quality,summarize,centers
from cell_alignment import initialize,overlap_statistics,assign,deform_points


def evaluate_masks(rows,methods,filename):
    records=[]
    for row in rows.itertuples(index=False):
        for modality,image in zip(['invivo','exvivo'],read_pair('Data/training',row.sample_id)):
            for method in methods[modality]:
                directory=OUT/'masks'/method/row.sample_id;directory.mkdir(parents=True,exist_ok=True)
                path=directory/f'{modality}.tif'
                mask=tifffile.imread(path) if path.exists() else segment_cells(image,modality,method)
                if not path.exists():tifffile.imwrite(path,mask,compression='zlib')
                pq,mapping=mask_quality(mask,json.loads(getattr(row,modality+'_instances')))
                records.append(dict(sample_id=row.sample_id,modality=modality,method=method,pq=pq,tp=len(mapping),predicted=int(mask.max())))
        pd.DataFrame(records).to_csv(OUT/filename,index=False)
        print('MASKS',row.sample_id,flush=True)
    return pd.DataFrame(records)


def compare():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv('Data/training/train_ground_truth.csv')
    methods=['baseline','local','otsu','contour','tophat','prominence','fiber','local_fiber','ellipse','ellipse_small','ellipse_large']
    pilot_path=OUT/'segmentation_pilot.csv'
    if not pilot_path.exists():
        mice=rows.sample_id.str.split('__').str[0]
        pilot=pd.concat([g.iloc[np.linspace(0,len(g)-1,2,dtype=int)] for _,g in rows.groupby(mice)])
        evaluate_masks(pilot,{m:methods for m in ['invivo','exvivo']},'segmentation_pilot.csv')
    pilot=pd.read_csv(pilot_path)
    scores=pilot.groupby(['method','modality']).pq.mean().unstack()
    retained={m:list(dict.fromkeys(['baseline']+scores[m].nlargest(2).index.tolist())) for m in ['invivo','exvivo']}
    (OUT/'retained.json').write_text(json.dumps(retained,indent=2))
    full=evaluate_masks(rows,retained,'segmentation_comparison.csv')
    means=full.groupby(['method','modality']).pq.mean().unstack()
    means.to_csv(OUT/'segmentation_summary.csv')
    best={m:means[m].idxmax() for m in ['invivo','exvivo']}
    combinations={'baseline':dict(invivo='baseline',exvivo='baseline'),'best':best,'grouped':best}
    # Image-only hypotheses are unchanged across mask variants; reuse the completed image search.
    coarse=json.loads(Path('outputs/q5/advanced/image_alignment.json').read_text()) if Path('outputs/q5/advanced/image_alignment.json').exists() else {}
    cache=OUT/'image_alignment.json'
    if cache.exists():coarse.update(json.loads(cache.read_text()))
    results=[];geometry_cache={}
    for name,methods in combinations.items():
        for row in rows.itertuples(index=False):
            chosen=methods
            if name=='grouped':
                other=full[full.sample_id.str.split('__').str[0]!=row.sample_id.split('__')[0]]
                table=other.groupby(['method','modality']).pq.mean().unstack()
                chosen={m:table[m].idxmax() for m in ['invivo','exvivo']}
            identity=(row.sample_id,chosen['invivo'],chosen['exvivo'])
            if identity in geometry_cache:
                results.extend([dict(r,method=name) for r in geometry_cache[identity]])
                continue
            images=read_pair('Data/training',row.sample_id)
            labels=[tifffile.imread(OUT/'masks'/chosen[m]/row.sample_id/f'{m}.tif') for m in ['invivo','exvivo']]
            if row.sample_id not in coarse:
                from cell_matching import register
                t,response=register(*images);coarse[row.sample_id]=[None if t is None else t.tolist(),float(response)]
            t,response=coarse[row.sample_id];initial=(np.array(t) if t is not None else None,response)
            matrix,stats=initialize(images,labels,coarse=initial)
            a,b=[centers(mask) for mask in labels];geometry=(a,b,[np.bincount(mask.ravel())[1:] for mask in labels])
            hypotheses={False:(matrix,stats)}
            if matrix is not None:
                fitted=mutual_information(images,matrix)
                confidence,support,close=overlap_statistics(a,b,fitted)
                hypotheses[True]=(fitted,dict(confidence=confidence,support=support,close=close)) if support>stats['support'] and close>=6 else (matrix,stats)
            else:hypotheses[True]=(matrix,stats)
            pqs,maps=zip(*(mask_quality(mask,json.loads(getattr(row,m+'_instances'))) for mask,m in zip(labels,['invivo','exvivo'])))
            verified={tuple(pair) for pair in json.loads(row.match_pairs)}
            recoverable=sum(i in maps[0].values() and j in maps[1].values() for i,j in verified)
            local=[]
            for mi,(matrix,stats) in hypotheses.items():
                warped=a@matrix[:,:2].T+matrix[:,2] if matrix is not None else None
                deformed=deform_points(a,b,matrix) if matrix is not None else None
                for deform,structure,gate,cutoff in itertools.product([False,True],[0,1],[4,6,8],[.15,.25,.35]):
                    pairs=assign(labels,deformed if deform else warped,matrix,gate,structure,geometry) if matrix is not None and stats['confidence']>=cutoff else []
                    local.append(dict(method=name,sample_id=row.sample_id,source_invivo=chosen['invivo'],source_exvivo=chosen['exvivo'],mi=mi,deform=deform,structure=structure,gate=gate,cutoff=cutoff,**stats,pq_invivo=pqs[0],pq_exvivo=pqs[1],correct_pairs=sum((maps[0].get(i),maps[1].get(j)) in verified for i,j in pairs),recoverable_pairs=recoverable,predicted_pairs=len(pairs),verified_pairs=len(verified)))
            geometry_cache[identity]=local;results.extend(local)
            pd.DataFrame(results).to_csv(OUT/'matching_comparison.csv',index=False)
            cache.write_text(json.dumps(coarse))
            print('MATCHING',name,row.sample_id,stats,flush=True)
    frame=pd.DataFrame(results);frame['matcher']='improved';frame['shape_weight']=1
    frame.to_csv(OUT/'matching_comparison.csv',index=False)
    return select_settings(frame,combinations)


def select_settings(frame,combinations):
    keys=['method','matcher','mi','deform','structure','shape_weight','gate','cutoff']
    summary=pd.DataFrame([dict(zip(keys,k),**summarize(g)) for k,g in frame.groupby(keys)]).sort_values(['score','mi','deform','structure'],ascending=[False,True,True,True])
    summary.to_csv(OUT/'matching_summary.csv',index=False)
    selected=summary.iloc[0].to_dict()
    keep=np.logical_and.reduce([frame[k].to_numpy()==selected[k] for k in keys])
    selected_regions=frame[keep];selected_regions.to_csv(OUT/'selected_per_region.csv',index=False)
    settings={k:bool(selected[k]) if k in ['mi','deform'] else int(selected[k]) if k in ['structure','shape_weight','gate'] else str(selected[k]) if k=='matcher' else float(selected[k]) for k in ['matcher','mi','deform','structure','shape_weight','gate','cutoff']}
    choice=dict(segmentation=combinations[selected['method']],matching=settings,selected_metrics=summarize(selected_regions))
    (OUT/'selected.json').write_text(json.dumps(choice,indent=2))
    mice=frame.sample_id.str.split('__').str[0];grouped=[]
    for mouse in sorted(mice.unique()):
        training=frame[mice!=mouse]
        ranking=pd.DataFrame([dict(zip(keys,k),**summarize(g)) for k,g in training.groupby(keys)]).sort_values(['score','mi','deform','structure'],ascending=[False,True,True,True])
        setting=ranking.iloc[0]
        keep=(mice==mouse)&np.logical_and.reduce([frame[k].to_numpy()==setting[k] for k in keys])
        grouped.append(frame[keep])
    grouped=pd.concat(grouped,ignore_index=True);grouped.to_csv(OUT/'grouped_per_region.csv',index=False)
    (OUT/'grouped_summary.json').write_text(json.dumps(summarize(grouped),indent=2))
    folds=pd.DataFrame([dict(mouse=m,**summarize(g)) for m,g in grouped.groupby(grouped.sample_id.str.split('__').str[0])]);folds.to_csv(OUT/'grouped_per_mouse.csv',index=False)
    print('SELECTED',choice,flush=True);print('GROUPED',summarize(grouped),flush=True)
    return choice


def compare_legacy():
    """Retain the old matcher as an explicit comparator, using the same mask candidates."""
    from cell_matching import register_cells,match_cells
    frame=pd.read_csv(OUT/'matching_comparison.csv')
    if 'matcher' not in frame:frame=frame.assign(matcher='improved',shape_weight=1)
    frame=frame[frame.matcher=='improved'].copy()
    rows=pd.read_csv('Data/training/train_ground_truth.csv').set_index('sample_id')
    records=[]
    for (sid,iv,ex),group in frame.groupby(['sample_id','source_invivo','source_exvivo']):
        labels=[tifffile.imread(OUT/'masks'/method/sid/f'{m}.tif') for method,m in zip([iv,ex],['invivo','exvivo'])]
        matrix,confidence=register_cells(labels)
        row=rows.loc[sid]
        pqs,maps=zip(*(mask_quality(mask,json.loads(row[m+'_instances'])) for mask,m in zip(labels,['invivo','exvivo'])))
        verified={tuple(p) for p in json.loads(row.match_pairs)}
        recoverable=sum(i in maps[0].values() and j in maps[1].values() for i,j in verified)
        for shape_weight,gate in itertools.product([0,2],[6,8]):
            pairs,_=match_cells(*labels,matrix,gate=gate,guard=True,shape_weight=shape_weight)
            correct=sum((maps[0].get(i),maps[1].get(j)) in verified for i,j in pairs)
            for method,cutoff in itertools.product(group.method.unique(),[.12,.16,.20,.24]):
                accepted=matrix is not None and confidence>=cutoff
                records.append(dict(method=method,matcher='legacy',sample_id=sid,source_invivo=iv,source_exvivo=ex,mi=False,deform=False,structure=0,shape_weight=shape_weight,gate=gate,cutoff=cutoff,confidence=confidence,support=0.,close=0,pq_invivo=pqs[0],pq_exvivo=pqs[1],correct_pairs=correct if accepted else 0,recoverable_pairs=recoverable,predicted_pairs=len(pairs) if accepted else 0,verified_pairs=len(verified)))
        print('LEGACY',sid,iv,ex,flush=True)
    frame=pd.concat([frame,pd.DataFrame(records)],ignore_index=True)
    frame.to_csv(OUT/'matching_comparison.csv',index=False)
    segmentation=pd.read_csv(OUT/'segmentation_comparison.csv')
    means=segmentation.groupby(['method','modality']).pq.mean().unstack()
    best={m:means[m].idxmax() for m in ['invivo','exvivo']}
    return select_settings(frame,{'baseline':dict(invivo='baseline',exvivo='baseline'),'best':best,'grouped':best})


if __name__=='__main__':
    compare()
    compare_legacy()
