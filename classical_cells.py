"""Classical Q5 mask refinements and saved-configuration inference; no learned models."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.filters import threshold_otsu
from skimage.morphology import h_maxima
from skimage.segmentation import morphological_chan_vese, watershed, relabel_sequential
from skimage.measure import regionprops
from cell_matching import (normalize, segment, clean_instances, REFINED_PARAMETERS,
                           read_pair, instances, validate_submission, centers, register_cells, match_cells)
from cell_alignment import initialize, overlap_statistics, assign, deform_points

OUT = Path('outputs/q5/classical_improved')


def refine_boundaries(image, labels, method):
    """Refine local masks inside nonoverlapping watershed territories."""
    signal = np.maximum(ndi.gaussian_filter(normalize(image), .7)
                         - ndi.gaussian_filter(normalize(image), 10), 0)
    territory = watershed(-signal, labels, mask=ndi.distance_transform_edt(labels == 0) <= 5)
    result = np.zeros_like(labels)
    for i, box in enumerate(ndi.find_objects(territory), 1):
        if box is None: continue
        allowed = territory[box] == i
        initial = labels[box] == i
        values = signal[box]
        if method == 'contour':
            mask = morphological_chan_vese(values, 8, init_level_set=initial,
                                           smoothing=1, lambda1=1, lambda2=1.5)
        else:
            sample = values[allowed]
            cutoff = threshold_otsu(sample) if np.ptp(sample) > 0 else np.inf
            mask = values > cutoff * (.5 if method == 'local' else 1)
        parts, _ = ndi.label(mask & allowed)
        ids, counts = np.unique(parts[initial], return_counts=True)
        counts[ids == 0] = 0
        if counts.max(initial=0):
            mask = ndi.binary_fill_holes(parts == ids[counts.argmax()]) & allowed
            if 25 <= mask.sum() <= 350: result[box][mask] = i
    return relabel_sequential(result)[0].astype(np.int32)


def filter_regions(image, labels):
    """Conservative whole-region fiber rejection; bright elongated cells may remain."""
    signal = normalize(image)
    result = labels.copy()
    for region in regionprops(labels):
        box = tuple(slice(max(0, a-3), min(n, b+3)) for a, b, n in
                    zip(region.bbox[:2], region.bbox[2:], labels.shape))
        mask = labels[box] == region.label
        ring = ndi.binary_dilation(mask, iterations=3) & (labels[box] == 0)
        compactness = 4*np.pi*region.area / max(region.perimeter**2, 1)
        contrast = np.median(signal[box][mask])-np.median(signal[box][ring]) if ring.any() else 1
        if (compactness < .35 and region.solidity < .8) or contrast < .015:
            result[labels == region.label] = 0
    return relabel_sequential(result)[0].astype(np.int32)


def smooth_shapes(labels, scale=1):
    """Fit image-derived mask contours with ellipses, preserving instance ownership."""
    territory=watershed(ndi.distance_transform_edt(labels==0),labels,
                        mask=ndi.distance_transform_edt(labels==0)<=4)
    result=np.zeros_like(labels)
    for i,box in enumerate(ndi.find_objects(territory),1):
        if box is None:continue
        contours,_=cv2.findContours((labels[box]==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea)
        mask=(labels[box]==i).astype(np.uint8)
        if len(contour)>=5:
            center,axes,angle=cv2.fitEllipse(contour)
            if np.isfinite([*center,*axes,angle]).all() and min(axes)>0:
                fitted=np.zeros_like(mask)
                cv2.ellipse(fitted,(center,tuple(v*scale for v in axes),angle),1,-1)
                fitted &= (territory[box]==i).astype(np.uint8)
                if 25<=fitted.sum()<=350:mask=fitted
        result[box][mask>0]=i
    return relabel_sequential(result)[0].astype(np.int32)


def segment_cells(image, modality, method='baseline'):
    settings = REFINED_PARAMETERS[modality].copy()
    labels = segment(image, **settings)
    if method.startswith('ellipse'):
        labels=smooth_shapes(labels,1.1 if method=='ellipse_large' else .9 if method=='ellipse_small' else 1)
    if method in ['local', 'otsu', 'contour']:
        labels = refine_boundaries(image, labels, method)
    elif method in ['tophat', 'prominence']:
        signal = normalize(image)
        if method == 'tophat':
            background = cv2.morphologyEx(signal, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25)))
            corrected = np.clip(signal-background, 0, 1)
            labels = segment(np.round(corrected*65535).astype(np.uint16), **settings)
        else:
            smooth = ndi.gaussian_filter(signal, 1)
            band = np.maximum(smooth-ndi.gaussian_filter(signal, 10), 0)
            response = ndi.gaussian_filter(band, 2.5)-ndi.gaussian_filter(band, 4)
            markers, _ = ndi.label(h_maxima(np.maximum(response, 0), .01))
            candidates = watershed(-band, markers, mask=band>.005)
            heights = ndi.maximum(band, candidates, np.arange(markers.max()+1))
            candidates[band < settings['fraction']*heights[candidates]] = 0
            areas = np.bincount(candidates.ravel()); keep=(areas>=25)&(areas<=350);keep[0]=False
            candidates[~keep[candidates]]=0
            labels = relabel_sequential(clean_instances(candidates))[0].astype(np.int32)
    if method == 'fiber' or (method == 'local_fiber' and modality == 'exvivo'):
        if method == 'local_fiber': labels=refine_boundaries(image, labels, 'local')
        labels = filter_regions(image, labels)
    return labels


def mutual_information(images, matrix):
    """Small bounded affine refinement using MI; matrix maps in vivo to ex vivo."""
    scale = .25
    a, b = [cv2.resize(normalize(im), None, fx=scale, fy=scale) for im in images]
    base = matrix.copy();base[:,2] *= scale
    def transform(p):
        linear = cv2.getRotationMatrix2D((0,0), float(p[0]), 1)[:,:2]
        delta = np.c_[linear@np.array([[1+p[1],p[3]],[0,1+p[2]]]), p[4:6]]
        return (np.vstack([delta,[0,0,1]])@np.vstack([base,[0,0,1]]))[:2]
    def cost(p):
        t=transform(p);shape=(b.shape[1],b.shape[0])
        moving=cv2.warpAffine(a,t,shape)
        valid=cv2.warpAffine(np.ones_like(a),t,shape)>.99
        valid &= (moving>.01)&(b>.01)
        if valid.sum()<100:return 0.
        joint=np.histogram2d(moving[valid],b[valid],bins=24,range=[[0,1],[0,1]])[0]
        joint/=joint.sum();expected=joint.sum(1)[:,None]*joint.sum(0)[None]
        keep=joint>0
        return -float((joint[keep]*np.log(joint[keep]/expected[keep])).sum())
    bounds=[(-3,3),(-.04,.04),(-.04,.04),(-.04,.04),(-3,3),(-3,3)]
    fitted=minimize(cost,np.zeros(6),method='Powell',bounds=bounds,
                    options=dict(maxiter=12,maxfev=250,xtol=.01,ftol=.001))
    result=transform(fitted.x);result[:,2]/=scale
    return result if np.isfinite(result).all() and fitted.fun<cost(np.zeros(6)) else matrix


def match_labels(images, labels, settings, coarse=None):
    if settings.get('matcher')=='legacy':
        matrix,confidence=register_cells(labels)
        stats=dict(confidence=confidence,support=0.,close=0)
        pairs,matrix=match_cells(*labels,matrix,gate=settings['gate'],guard=True,shape_weight=settings.get('shape_weight',2)) if matrix is not None and confidence>=settings['cutoff'] else ([],matrix)
        return pairs,matrix,stats
    matrix, stats = initialize(images, labels, coarse=coarse)
    if matrix is None:return [],matrix,stats
    a,b=[centers(mask) for mask in labels]
    if settings.get('mi',False):
        candidate=mutual_information(images,matrix)
        confidence,support,close=overlap_statistics(a,b,candidate)
        if support>stats['support'] and close>=6:
            matrix=candidate;stats=dict(confidence=confidence,support=support,close=close)
    if stats['confidence']<settings['cutoff']:return [],matrix,stats
    warped=deform_points(a,b,matrix) if settings.get('deform',False) else a@matrix[:,:2].T+matrix[:,2]
    return assign(labels,warped,matrix,settings['gate'],settings['structure']),matrix,stats


def write_submission(output='Kaggle Submissions/classical_improved.csv'):
    choice=json.loads((OUT/'selected.json').read_text())
    sample=pd.read_csv('Data/sample_submission.csv',dtype=str,keep_default_na=False)
    rows=[];diagnostics=[]
    for sample_id in sample.sample_id:
        images=read_pair('Data/hidden_test',sample_id)
        labels=[segment_cells(im,m,choice['segmentation'][m]) for im,m in zip(images,['invivo','exvivo'])]
        pairs,matrix,stats=match_labels(images,labels,choice['matching'])
        row=dict(sample_id=sample_id)
        directory=OUT/'inference'/sample_id;directory.mkdir(parents=True,exist_ok=True)
        for mask,m,prefix in zip(labels,['invivo','exvivo'],['IV','EX']):
            row[m+'_instances']=json.dumps(instances(mask,prefix),separators=(',',':'))
            tifffile.imwrite(directory/f'{m}.tif',mask,compression='zlib')
        row['match_pairs']=json.dumps([[f'IV_{i:06d}',f'EX_{j:06d}'] for i,j in pairs])
        rows.append(row);diagnostics.append(dict(sample_id=sample_id,pairs=len(pairs),**stats))
        print(sample_id,[int(mask.max()) for mask in labels],len(pairs),flush=True)
    frame=pd.DataFrame(rows,columns=sample.columns);validate_submission(frame,sample,'Data/hidden_test')
    Path(output).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
    assert pd.read_csv(output,dtype=str,keep_default_na=False).equals(frame)
    pd.DataFrame(diagnostics).to_csv(OUT/'submission_diagnostics.csv',index=False)
    return frame


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='Kaggle Submissions/classical_improved.csv')
    write_submission(parser.parse_args().output)
