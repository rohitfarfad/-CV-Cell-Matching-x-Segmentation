"""Fast contrast, scale, and reflection checks for the unchanged small U-Net."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from unet_cells import load_models, instances_from_probabilities
from unet_inference import dense_prediction
from cell_matching import normalize, read_pair, mask_quality

ROOT = Path('outputs/q5/unet_refined')


def input_prediction(image, model, method, baseline=None):
    if method.startswith('gamma'):
        gamma = float(method.split('_')[1])
        changed = np.round(normalize(image)**gamma*65535).astype(np.uint16)
        return dense_prediction(changed, model, tile=128, augment=True)
    if method.startswith('scale'):
        scale = float(method.split('_')[1])
        changed = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        probability = dense_prediction(changed, model, tile=128, augment=True)
        return np.stack([cv2.resize(p, image.shape[::-1], interpolation=cv2.INTER_LINEAR) for p in probability])
    baseline = dense_prediction(image, model, tile=128, augment=True) if baseline is None else baseline
    if method == 'reflection':
        flipped = dense_prediction(image[:, ::-1].copy(), model, tile=128, augment=True)[:, :, ::-1]
        return (baseline+flipped)*.5
    if method == 'multiscale':
        return (baseline+input_prediction(image, model, 'scale_0.85')+input_prediction(image, model, 'scale_1.15'))/3
    assert method == 'baseline'
    return baseline


def pilot():
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    pilot_ids = pd.read_csv(ROOT / 'pilot.csv').sample_id.unique()
    records = []
    methods = ['baseline', 'gamma_0.8', 'gamma_1.2', 'scale_0.85', 'scale_1.15', 'reflection', 'multiscale']
    for mouse, group in rows[rows.sample_id.isin(pilot_ids)].groupby(rows[rows.sample_id.isin(pilot_ids)].sample_id.str.split('__').str[0]):
        models, metadata = load_models(Path('outputs/q5/unet_complete') / mouse)
        for row in group.itertuples(index=False):
            for m, image in zip(['invivo', 'exvivo'], read_pair('Data/training', row.sample_id)):
                with np.load(ROOT / 'pilot_probabilities' / row.sample_id / f'{m}.npz') as saved:
                    baseline = saved['probability']
                predictions = {}
                for method in methods:
                    probability = ((baseline+predictions['scale_0.85']+predictions['scale_1.15'])/3 if method == 'multiscale'
                                   else input_prediction(image, models[m], method, baseline=baseline))
                    if method.startswith('scale'):
                        predictions[method] = probability
                    labels = instances_from_probabilities(probability, metadata['area_limits'][m], **metadata['postprocessing'][m])
                    records.append(dict(sample_id=row.sample_id, modality=m, method=method,
                                        pq=mask_quality(labels, json.loads(getattr(row, m+'_instances')))[0]))
                print('INPUT PILOT', row.sample_id, m, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'input_pilot.csv', index=False)
    means = frame.groupby(['method', 'modality']).pq.mean().unstack()
    means.to_csv(ROOT / 'input_pilot_summary.csv')
    choice = {m: means[m].idxmax() for m in ['invivo', 'exvivo']}
    (ROOT / 'input_selected.json').write_text(json.dumps(choice, indent=2))
    print(means.to_string(), flush=True)
    print('INPUT SELECTED', choice, flush=True)


def benchmark():
    import tifffile
    from cell_alignment import initialize
    from unet_quick import quick_pairs
    from cell_matching import evaluate, summarize
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    choice = json.loads((ROOT / 'input_selected.json').read_text())
    coarse = json.loads(Path('outputs/q5/advanced/image_alignment.json').read_text())
    results = []
    for mouse, group in rows.groupby(rows.sample_id.str.split('__').str[0]):
        models, metadata = load_models(Path('outputs/q5/unet_complete') / mouse)
        for row in group.itertuples(index=False):
            assert row.sample_id not in metadata['training_ids']
            for m, image in zip(['invivo', 'exvivo'], read_pair('Data/training', row.sample_id)):
                for method in ['baseline', choice[m]]:
                    path = ROOT / 'inputs' / method / row.sample_id / f'{m}.tif'
                    if path.exists():
                        labels = tifffile.imread(path)
                    else:
                        labels = (tifffile.imread(Path('outputs/q5/advanced/masks/context128_tta') / row.sample_id / f'{m}.tif')
                                  if method == 'baseline' else instances_from_probabilities(input_prediction(image, models[m], method),
                                      metadata['area_limits'][m], **metadata['postprocessing'][m]))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(path, labels, compression='zlib')
                    results.append(dict(sample_id=row.sample_id, modality=m, method=method,
                                        pq=mask_quality(labels, json.loads(getattr(row, m+'_instances')))[0]))
            print('INPUT VALIDATED', row.sample_id, flush=True)
    segmentation = pd.DataFrame(results)
    segmentation.to_csv(ROOT / 'input_segmentation.csv', index=False)
    selected = {m: segmentation[segmentation.modality == m].groupby('method').pq.mean().idxmax() for m in ['invivo', 'exvivo']}
    records, cache = [], {}
    for mode in ['selected', 'grouped']:
        for row in rows.itertuples(index=False):
            methods = selected
            if mode == 'grouped':
                train = segmentation[segmentation.sample_id.str.split('__').str[0] != row.sample_id.split('__')[0]]
                methods = {m: train[train.modality == m].groupby('method').pq.mean().idxmax() for m in ['invivo', 'exvivo']}
            key = (row.sample_id, methods['invivo'], methods['exvivo'])
            if key not in cache:
                labels = [tifffile.imread(ROOT / 'inputs' / methods[m] / row.sample_id / f'{m}.tif') for m in ['invivo', 'exvivo']]
                matrix, response = coarse[row.sample_id]
                matrix, stats = initialize(read_pair('Data/training', row.sample_id), labels,
                    coarse=(np.asarray(matrix) if matrix is not None else None, response))
                pairs = quick_pairs(labels, matrix, stats, gate=8, cutoff=.28, mutual=True)
                cache[key] = evaluate(labels, pairs, row)
            records.append(dict(mode=mode, sample_id=row.sample_id, source_invivo=methods['invivo'],
                                source_exvivo=methods['exvivo'], **cache[key]))
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'input_matching.csv', index=False)
    summaries = {mode: summarize(group) for mode, group in frame.groupby('mode')}
    (ROOT / 'input_summary.json').write_text(json.dumps(summaries, indent=2))
    (ROOT / 'input_deployment.json').write_text(json.dumps(selected, indent=2))
    print('INPUT SUMMARY', summaries, selected, flush=True)


def write_submission(output='Kaggle Submissions/unet_refined.csv'):
    import tifffile
    from cell_alignment import initialize
    from cell_local_matching import corrected_pairs, fast_instances, check_instances
    choice = json.loads((ROOT / 'deployment.json').read_text())
    sample = pd.read_csv('Data/sample_submission.csv', dtype=str, keep_default_na=False)
    records, diagnostics = [], []
    models = metadata = None
    for sid in sample.sample_id:
        images = read_pair('Data/hidden_test', sid)
        labels = []
        for m, image in zip(['invivo', 'exvivo'], images):
            method = choice['segmentation'][m]
            path = (Path('outputs/q5/advanced/inference') if method == 'baseline' else ROOT / 'input_inference' / method) / sid / f'{m}.tif'
            if path.exists():
                mask = tifffile.imread(path)
                if method == 'baseline':
                    assert json.loads(Path('outputs/q5/advanced/inference_config.json').read_text()) == dict(invivo='context128_tta',exvivo='context128_tta')
            else:
                if models is None:
                    models, metadata = load_models('outputs/q5/unet_complete/final')
                probability = input_prediction(image, models[m], method)
                mask = instances_from_probabilities(probability, metadata['area_limits'][m], **metadata['postprocessing'][m])
                # New inference results are isolated from earlier prediction artifacts.
                path = ROOT / 'input_inference' / method / sid / f'{m}.tif'
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, mask, compression='zlib')
            assert mask.shape == image.shape
            labels.append(mask)
        matrix, stats = initialize(images, labels)
        pairs = corrected_pairs(labels, matrix, stats, **choice['matching'])
        dictionaries = [fast_instances(mask, prefix) for mask, prefix in zip(labels, ['IV', 'EX'])]
        for dictionary, mask in zip(dictionaries, labels):
            check_instances(dictionary, mask)
        named_pairs = [[f'IV_{i:06d}', f'EX_{j:06d}'] for i, j in pairs]
        assert all(i in dictionaries[0] and j in dictionaries[1] for i, j in named_pairs)
        assert len({i for i,j in named_pairs}) == len(named_pairs) == len({j for i,j in named_pairs})
        records.append(dict(sample_id=sid, invivo_instances=json.dumps(dictionaries[0],separators=(',',':')),
                            exvivo_instances=json.dumps(dictionaries[1],separators=(',',':')),match_pairs=json.dumps(named_pairs)))
        diagnostics.append(dict(sample_id=sid,invivo=int(labels[0].max()),exvivo=int(labels[1].max()),pairs=len(pairs),**stats))
        print('CANDIDATE', sid, len(pairs), flush=True)
    frame = pd.DataFrame(records, columns=sample.columns)
    assert frame.sample_id.tolist() == sample.sample_id.tolist() and frame.sample_id.is_unique
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output,index=False)
    assert pd.read_csv(output,dtype=str,keep_default_na=False).equals(frame)
    pd.DataFrame(diagnostics).to_csv(ROOT / 'submission_diagnostics.csv',index=False)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    write_submission(args.output) if args.output else benchmark() if args.benchmark else pilot()
