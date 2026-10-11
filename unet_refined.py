"""Recalibrate saved U-Net boundaries and instance splitting without retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from unet_cells import load_models, instances_from_probabilities
from unet_inference import dense_prediction
from unet_quick import quick_pairs
from cell_alignment import initialize
from cell_matching import read_pair, mask_quality, summarize, evaluate, instances, validate_submission

OUT = Path('outputs/q5/unet_refined')
OLD = Path('outputs/q5/advanced')
MODALITIES = ['invivo', 'exvivo']


def candidates():
    return {f'{method}_{threshold:.2f}': dict(method=method, threshold=threshold)
            for method in ['threshold', 'smooth', 'peaks', 'probability']
            for threshold in [.35, .45, .55, .65]}


def refined_instances(probability, limits, method='threshold', threshold=.5, core_threshold=.5):
    signal = ndi.gaussian_filter(probability[0], .5) if method == 'smooth' else probability[0]
    foreground = ndi.binary_fill_holes(signal > threshold)
    if method == 'peaks':
        points = peak_local_max(ndi.gaussian_filter(probability[1], .7), min_distance=4,
                               threshold_abs=core_threshold, labels=foreground.astype(np.uint8), exclude_border=False)
        markers = np.zeros(foreground.shape, np.int32)
        markers[tuple(points.T)] = np.arange(1, len(points)+1)
    else:
        markers, _ = ndi.label((probability[1] > core_threshold) & foreground)
    surface = -signal if method == 'probability' else -ndi.distance_transform_edt(foreground)
    labels = watershed(surface, markers, mask=foreground)
    areas = np.bincount(labels.ravel())
    keep = (areas >= limits[0]) & (areas <= limits[1])
    keep[0] = False
    labels[~keep[labels]] = 0
    return relabel_sequential(labels)[0].astype(np.int32)


def predict(image, model, sid, modality, pilot=False):
    path = OUT / 'pilot_probabilities' / sid / f'{modality}.npz'
    if path.exists():
        with np.load(path) as saved:
            return saved['probability']
    probability = dense_prediction(image, model, tile=128, augment=True)
    if pilot:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, probability=probability)
    return probability


def pilot():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    results = []
    for mouse, group in rows.groupby(rows.sample_id.str.split('__').str[0]):
        models, metadata = load_models(Path('outputs/q5/unet_complete') / mouse)
        sizes = [(sum(im.size for im in read_pair('Data/training', r.sample_id)), r) for r in group.itertuples(index=False)]
        for _, row in sorted(sizes, key=lambda item: item[0])[:2]:
            assert row.sample_id not in metadata['training_ids']
            for modality, image in zip(MODALITIES, read_pair('Data/training', row.sample_id)):
                probability = predict(image, models[modality], row.sample_id, modality, pilot=True)
                truth = json.loads(getattr(row, modality+'_instances'))
                settings = {'baseline': None, **candidates()}
                for name, setting in settings.items():
                    mask = (tifffile.imread(OLD / 'masks/context128_tta' / row.sample_id / f'{modality}.tif')
                            if setting is None else refined_instances(probability, metadata['area_limits'][modality], **setting))
                    results.append(dict(sample_id=row.sample_id, modality=modality, method=name,
                                        pq=mask_quality(mask, truth)[0]))
                print('PILOT', row.sample_id, modality, flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / 'pilot.csv', index=False)
    means = frame.groupby(['method', 'modality']).pq.mean().unstack()
    means.to_csv(OUT / 'pilot_summary.csv')
    retained = {m: list(dict.fromkeys(['baseline'] + means[m].nlargest(2).index.tolist())) for m in MODALITIES}
    (OUT / 'retained.json').write_text(json.dumps(retained, indent=2))
    print('RETAINED', retained, flush=True)
    print(means.sort_values('exvivo', ascending=False).head(8).to_string(), flush=True)


def benchmark():
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    retained = json.loads((OUT / 'retained.json').read_text())
    results = []
    for mouse, group in rows.groupby(rows.sample_id.str.split('__').str[0]):
        models, metadata = load_models(Path('outputs/q5/unet_complete') / mouse)
        for row in group.itertuples(index=False):
            assert row.sample_id not in metadata['training_ids']
            for m, image in zip(MODALITIES, read_pair('Data/training', row.sample_id)):
                existing = {name: OUT / 'masks' / name / row.sample_id / f'{m}.tif' for name in retained[m]}
                probability = None if all(path.exists() for path in existing.values()) else predict(image, models[m], row.sample_id, m)
                for name in retained[m]:
                    mask = (tifffile.imread(existing[name]) if existing[name].exists() else
                            tifffile.imread(OLD / 'masks/context128_tta' / row.sample_id / f'{m}.tif') if name == 'baseline' else
                            refined_instances(probability, metadata['area_limits'][m], **candidates()[name]))
                    directory = OUT / 'masks' / name / row.sample_id
                    directory.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(directory / f'{m}.tif', mask, compression='zlib')
                    results.append(dict(sample_id=row.sample_id, modality=m, method=name,
                                        pq=mask_quality(mask, json.loads(getattr(row, m+'_instances')))[0]))
            pd.DataFrame(results).to_csv(OUT / 'segmentation.csv', index=False)
            print('SEGMENTED', row.sample_id, flush=True)
    frame = pd.DataFrame(results)
    selected = {m: frame[frame.modality == m].groupby('method').pq.mean().idxmax() for m in MODALITIES}
    (OUT / 'selected.json').write_text(json.dumps(dict(segmentation=selected, matching=dict(gate=8, cutoff=.28, mutual=True)), indent=2))
    print('SEGMENTATION SELECTED', selected, flush=True)


def matching():
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    settings = json.loads((OUT / 'selected.json').read_text())
    segmentation = pd.read_csv(OUT / 'segmentation.csv')
    coarse = json.loads((OLD / 'image_alignment.json').read_text())
    records, evaluated = [], {}
    for mode in ['selected', 'grouped']:
        for row in rows.itertuples(index=False):
            methods = settings['segmentation']
            if mode == 'grouped':
                mouse = row.sample_id.split('__')[0]
                training = segmentation[segmentation.sample_id.str.split('__').str[0] != mouse]
                methods = {m: training[training.modality == m].groupby('method').pq.mean().idxmax() for m in MODALITIES}
            key = (row.sample_id, tuple(methods[m] for m in MODALITIES))
            if key not in evaluated:
                labels = [tifffile.imread(OUT / 'masks' / methods[m] / row.sample_id / f'{m}.tif') for m in MODALITIES]
                matrix, response = coarse[row.sample_id]
                matrix, statistics = initialize(read_pair('Data/training', row.sample_id), labels,
                    coarse=(np.asarray(matrix) if matrix is not None else None, response))
                pairs = quick_pairs(labels, matrix, statistics, **settings['matching'])
                evaluated[key] = evaluate(labels, pairs, row)
            records.append(dict(mode=mode, sample_id=row.sample_id, source_invivo=methods['invivo'],
                                source_exvivo=methods['exvivo'], **evaluated[key]))
        print(mode, summarize(pd.DataFrame(records).query('mode == @mode')), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'matching.csv', index=False)
    summaries = {mode: summarize(group) for mode, group in frame.groupby('mode')}
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def write_submission(output='Kaggle Submissions/unet_refined.csv'):
    choice = json.loads((OUT / 'selected.json').read_text())
    models, metadata = load_models('outputs/q5/unet_complete/final')
    sample = pd.read_csv('Data/sample_submission.csv', dtype=str, keep_default_na=False)
    rows, diagnostics = [], []
    for sid in sample.sample_id:
        images = read_pair('Data/hidden_test', sid)
        labels = []
        for m, image in zip(MODALITIES, images):
            method = choice['segmentation'][m]
            if method == 'baseline':
                path = OLD / 'inference' / sid / f'{m}.tif'
                mask = (tifffile.imread(path) if path.exists() else instances_from_probabilities(
                    dense_prediction(image, models[m], tile=128, augment=True), metadata['area_limits'][m], **metadata['postprocessing'][m]))
            else:
                mask = refined_instances(dense_prediction(image, models[m], tile=128, augment=True),
                                         metadata['area_limits'][m], **candidates()[method])
            labels.append(mask)
            directory = OUT / 'inference' / sid
            directory.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(directory / f'{m}.tif', mask, compression='zlib')
        matrix, stats = initialize(images, labels)
        pairs = quick_pairs(labels, matrix, stats, **choice['matching'])
        row = dict(sample_id=sid, match_pairs=json.dumps([[f'IV_{i:06d}', f'EX_{j:06d}'] for i, j in pairs]))
        for m, prefix, mask in zip(MODALITIES, ['IV', 'EX'], labels):
            row[m+'_instances'] = json.dumps(instances(mask, prefix), separators=(',', ':'))
        rows.append(row)
        diagnostics.append(dict(sample_id=sid, invivo=int(labels[0].max()), exvivo=int(labels[1].max()), pairs=len(pairs), **stats))
        print('PREDICTED', sid, len(pairs), flush=True)
    frame = pd.DataFrame(rows, columns=sample.columns)
    validate_submission(frame, sample, 'Data/hidden_test')
    frame.to_csv(output, index=False)
    assert pd.read_csv(output, dtype=str, keep_default_na=False).equals(frame)
    pd.DataFrame(diagnostics).to_csv(OUT / 'submission_diagnostics.csv', index=False)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'benchmark', 'matching', 'submission'], required=True)
    parser.add_argument('--output', default='Kaggle Submissions/unet_refined.csv')
    args = parser.parse_args()
    {'pilot': pilot, 'benchmark': benchmark, 'matching': matching,
     'submission': lambda: write_submission(args.output)}[args.stage]()
