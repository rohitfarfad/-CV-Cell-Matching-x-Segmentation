"""Quick matching calibration using saved U-Net masks; no retraining."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from cell_matching import read_pair, centers, mask_quality, summarize, instances, validate_submission
from cell_alignment import initialize, assign

OUT = Path('outputs/q5/unet_quick')
SAVED = Path('outputs/q5/advanced')


def quick_pairs(labels, matrix, statistics, gate=8, cutoff=.25, mutual=False):
    if matrix is None or statistics['confidence'] < cutoff:
        return []
    a, b = [centers(mask) for mask in labels]
    warped = a @ matrix[:, :2].T + matrix[:, 2]
    pairs = assign(labels, warped, matrix, gate=gate, structure=1)
    if mutual and pairs:
        _, forward = cKDTree(b).query(warped)
        _, reverse = cKDTree(warped).query(b)
        pairs = [(i, j) for i, j in pairs if forward[i-1] == j-1 and reverse[j-1] == i-1]
    return pairs


def benchmark():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    coarse = json.loads((SAVED / 'image_alignment.json').read_text())
    results = []
    for row in rows.itertuples(index=False):
        labels = [tifffile.imread(SAVED / 'masks/context128_tta' / row.sample_id / f'{m}.tif')
                  for m in ['invivo', 'exvivo']]
        image_matrix, response = coarse[row.sample_id]
        initial = initialize(read_pair('Data/training', row.sample_id), labels,
                             coarse=(np.asarray(image_matrix) if image_matrix is not None else None, response))
        pqs, maps = zip(*(mask_quality(mask, json.loads(getattr(row, m+'_instances')))
                         for mask, m in zip(labels, ['invivo', 'exvivo'])))
        verified = {tuple(pair) for pair in json.loads(row.match_pairs)}
        # Assign once per gate; the other checks only remove pairs.
        for gate in [8, 10]:
            for mutual in [False, True]:
                pairs = quick_pairs(labels, *initial, gate=gate, cutoff=0, mutual=mutual)
                correct = sum((maps[0].get(i), maps[1].get(j)) in verified for i, j in pairs)
                for cutoff in [.25, .28, .30]:
                    accepted = initial[1]['confidence'] >= cutoff
                    results.append(dict(sample_id=row.sample_id, gate=gate, cutoff=cutoff, mutual=mutual,
                        pq_invivo=pqs[0], pq_exvivo=pqs[1], correct_pairs=correct if accepted else 0,
                        predicted_pairs=len(pairs) if accepted else 0, verified_pairs=len(verified)))
        print('EVALUATED', row.sample_id, flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / 'per_region.csv', index=False)
    columns = ['gate', 'cutoff', 'mutual']
    table = pd.DataFrame([dict(zip(columns, key), **summarize(group))
                          for key, group in frame.groupby(columns)]).sort_values('score', ascending=False)
    table.to_csv(OUT / 'comparison.csv', index=False)
    choice = table.iloc[0][columns].to_dict()
    choice = dict(gate=int(choice['gate']), cutoff=float(choice['cutoff']), mutual=bool(choice['mutual']))
    (OUT / 'selected.json').write_text(json.dumps(choice, indent=2))
    held = []
    mice = frame.sample_id.str.split('__').str[0]
    for mouse in sorted(mice.unique()):
        train = frame[mice != mouse]
        candidates = [(summarize(g)['score'], key) for key, g in train.groupby(columns)]
        _, setting = max(candidates)
        keep = (mice == mouse) & np.logical_and.reduce([frame[k] == v for k, v in zip(columns, setting)])
        held.append(frame[keep])
    held = pd.concat(held, ignore_index=True)
    held.to_csv(OUT / 'grouped_per_region.csv', index=False)
    (OUT / 'grouped_summary.json').write_text(json.dumps(summarize(held), indent=2))
    print(table.head().to_string(index=False), flush=True)
    print('GROUPED', summarize(held), flush=True)
    return table


def write_submission(output='Kaggle Submissions/unet_quick.csv'):
    settings = json.loads((OUT / 'selected.json').read_text())
    sample = pd.read_csv('Data/sample_submission.csv', dtype=str, keep_default_na=False)
    rows, diagnostics = [], []
    models = metadata = None
    for sid in sample.sample_id:
        images = read_pair('Data/hidden_test', sid)
        paths = [SAVED / 'inference' / sid / f'{m}.tif' for m in ['invivo', 'exvivo']]
        if all(p.exists() for p in paths):
            assert json.loads((SAVED / 'inference_config.json').read_text()) == dict(invivo='context128_tta', exvivo='context128_tta')
            labels = [tifffile.imread(p) for p in paths]
        else:
            from unet_cells import load_models, instances_from_probabilities
            from unet_inference import dense_prediction
            if models is None:
                models, metadata = load_models('outputs/q5/unet_complete/final')
            labels = [instances_from_probabilities(dense_prediction(im, models[m], tile=128, augment=True),
                metadata['area_limits'][m], **metadata['postprocessing'][m])
                for im, m in zip(images, ['invivo', 'exvivo'])]
        matrix, stats = initialize(images, labels)
        pairs = quick_pairs(labels, matrix, stats, **settings)
        row = dict(sample_id=sid, match_pairs=json.dumps([[f'IV_{i:06d}', f'EX_{j:06d}'] for i, j in pairs]))
        for m, prefix, mask in zip(['invivo', 'exvivo'], ['IV', 'EX'], labels):
            row[m+'_instances'] = json.dumps(instances(mask, prefix), separators=(',', ':'))
        rows.append(row)
        diagnostics.append(dict(sample_id=sid, invivo=int(labels[0].max()), exvivo=int(labels[1].max()), pairs=len(pairs), **stats))
        print('MATCHED', sid, len(pairs), flush=True)
    frame = pd.DataFrame(rows, columns=sample.columns)
    validate_submission(frame, sample, 'Data/hidden_test')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    assert pd.read_csv(output, dtype=str, keep_default_na=False).equals(frame)
    pd.DataFrame(diagnostics).to_csv(OUT / 'submission_diagnostics.csv', index=False)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', default='Kaggle Submissions/unet_quick.csv')
    args = parser.parse_args()
    benchmark() if args.benchmark else write_submission(args.output)
