"""Evaluate cheap correspondence calibration by excluded mouse."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from cell_matching import read_pair, mask_quality, summarize
from cell_alignment import initialize
from cell_pair_scores import pair_features, correspondence_labels, fit_calibration, pair_scores


def benchmark(refined=False):
    root = Path('outputs/q5/unet_refined')
    out = root / ('pair_refined' if refined else 'pair_baseline')
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    methods = json.loads((root / 'selected.json').read_text())['segmentation'] if refined else None
    coarse = json.loads(Path('outputs/q5/advanced/image_alignment.json').read_text())
    features, labels, regions, metrics = [], [], [], []
    for row in rows.itertuples(index=False):
        masks = [tifffile.imread((root / 'masks' / methods[m] if refined else Path('outputs/q5/advanced/masks/context128_tta'))
                                / row.sample_id / f'{m}.tif') for m in ['invivo', 'exvivo']]
        images = read_pair('Data/training', row.sample_id)
        matrix, response = coarse[row.sample_id]
        matrix, stats = initialize(images, masks, coarse=(np.asarray(matrix) if matrix is not None else None, response))
        pairs, x = pair_features(images, masks, matrix, stats)
        pqs, maps = zip(*(mask_quality(mask, json.loads(getattr(row, m+'_instances')))
                         for mask, m in zip(masks, ['invivo', 'exvivo'])))
        verified = {tuple(pair) for pair in json.loads(row.match_pairs)}
        features.extend(x);labels.extend(correspondence_labels(pairs, maps, verified))
        regions.extend([row.sample_id]*len(pairs))
        metrics.append(dict(sample_id=row.sample_id, pq_invivo=pqs[0], pq_exvivo=pqs[1], verified_pairs=len(verified)))
        print('FEATURES', row.sample_id, len(pairs), flush=True)
    x, y, ids = np.asarray(features), np.asarray(labels), np.asarray(regions)
    np.savez_compressed(out / 'features.npz', features=x, labels=y, sample_ids=ids)
    base = pd.DataFrame(metrics)
    def score_regions(scores, threshold, selected_ids):
        records = []
        for record in base[base.sample_id.isin(selected_ids)].to_dict('records'):
            accepted = (ids == record['sample_id']) & (scores >= threshold)
            records.append(dict(**record, predicted_pairs=int(accepted.sum()), correct_pairs=int((accepted & (y == 1)).sum())))
        return pd.DataFrame(records)
    subjects = np.array([sid.split('__')[0] for sid in ids])
    grouped = []
    for mouse in sorted(rows.sample_id.str.split('__').str[0].unique()):
        train = subjects != mouse
        calibration = fit_calibration(x[train], y[train])
        scores = pair_scores(x, calibration)
        train_ids = base[base.sample_id.str.split('__').str[0] != mouse].sample_id
        candidates = [(summarize(score_regions(scores, threshold, train_ids))['score'], threshold)
                      for threshold in [.2, .35, .5, .65, .8]]
        _, cutoff = max(candidates)
        held_ids = base[base.sample_id.str.split('__').str[0] == mouse].sample_id
        held = score_regions(scores, cutoff, held_ids)
        held['cutoff'] = cutoff;grouped.append(held)
        print('HELD', mouse, cutoff, summarize(held), flush=True)
    grouped = pd.concat(grouped, ignore_index=True)
    grouped.to_csv(out / 'grouped.csv', index=False)
    calibration = fit_calibration(x, y)
    scores = pair_scores(x, calibration)
    candidates = [(summarize(score_regions(scores, t, base.sample_id))['score'], t) for t in [.2, .35, .5, .65, .8]]
    _, cutoff = max(candidates)
    selected = score_regions(scores, cutoff, base.sample_id)
    selected.to_csv(out / 'selected.csv', index=False)
    (out / 'calibration.json').write_text(json.dumps(dict(calibration=calibration, cutoff=cutoff), indent=2))
    summary = dict(grouped=summarize(grouped), selected=summarize(selected),
                   positives=int((y == 1).sum()), confirmed_errors=int((y == 0).sum()), unknown=int((y == -1).sum()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', summary, flush=True)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refined', action='store_true')
    args = parser.parse_args()
    benchmark(args.refined)
