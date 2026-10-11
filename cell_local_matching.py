"""Smooth local centroid corrections from reliable mutual neighbors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from cell_alignment import initialize, assign
from cell_matching import centers, read_pair, mask_quality, summarize


def fast_instances(labels, prefix):
    """Encode each instance inside its bounding box using full-image RLE indices."""
    result = {}
    for label, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        y, x = np.where(labels[box] == label)
        indices = (y+box[0].start)*labels.shape[1]+x+box[1].start
        ends = np.r_[np.flatnonzero(np.diff(indices) != 1)+1, len(indices)]
        starts = np.r_[0, ends[:-1]]
        result[f'{prefix}_{label:06d}'] = ' '.join(f'{s} {n}' for s, n in zip(indices[starts], ends-starts))
    return result


def check_instances(dictionary, labels):
    """Decode all RLEs into one native label image and verify exact ownership."""
    decoded = np.zeros_like(labels)
    for key, rle in dictionary.items():
        values = np.fromstring(rle, dtype=np.int64, sep=' ')
        assert values.size and values.size % 2 == 0
        assert ' '.join(map(str, values)) == rle
        runs = values.reshape(-1, 2)
        start, length = runs.T
        assert (start >= 0).all() and (length > 0).all() and (start+length <= labels.size).all()
        assert (start[1:] > (start+length)[:-1]).all()
        for s, n in runs:
            assert not decoded.ravel()[s:s+n].any()
            decoded.ravel()[s:s+n] = int(key.rsplit('_', 1)[1])
    assert np.array_equal(decoded, labels)


def local_warp(source, target, matrix, radius=40, anchor_gate=3):
    warped = source @ matrix[:, :2].T + matrix[:, 2]
    if min(len(source), len(target)) < 6:
        return warped
    distance, nearest = cKDTree(target).query(warped, k=2)
    _, reverse = cKDTree(warped).query(target)
    keep = (distance[:, 0] < anchor_gate) & (distance[:, 1] > 2*distance[:, 0]) & (reverse[nearest[:, 0]] == np.arange(len(source)))
    if keep.sum() < 6:
        return warped
    delta = target[nearest[keep, 0]]-warped[keep]
    squared = np.sum((warped[:, None]-warped[None, keep])**2, axis=2)
    weight = np.exp(-squared/(2*radius**2))
    correction = weight@delta/np.maximum(weight.sum(1)[:, None], 1)
    length = np.linalg.norm(correction, axis=1)
    correction *= np.minimum(1, 3/np.maximum(length, 1e-6))[:, None]
    return warped+correction


def corrected_pairs(labels, matrix, stats, method='affine', cutoff=.28):
    if matrix is None or stats['confidence'] < cutoff:
        return []
    a, b = [centers(mask) for mask in labels]
    warped = a @ matrix[:, :2].T + matrix[:, 2] if method == 'affine' else local_warp(a, b, matrix,
                radius=40 if method == 'local40' else 60, anchor_gate=3 if method == 'local40' else 5)
    pairs = assign(labels, warped, matrix, gate=8, structure=1)
    if not pairs:
        return []
    _, forward = cKDTree(b).query(warped)
    _, reverse = cKDTree(warped).query(b)
    return [(i, j) for i, j in pairs if forward[i-1] == j-1 and reverse[j-1] == i-1]


def benchmark(inputs=False):
    out = Path('outputs/q5/unet_refined')
    rows = pd.read_csv('Data/training/train_ground_truth.csv')
    coarse = json.loads(Path('outputs/q5/advanced/image_alignment.json').read_text())
    choices = json.loads((out / 'input_deployment.json').read_text()) if inputs else None
    suffix = '_input' if inputs else ''
    records = []
    for row in rows.itertuples(index=False):
        masks = [tifffile.imread((out / 'inputs' / choices[m] if inputs else Path('outputs/q5/advanced/masks/context128_tta'))
                                / row.sample_id / f'{m}.tif') for m in ['invivo', 'exvivo']]
        matrix, response = coarse[row.sample_id]
        matrix, stats = initialize(read_pair('Data/training', row.sample_id), masks,
                                  coarse=(np.asarray(matrix) if matrix is not None else None, response))
        pqs, maps = zip(*(mask_quality(mask, json.loads(getattr(row, m+'_instances')))
                         for mask, m in zip(masks, ['invivo', 'exvivo'])))
        verified = {tuple(pair) for pair in json.loads(row.match_pairs)}
        for method in ['affine', 'local40', 'local60']:
            pairs = corrected_pairs(masks, matrix, stats, method, cutoff=0)
            correct = sum((maps[0].get(i), maps[1].get(j)) in verified for i, j in pairs)
            for cutoff in [.25, .28, .30]:
                accept = stats['confidence'] >= cutoff
                records.append(dict(sample_id=row.sample_id, method=method, cutoff=cutoff, pq_invivo=pqs[0], pq_exvivo=pqs[1],
                    correct_pairs=correct if accept else 0, predicted_pairs=len(pairs) if accept else 0, verified_pairs=len(verified)))
        print('LOCAL MATCHING', row.sample_id, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / f'local_matching{suffix}.csv', index=False)
    columns = ['method', 'cutoff']
    table = pd.DataFrame([dict(zip(columns, key), **summarize(group)) for key, group in frame.groupby(columns)]).sort_values('score', ascending=False)
    table.to_csv(out / f'local_summary{suffix}.csv', index=False)
    held = []
    subjects = frame.sample_id.str.split('__').str[0]
    for mouse in sorted(subjects.unique()):
        train = frame[subjects != mouse]
        _, setting = max((summarize(g)['score'], key) for key, g in train.groupby(columns))
        keep = (subjects == mouse) & np.logical_and.reduce([frame[k] == v for k, v in zip(columns, setting)])
        held.append(frame[keep])
    held = pd.concat(held, ignore_index=True)
    held.to_csv(out / f'local_grouped{suffix}.csv', index=False)
    (out / f'local_grouped_summary{suffix}.json').write_text(json.dumps(summarize(held), indent=2))
    print(table.to_string(index=False), flush=True)
    print('LOCAL GROUPED', summarize(held), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', action='store_true')
    args = parser.parse_args()
    benchmark(args.inputs)
