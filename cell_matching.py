"""Classical cell segmentation, registration and the handout's local metric."""
from pathlib import Path
import json
import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.segmentation import watershed, relabel_sequential


def encode_rle(mask):
    flat = np.asarray(mask, bool).ravel()
    edges = np.flatnonzero(np.diff(np.r_[False, flat, False])).reshape(-1, 2)
    return ' '.join(f'{s} {e-s}' for s, e in edges)


def decode_rle(rle, shape):
    runs = np.array(rle.split(), dtype=np.int64).reshape(-1, 2)
    flat = np.zeros(np.prod(shape), bool)
    if len(runs):
        starts, lengths = runs.T
        assert (starts >= 0).all() and (lengths > 0).all()
        assert (starts + lengths <= flat.size).all()
        assert (starts[1:] >= (starts + lengths)[:-1]).all()
        for start, length in runs:
            flat[start:start+length] = True
    return flat.reshape(shape)


def read_pair(root, sample_id):
    folder = Path(root).joinpath(*sample_id.split('__'))
    images = [tifffile.imread(folder / f'{m}.tif') for m in ['invivo', 'exvivo']]
    assert all(im.ndim == 2 and im.dtype == np.uint16 for im in images)
    return images


def normalize(image):
    low, high = np.percentile(image, [1, 99.8])
    return np.clip((image.astype(np.float32) - low) / max(high-low, 1), 0, 1).astype(np.float32)


def segment(image, sigma=2.5, threshold=.025, fraction=.3, min_area=25, max_area=350, blobness=0, dilation=0, max_elongation=3, denoise=False, multiscale=False, clean=False):
    image = normalize(image)
    if denoise:
        image = ndi.median_filter(image, size=3)
    smooth = ndi.gaussian_filter(image, 1)
    signal = np.maximum(smooth - ndi.gaussian_filter(image, 10), 0)
    scales = [sigma*.7, sigma, sigma*1.3] if multiscale else [sigma]
    response = np.maximum.reduce([ndi.gaussian_filter(signal, s) - ndi.gaussian_filter(signal, s*1.6)
                                  for s in scales])
    peaks = peak_local_max(response, min_distance=4, threshold_abs=threshold, exclude_border=False)
    if blobness and len(peaks):
        xx = ndi.gaussian_filter(image, sigma, order=(0, 2))[tuple(peaks.T)]
        yy = ndi.gaussian_filter(image, sigma, order=(2, 0))[tuple(peaks.T)]
        xy = ndi.gaussian_filter(image, sigma, order=(1, 1))[tuple(peaks.T)]
        strength = 4 * np.maximum(xx * yy - xy**2, 0) / ((xx + yy)**2 + 1e-9)
        peaks = peaks[(strength > blobness) & (xx + yy < 0)]
    markers = np.zeros(image.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks)+1)
    labels = watershed(-signal, markers, mask=signal > .005)
    heights = ndi.maximum(signal, labels, np.arange(len(peaks)+1))
    labels[signal < fraction * heights[labels]] = 0
    labels[ndi.distance_transform_edt(markers == 0) > 12] = 0
    counts = np.bincount(labels.ravel())
    keep = (counts >= min_area) & (counts <= max_area)
    keep[0] = False
    labels[~keep[labels]] = 0
    if dilation:
        from skimage.segmentation import expand_labels
        labels = expand_labels(labels, distance=dilation)
    props = regionprops_table(labels, properties=["label", "axis_major_length", "axis_minor_length"])
    ratio = props["axis_major_length"] / np.maximum(props["axis_minor_length"], 1)
    labels[np.isin(labels, props["label"][ratio > max_elongation])] = 0
    if clean:
        labels = clean_instances(labels)
    return relabel_sequential(labels)[0].astype(np.int32)


def clean_instances(labels):
    """Keep each instance's largest component and fill holes without overwriting neighbors."""
    result = labels.copy()
    for i, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            continue
        parts, _ = ndi.label(labels[slices] == i)
        mask = ndi.binary_fill_holes(parts == (np.bincount(parts.ravel())[1:].argmax()+1))
        result[slices][result[slices] == i] = 0
        result[slices][mask & (result[slices] == 0)] = i
    return result


def mask_quality(prediction, truth):
    """PQ and predicted-label -> GT-ID map; strict IoU > .75.

    Empty/empty PQ is defined as 1 locally; official empty convention unverified.
    Ground-truth masks are handled independently, allowing overlaps.
    """
    areas = np.bincount(prediction.ravel())
    candidates = []
    for key, rle in truth.items():
        runs = np.array(rle.split(), dtype=np.int64).reshape(-1, 2)
        pixels = np.concatenate([prediction.ravel()[s:s+n] for s, n in runs]) if len(runs) else []
        overlap = np.bincount(np.asarray(pixels, dtype=int), minlength=len(areas))
        iou = overlap / np.maximum(areas + runs[:, 1].sum() - overlap, 1)
        for label in np.flatnonzero(iou[1:] > .75) + 1:
            candidates.append((iou[label], int(label), key))
    matched, used, total = {}, set(), 0.
    for iou, label, key in sorted(candidates, reverse=True):
        if label not in matched and key not in used:
            matched[label] = key
            used.add(key)
            total += iou
    denominator = .5 * (np.count_nonzero(areas[1:]) + len(truth))
    return total / denominator if denominator else 1., matched


def centers(labels):
    ids = np.arange(1, labels.max()+1)
    if not len(ids):
        return np.empty((0, 2))
    return np.array(ndi.center_of_mass(labels > 0, labels, ids))[:, ::-1].reshape(-1, 2)


def register(source, target):
    """Search rotation, unequal scales, shear and translation; source -> target."""
    from itertools import product
    def response(image):
        image = normalize(image)
        band = np.maximum(ndi.gaussian_filter(image, 2) - ndi.gaussian_filter(image, 6), 0)
        return cv2.resize(band, None, fx=.33, fy=.33, interpolation=cv2.INTER_AREA)
    a, b = response(source), response(target)
    if min(a.std(), b.std()) < 1e-8:
        return None, 0.
    h, w = a.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    best = (-1., None, (0., 1., 1., 0.))
    for stage in range(2):
        if stage == 0:
            grid = [np.arange(-20, 21, 2), [.9, 1., 1.1], [.8, .9, 1.], [-.1, 0., .1]]
        else:
            grid = [value + np.array([-step, 0, step])
                    for value, step in zip(best[2], [1, .03, .03, .05])]
        for angle, sx, sy, shear in product(*grid):
            linear = cv2.getRotationMatrix2D((0, 0), float(angle), 1)[:, :2] @ [[sx, shear], [0, sy]]
            warped = corners @ linear.T
            low, high = np.floor(warped.min(0)), np.ceil(warped.max(0))
            matrix = np.c_[linear, -low]
            template = cv2.warpAffine(a, matrix, tuple((high-low).astype(int)))
            if any(x > y for x, y in zip(template.shape, b.shape)):
                continue
            _, score, _, position = cv2.minMaxLoc(cv2.matchTemplate(b, template, cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                matrix[:, 2] += position
                best = (score, matrix.copy(), (angle, sx, sy, shear))
    matrix = best[1]
    if matrix is not None:
        matrix[:, 2] /= .33
    return matrix, best[0]


def alignment_support(source, target, matrix, radius=np.inf):
    warped = source @ matrix[:, :2].T + matrix[:, 2]
    distance, nearest = cKDTree(target).query(warped)
    _, reverse = cKDTree(warped).query(target)
    mutual = (reverse[nearest] == np.arange(len(source))) & (distance < radius)
    return float(np.exp(-.5*(distance[mutual]/3)**2).sum())


def match_cells(source_labels, target_labels, matrix, gate=6, guard=False, shape_weight=0):
    """Refine reliable mutual neighbors, then assign with explicit unmatched options."""
    a, b = centers(source_labels), centers(target_labels)
    if matrix is None or min(len(a), len(b)) < 3:
        return [], matrix
    tree = cKDTree(b)
    for _ in range(5):
        warped = a @ matrix[:, :2].T + matrix[:, 2]
        distance, nearest = tree.query(warped)
        _, reverse = cKDTree(warped).query(b)
        keep = (distance < 10) & (reverse[nearest] == np.arange(len(a)))
        if keep.sum() < 6:
            break
        cv2.setRNGSeed(7)
        fitted, inliers = cv2.estimateAffine2D(a[keep], b[nearest[keep]], method=cv2.RANSAC,
                                                    ransacReprojThreshold=3)
        if fitted is None or inliers.sum() < 6:
            break
        if np.linalg.det(fitted[:, :2]) <= 0 or np.linalg.cond(fitted[:, :2]) > 3:
            break
        if guard and alignment_support(a, b, fitted) <= alignment_support(a, b, matrix):
            break
        matrix = fitted
    distance = np.linalg.norm((a @ matrix[:, :2].T + matrix[:, 2])[:, None] - b, axis=2)
    cost = distance.copy()
    if shape_weight:
        areas = [np.bincount(mask.ravel())[1:] for mask in [source_labels, target_labels]]
        ratio = areas[0][:, None] * abs(np.linalg.det(matrix[:, :2])) / np.maximum(areas[1], 1)
        cost += shape_weight * abs(np.log(np.maximum(ratio, 1e-6)))
    # A dummy column allows every source cell to remain unmatched.
    costs = np.c_[np.where(distance < gate, cost, 1e6), np.full((len(a), len(a)), gate)]
    rows, cols = linear_sum_assignment(costs)
    pairs = [(int(i+1), int(j+1)) for i, j in zip(rows, cols) if j < len(b) and cost[i, j] < gate]
    return pairs, matrix


def register_cells(labels):
    """Generate affine hypotheses from local cell triangles; no annotated correspondences."""
    from itertools import combinations

    def triangles(points):
        _, neighbors = cKDTree(points).query(points, k=min(10, len(points)))
        ids = np.array(sorted({tuple(sorted((i, int(j), int(k)))) for i, row in enumerate(neighbors)
                               for j, k in combinations(row[1:], 2)}))
        p = points[ids]
        sides = np.stack([np.linalg.norm(p[:, j]-p[:, k], axis=1)
                          for j, k in [(1, 2), (0, 2), (0, 1)]], axis=1)
        p = np.take_along_axis(p, sides.argsort(1)[:, :, None], axis=1)
        lengths = np.sort(sides, axis=1)
        cross = np.linalg.det(np.stack([p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]], axis=2))
        keep = (abs(cross) > 60) & (lengths[:, 0] > 10)
        descriptor = np.c_[lengths[:, :2]/lengths[:, 2, None], lengths[:, 2]/100]
        return p[keep], descriptor[keep]

    a, b = [centers(mask) for mask in labels]
    if min(len(a), len(b)) < 10:
        return None, 0.
    p, x = triangles(a)
    q, y = triangles(b)
    if min(len(p), len(q)) == 0:
        return None, 0.
    distance, nearest = cKDTree(y).query(x, k=min(10, len(y)))
    distance, nearest = distance.reshape(len(x), -1), nearest.reshape(len(x), -1)
    i, j = np.where(distance < .12)
    p, q, difference = p[i], q[nearest[i, j]], distance[i, j]
    u = (np.stack([q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]], axis=2)
         @ np.linalg.inv(np.stack([p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]], axis=2)))
    translation = q[:, 0] - np.einsum('nij,nj->ni', u, p[:, 0])
    scales = np.linalg.svd(u, compute_uv=False)
    angle = np.degrees(np.arctan2(-u[:, 1, 0], u[:, 0, 0]))
    keep = (np.linalg.det(u) > 0) & (scales.min(1) > .8) & (scales.max(1) < 1.2) & (abs(angle) < 25)
    hypotheses = np.concatenate([u[keep], translation[keep, :, None]], axis=2)
    tree, ranked = cKDTree(b), []
    for index in np.argsort(difference[keep])[:4000]:
        t = hypotheses[index]
        d, _ = tree.query(a @ t[:, :2].T + t[:, 2])
        ranked.append((np.exp(-.5*(d/3)**2).sum(), index))
    best, matrix = -1., None
    for _, index in sorted(ranked, reverse=True)[:12]:
        _, fitted = match_cells(*labels, hypotheses[index])
        quality = alignment_support(a, b, fitted, radius=8)
        if quality > best:
            best, matrix = quality, fitted
    return matrix, max(best, 0.) / min(len(a), len(b))


def evaluate(labels, pairs, row):
    pqs, maps = zip(*(mask_quality(mask, json.loads(getattr(row, m+'_instances')))
                      for mask, m in zip(labels, ['invivo', 'exvivo'])))
    verified = {tuple(pair) for pair in json.loads(row.match_pairs)}
    correct = sum((maps[0].get(i), maps[1].get(j)) in verified for i, j in pairs)
    recoverable = sum(i in maps[0].values() and j in maps[1].values() for i, j in verified)
    return dict(pq_invivo=pqs[0], pq_exvivo=pqs[1], correct_pairs=correct, recoverable_pairs=recoverable,
                predicted_pairs=len(pairs), verified_pairs=len(verified))


def summarize(frame):
    pq_iv, pq_ex = frame[['pq_invivo', 'pq_exvivo']].mean()
    denominator = frame.predicted_pairs.sum() + frame.verified_pairs.sum()
    f1 = 2 * frame.correct_pairs.sum() / denominator if denominator else 1.
    return dict(pq_invivo=float(pq_iv), pq_exvivo=float(pq_ex), f1_match=float(f1),
                score=float(.25*pq_iv + .25*pq_ex + .5*f1))


def instances(labels, prefix):
    return {f'{prefix}_{i:06d}': encode_rle(labels == i) for i in range(1, labels.max()+1)}


# Preserve the previous settings/defaults for the separate U-Net experiment.
PARAMETERS = {
    'invivo': dict(sigma=2.5, threshold=.025, fraction=.15, dilation=1, max_elongation=3),
    'exvivo': dict(sigma=3, threshold=.12, fraction=.15, blobness=.6, max_elongation=3),
    'matching': dict(gate=8, cutoff=.14),
}


REFINED_PARAMETERS = {
    'invivo': PARAMETERS['invivo'] | dict(denoise=True, multiscale=True, fraction=.2, clean=True),
    'exvivo': PARAMETERS['exvivo'] | dict(denoise=True, fraction=.2, clean=True),
    'matching': dict(method='triangles', gate=8, cutoff=.16, guard=True, shape_weight=2),
}


def predict_pair(images, parameters=None):
    parameters = PARAMETERS if parameters is None else parameters
    labels = [segment(im, **parameters[m]) for im, m in zip(images, ['invivo', 'exvivo'])]
    matching = parameters['matching'].copy()
    method, cutoff = matching.pop('method', 'correlation'), matching.pop('cutoff')
    matrix, correlation = register_cells(labels) if method == 'triangles' else register(*images)
    pairs, matrix = match_cells(*labels, matrix, **matching) if correlation >= cutoff else ([], matrix)
    return labels, pairs, matrix, correlation


def validate_submission(frame, sample, root):
    assert list(frame.columns) == list(sample.columns)
    assert frame.sample_id.tolist() == sample.sample_id.tolist() and frame.sample_id.is_unique
    for row in frame.itertuples(index=False):
        dictionaries = [json.loads(row.invivo_instances), json.loads(row.exvivo_instances)]
        pairs = json.loads(row.match_pairs)
        assert all(i in dictionaries[0] and j in dictionaries[1] for i, j in pairs)
        assert len({i for i, j in pairs}) == len(pairs) == len({j for i, j in pairs})
        folder = Path(root).joinpath(*row.sample_id.split('__'))
        for m, dictionary in zip(['invivo', 'exvivo'], dictionaries):
            with tifffile.TiffFile(folder / f'{m}.tif') as image:
                shape = image.pages[0].shape
            for rle in dictionary.values():
                mask = decode_rle(rle, shape)
                assert mask.any() and encode_rle(mask) == rle


def write_submission(root='Data', output='Kaggle Submissions/classical_refined.csv', parameters=None):
    import pandas as pd
    root = Path(root)
    sample = pd.read_csv(root / 'sample_submission.csv', dtype=str, keep_default_na=False)
    rows = []
    for sample_id in sample.sample_id:
        labels, pairs, _, correlation = predict_pair(read_pair(root / 'hidden_test', sample_id),
                                                    REFINED_PARAMETERS if parameters is None else parameters)
        row = dict(sample_id=sample_id)
        for m, prefix, mask in zip(['invivo', 'exvivo'], ['IV', 'EX'], labels):
            row[m+'_instances'] = json.dumps(instances(mask, prefix), separators=(',', ':'))
        row['match_pairs'] = json.dumps([[f'IV_{i:06d}', f'EX_{j:06d}'] for i, j in pairs])
        rows.append(row)
        print(f'{sample_id}: {labels[0].max()}/{labels[1].max()} cells, {len(pairs)} pairs, correlation={correlation:.3f}')
    frame = pd.DataFrame(rows, columns=sample.columns)
    validate_submission(frame, sample, root / 'hidden_test')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    assert pd.read_csv(output, dtype=str, keep_default_na=False).equals(frame)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='Data')
    parser.add_argument('--output', default='Kaggle Submissions/classical_refined.csv')
    args = parser.parse_args()
    write_submission(args.data, args.output)
