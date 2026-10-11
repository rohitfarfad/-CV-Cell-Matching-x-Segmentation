"""Small regularized matching calibration; ignore unknown correspondence labels."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from cell_matching import centers, normalize
from cell_alignment import assign


def pair_features(images, labels, matrix, statistics):
    if matrix is None:
        return [], np.empty((0, 11))
    a, b = [centers(mask) for mask in labels]
    if min(len(a), len(b)) == 0:
        return [], np.empty((0, 11))
    warped = a @ matrix[:, :2].T + matrix[:, 2]
    pairs = assign(labels, warped, matrix, gate=10, structure=1)
    forward_distance, forward = cKDTree(b).query(warped)
    reverse_distance, reverse = cKDTree(warped).query(b)
    descriptions = []
    for image, mask in zip(images, labels):
        signal = normalize(image)
        values = []
        for region in regionprops(mask):
            box = tuple(slice(max(0, x-2), min(n, y+2)) for x, y, n in
                        zip(region.bbox[:2], region.bbox[2:], mask.shape))
            inside = mask[box] == region.label
            ring = ndi.binary_dilation(inside, iterations=2) & (mask[box] == 0)
            contrast = signal[box][inside].mean() - signal[box][ring].mean() if ring.any() else 0.
            values.append([region.area, 4*np.pi*region.area/max(region.perimeter**2, 1),
                           region.eccentricity, contrast])
        descriptions.append(np.asarray(values))
    scale = abs(np.linalg.det(matrix[:, :2]))
    features = []
    for i, j in pairs:
        x, y = i-1, j-1
        distance = np.linalg.norm(warped[x]-b[y])
        p, q = descriptions[0][x], descriptions[1][y]
        features.append([distance, abs(np.log(max(p[0]*scale/q[0], 1e-6))),
            statistics['confidence'], float(forward[x] == y and reverse[y] == x),
            distance-forward_distance[x], distance-reverse_distance[y],
            p[1], q[1], abs(p[2]-q[2]), p[3], q[3]])
    return pairs, np.asarray(features).reshape(-1, 11)


def correspondence_labels(pairs, maps, verified):
    forward, reverse = dict(verified), {j: i for i, j in verified}
    result = []
    for i, j in pairs:
        a, b = maps[0].get(i), maps[1].get(j)
        if (a, b) in verified:
            result.append(1)
        elif a is None or b is None or (a in forward and forward[a] != b) or (b in reverse and reverse[b] != a):
            result.append(0)
        else:
            result.append(-1)  # An unverified correspondence is not a known negative.
    return np.asarray(result)


def fit_calibration(features, labels, regularization=.02):
    known = labels >= 0
    x, y = features[known], labels[known]
    assert len(x) and len(np.unique(y)) == 2, 'Calibration needs confirmed positives and negatives.'
    mean, std = x.mean(0), np.maximum(x.std(0), .05)
    x = np.c_[np.ones(len(x)), (x-mean)/std]
    def objective(weights):
        logits = x @ weights
        loss = np.mean(np.logaddexp(0, logits)-y*logits) + regularization*np.sum(weights[1:]**2)
        gradient = x.T @ (expit(logits)-y)/len(y)
        gradient[1:] += 2*regularization*weights[1:]
        return loss, gradient
    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter':100})
    assert fit.success and np.isfinite(fit.x).all(), fit.message
    return dict(mean=mean.tolist(), std=std.tolist(), weights=fit.x.tolist())


def pair_scores(features, calibration):
    x = (features-np.asarray(calibration['mean']))/np.asarray(calibration['std'])
    weights = np.asarray(calibration['weights'])
    return expit(weights[0]+x@weights[1:])
