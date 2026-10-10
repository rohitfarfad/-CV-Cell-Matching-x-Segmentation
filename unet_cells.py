"""Small, modality-specific U-Nets and reproducible Q5 inference (no external weights)."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import tifffile, torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, watershed, relabel_sequential
from skimage.feature import peak_local_max
from cell_matching import (normalize, read_pair, mask_quality, register_cells, match_cells,
                           evaluate, summarize, instances, validate_submission)

SEED, EPOCHS, STEPS, BATCH, PATCH = 7, 36, 60, 8, 128
MODALITIES = ('invivo', 'exvivo')
torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def load_record(row, modality, root):
    image = normalize(tifffile.imread(root / "training" / Path(*row.sample_id.split("__")) / f"{modality}.tif"))
    labels = np.zeros(image.shape, np.int32)
    centers, areas = [], []
    for i, rle in enumerate(json.loads(getattr(row, modality + "_instances")).values(), 1):
        runs = np.array(rle.split(), int).reshape(-1, 2)
        pixels = np.concatenate([np.arange(s, s+n) for s, n in runs])
        labels.ravel()[pixels] = i
        centers.append([np.mean(pixels // image.shape[1]), np.mean(pixels % image.shape[1])])
        areas.append(len(pixels))
    foreground = labels > 0
    core = ndi.binary_erosion(foreground & ~find_boundaries(labels, mode="inner"))
    targets = np.stack([foreground, core]).astype(np.float32)
    background = np.argwhere((image > np.percentile(image, 90)) & ~ndi.binary_dilation(foreground, iterations=3))
    assert min(image.shape) >= PATCH
    return image, targets, np.array(centers), background, np.array(areas)


def sample_batch(records, rng):
    xs, ys = [], []
    for _ in range(BATCH):
        image, targets, centers, background, _ = records[rng.integers(len(records))]
        h, w = image.shape
        choice = rng.random()
        points = centers if choice < .6 else background if choice < .8 else []
        y, x = points[rng.integers(len(points))] if len(points) else [rng.integers(h), rng.integers(w)]
        y, x = int(np.clip(y-PATCH//2, 0, h-PATCH)), int(np.clip(x-PATCH//2, 0, w-PATCH))
        a, b = image[y:y+PATCH, x:x+PATCH], targets[:, y:y+PATCH, x:x+PATCH]
        k = rng.integers(4)
        a, b = np.rot90(a, k), np.rot90(b, k, axes=(-2, -1))
        if rng.random() < .5:
            a, b = a[:, ::-1], b[:, :, ::-1]
        a = np.clip(a ** rng.uniform(.8, 1.2) * rng.uniform(.85, 1.15) + rng.normal(0, .01, a.shape), 0, 1)
        xs.append(a[None].astype(np.float32))
        ys.append(b.copy())
    return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device)

def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.bottom = block(1, 16), block(16, 32), block(32, 64)
        self.dec2, self.dec1 = block(96, 32), block(48, 16)
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        d = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return self.head(d)


def loss_function(logits, targets):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((0, 2, 3))
    dice = (2 * intersection + 1) / (probabilities.sum((0, 2, 3)) + targets.sum((0, 2, 3)) + 1)
    # Give the strict-IoU mask boundary extra weight without changing the network.
    boundary = F.max_pool2d(targets, 3, 1, 1) + F.max_pool2d(-targets, 3, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (bce * (1 + 2*boundary)).mean() + (1 - dice).mean()


@torch.inference_mode()
def probabilities(image, model, tile=256, stride=192):
    image = normalize(image)
    h, w = image.shape
    padded = np.pad(image, ((0, max(0, tile-h)), (0, max(0, tile-w))), mode="reflect")
    ph, pw = padded.shape
    ys = sorted(set(list(range(0, ph-tile+1, stride)) + [ph-tile]))
    xs = sorted(set(list(range(0, pw-tile+1, stride)) + [pw-tile]))
    positions = [(y, x) for y in ys for x in xs]
    result, count = np.zeros((2, ph, pw), np.float32), np.zeros((ph, pw), np.float32)
    model.eval()
    for start in range(0, len(positions), BATCH):
        batch = positions[start:start+BATCH]
        patches = np.stack([padded[y:y+tile, x:x+tile] for y, x in batch])[:, None]
        values = model(torch.from_numpy(patches).to(device)).sigmoid().cpu().numpy()
        for (y, x), value in zip(batch, values):
            result[:, y:y+tile, x:x+tile] += value
            count[y:y+tile, x:x+tile] += 1
    assert (count > 0).all()
    return (result / count)[:, :h, :w]


def instances_from_probabilities(probability, limits, threshold=.5, seeds='components'):
    foreground = ndi.binary_fill_holes(probability[0] > threshold)
    if seeds == 'components':
        markers, _ = ndi.label((probability[1] > .5) & foreground)
    else:
        peaks = peak_local_max(ndi.gaussian_filter(probability[1], .7), min_distance=3,
                               threshold_abs=.5, labels=foreground.astype(np.uint8), exclude_border=False)
        markers = np.zeros(foreground.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks)+1)
    labels = watershed(-ndi.distance_transform_edt(foreground), markers, mask=foreground)
    areas = np.bincount(labels.ravel())
    keep = (areas >= limits[0]) & (areas <= limits[1])
    keep[0] = False
    labels[~keep[labels]] = 0
    return relabel_sequential(labels)[0].astype(np.int32)


def train_models(rows, output, root='Data'):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models, history, limits, settings = {}, [], {}, {}
    for modality in MODALITIES:
        records = [load_record(row, modality, root) for row in rows.itertuples(index=False)]
        areas = np.concatenate([record[-1] for record in records])
        limits[modality] = [max(5, int(np.quantile(areas, .005))), int(np.quantile(areas, .995))]
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = SmallUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-4)
        start = time.time()
        for epoch in range(EPOCHS):
            model.train()
            losses = []
            for _ in range(STEPS):
                x, y = sample_batch(records, rng)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(x), y)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            history.append(dict(modality=modality, epoch=epoch+1, loss=float(np.mean(losses)), seconds=time.time()-start))
            pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
            if (epoch+1) % 6 == 0:
                print(f'{output.name} {modality} epoch {epoch+1}/{EPOCHS}: {np.mean(losses):.4f}', flush=True)
        model.eval()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / f'{modality}.pt')
        models[modality] = model
        # Calibrate on weight-training regions only; excluded mouse labels are never used here.
        candidates = [dict(threshold=t, seeds=s) for t in [.4, .5, .6] for s in ['peaks', 'components']]
        scores = [[] for _ in candidates]
        for record, row in zip(records, rows.itertuples(index=False)):
            probability = probabilities(record[0], model)
            truth = json.loads(getattr(row, modality+'_instances'))
            for values, candidate in zip(scores, candidates):
                labels = instances_from_probabilities(probability, limits[modality], **candidate)
                values.append(mask_quality(labels, truth)[0])
        means = [float(np.mean(values)) for values in scores]
        settings[modality] = candidates[int(np.argmax(means))]
        pd.DataFrame([dict(**candidate, pq=pq) for candidate, pq in zip(candidates, means)]).to_csv(output / f'{modality}_calibration.csv', index=False)
        print(modality, settings[modality], 'training calibration PQ', max(means), flush=True)
    metadata = dict(seed=SEED, epochs=EPOCHS, steps=STEPS, batch=BATCH, patch=PATCH,
                    torch_version=torch.__version__, device=str(device), area_limits=limits,
                    postprocessing=settings, training_ids=rows.sample_id.tolist(),
                    matching=dict(gate=8, cutoff=.16, guard=True, shape_weight=2))
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    return models, metadata


def load_models(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'experiment.json').read_text())
    models = {}
    for modality in MODALITIES:
        model = SmallUNet().to(device)
        model.load_state_dict(torch.load(directory / f'{modality}.pt', map_location='cpu', weights_only=True))
        models[modality] = model.eval()
    return models, metadata


def predict_pair(images, models, metadata):
    labels = [instances_from_probabilities(probabilities(image, models[modality]),
              metadata['area_limits'][modality], **metadata['postprocessing'][modality])
              for image, modality in zip(images, MODALITIES)]
    matrix, support = register_cells(labels)
    matching = metadata['matching'].copy()
    cutoff = matching.pop('cutoff')
    pairs, matrix = match_cells(*labels, matrix, **matching) if support >= cutoff else ([], matrix)
    return labels, pairs, matrix, support


def grouped_validation(rows, output, root='Data'):
    output, root = Path(output), Path(root)
    records = []
    subjects = rows.sample_id.str.split('__').str[0]
    for subject in sorted(subjects.unique()):
        training, excluded = rows[subjects != subject], rows[subjects == subject]
        models, metadata = train_models(training, output / subject, root)
        assert set(metadata['training_ids']).isdisjoint(excluded.sample_id)
        for row in excluded.itertuples(index=False):
            labels, pairs, _, support = predict_pair(read_pair(root / 'training', row.sample_id), models, metadata)
            records.append(dict(sample_id=row.sample_id, excluded_mouse=subject, support=support, **evaluate(labels, pairs, row)))
            pd.DataFrame(records).to_csv(output / 'evaluation_per_region.csv', index=False)
        print(subject, summarize(pd.DataFrame(records)[lambda x: x.excluded_mouse == subject]), flush=True)
        del models
    summary = summarize(pd.DataFrame(records))
    (output / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
    return pd.DataFrame(records), summary


def write_submission(directory='outputs/q5/unet_complete/final', root='Data', output='Kaggle Submissions/unet_submission.csv'):
    root = Path(root)
    models, metadata = load_models(directory)
    sample = pd.read_csv(root / 'sample_submission.csv', dtype=str, keep_default_na=False)
    rows = []
    for sample_id in sample.sample_id:
        labels, pairs, _, support = predict_pair(read_pair(root / 'hidden_test', sample_id), models, metadata)
        row = dict(sample_id=sample_id)
        for modality, prefix, mask in zip(MODALITIES, ['IV', 'EX'], labels):
            row[modality+'_instances'] = json.dumps(instances(mask, prefix), separators=(',', ':'))
        row['match_pairs'] = json.dumps([[f'IV_{i:06d}', f'EX_{j:06d}'] for i, j in pairs])
        rows.append(row)
        print(sample_id, [int(mask.max()) for mask in labels], len(pairs), f'support={support:.3f}', flush=True)
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
    parser.add_argument('--models', default='outputs/q5/unet_complete/final')
    parser.add_argument('--data', default='Data')
    parser.add_argument('--output', default='Kaggle Submissions/unet_submission.csv')
    args = parser.parse_args()
    write_submission(args.models, args.data, args.output)
