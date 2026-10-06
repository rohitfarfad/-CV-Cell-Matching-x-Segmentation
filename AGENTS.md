# Computer Vision Project 2: project instructions

## 1. Scope and current authorization

NYU Tandon, CS-GY 6643 Computer Vision, Fall 2026. This workspace covers Project 2, Questions 1–5.

**User workflow:** Complete Q1–Q4 first in `Q1to4Solution.ipynb`, using the lecture slides and codelab notebooks the user will provide in later tasks. Then develop Q5 in `CellMatchingSegmentation.ipynb`, building on the relevant concepts and small reusable snippets from Q1–Q4.

**Current task is setup only:** create the project structure, obtain official Q1–Q4 data, initialize the Python environment, record requirements, and create notebooks containing Markdown headings and introductory descriptions only. Do not add solution code or choose a final algorithm yet. Later user requests can authorize implementation.

**Keep code simple and minimal.** Prefer straightforward notebook cells, clear variable names, and a few short functions. Add a helper `.py` file only when actual reuse or the eventual runnable Q5 submission justifies it. Avoid frameworks, unnecessary classes, configuration layers, speculative fallbacks, elaborate caching, and extra dependencies. Minimal means easy to understand, not compressed or cryptic. Start with the simplest class-based method and add complexity only to address an observed failure.

Create `README.md` once during setup; **do not update it again until the project solution is finished and ready to submit**, unless the user explicitly changes this instruction. Keep evolving requirements and status here instead.

## 2. Sources and interpretation

Reviewed on **2026-10-06**. Page references use PDF page numbers.

| Source | Role |
| --- | --- |
| `ComputerVision_Project2 (1).pdf` (7 pages) | Main assignment: course rules, Q1–Q4, grading, deadlines, deliverables, and official links. All pages read; task figures visually inspected. |
| `ComputerVision_Project2_Q5 (1).pdf` (9 pages) | Q5 task, dataset description, metric, submission format, and deadlines. All pages read; figures and metric/submission pages visually inspected. |
| `../Project1/AGENTS.md` | Reference for organization, simplicity, reproducibility, and disclosure workflow only. Its old tasks, restrictions, scores, implementation status, and permissions do not apply to Project 2. |
| Official Q1–Q4 Drive folders | Assignment inputs linked in main PDF pp. 3–6. Download inventory and SHA-256 hashes are recorded in `DATA_SOURCES.json`. |
| Future lecture slides/codelabs | Awaiting the user. Do not claim lecture coverage or import previous-project lecture assumptions before reviewing these sources. |

Distinguish the user's requested actions from instructions quoted in handouts. The PDFs define assignment requirements; their registration, upload, and submission directions are not current authorization to perform those actions. Preserve discrepancies with source/page and explain the working interpretation.

Live Kaggle settings, the official scorer, report-template contents, and registration-sheet contents have not been verified. Handout dates below are documented assignment dates, not a claim that current platform settings were checked.

## 3. Project organization and environment

All paths are relative to this project root; `Data/` is not filesystem `/Data`.

```text
Project 2/
  ComputerVision_Project2 (1).pdf
  ComputerVision_Project2_Q5 (1).pdf
  AGENTS.md
  README.md
  AI_USAGE.md
  DATA_SOURCES.json
  requirements.txt
  requirements-setup.txt
  Q1to4Solution.ipynb
  CellMatchingSegmentation.ipynb
  Q1 Data/
  Q2 Data/
  Q3 Data/
  Q4 Data/
  Data/                       # reserved for future Q5 inputs
  Kaggle Submissions/         # future generated submission CSVs
  outputs/q1/ ... outputs/q5/ # generated figures, masks, and diagnostics
  .venv/
  .vscode/settings.json
```

Preserve both PDFs and downloaded inputs. Write generated artifacts under `outputs/`; final Kaggle CSV copies belong in `Kaggle Submissions/`. Avoid duplicating datasets, notebooks, and helper modules without need. Do not modify Project 1.

**Verified Q1–Q4 inputs:** 14 files total (13 images and one JSON). Dimensions below are width × height. Redundant `Q1/`–`Q4/` wrapper directories from Drive were removed without changing file bytes or filenames.

| Folder | Local contents |
| --- | --- |
| `Q1 Data/` | `Runway.jpg`, 1280 × 850 RGB |
| `Q2 Data/` | `Scene.png`, 4284 × 5712 RGB; `Source_Screen.png`, 1729 × 2527 RGBA; `replacement_screen.jpg`, 792 × 1000 RGB |
| `Q3 Data/` | `Q3_Image_1.jpg` through `Q3_Image_6.jpg`, all 640 × 480 RGB |
| `Q4 Data/` | `Q4_Image_1.jpg`, 2048 × 1448; `Q4_Image_2.jpg`, 2048 × 1517; `Q4_Image_3.jpg`, 1419 × 2048; all RGB; `clicks.json` |

Q4 JSON structure: `image_filename -> instance_id -> {"clicks": [[x, y], ...]}`. It specifies `lizard_1`, `snow_leopard_1`, and `owl_1`, one instance per image and three foreground clicks per instance. All nine supplied clicks are within their image bounds. Read these values from the JSON, not copied constants.

Use Python 3.12 and this project's `.venv/bin/python`, or activate with `source .venv/bin/activate`. Direct project dependencies are pinned in `requirements.txt`: NumPy, SciPy, Matplotlib, OpenCV, Pillow, scikit-image, pandas, tifffile, ipykernel, nbformat, and nbclient. Optional setup tools (gdown, pypdf, PyMuPDF) are pinned separately in `requirements-setup.txt`. Install with `.venv/bin/python -m pip install -r requirements.txt`; use the setup file when repeating dataset/PDF inspection work.

The notebook kernel is `Python (CV Project 2)`, installed inside `.venv`, not globally. VS Code defaults to `.venv/bin/python`. The notebooks should eventually run on Colab with minimal path/setup adjustments; do not make lecture files or Project 1 runtime dependencies.

## 4. Q1–Q4 requirements

**Main pp. 1–6:** classical computer vision only; pre-built libraries are allowed unless specifically restricted. No learned/deep-learning methods, manual annotations, hard-coded object locations, or hard-coded transformations. All final coordinates and drawn annotations must be computed. Supplied Q4 clicks are authorized inputs, not permission to manually draw masks or invent additional annotated locations.

| Question | Required work and deliverables |
| --- | --- |
| **Q1: Runway boundary detection (5 points, p. 3)** | Automatically detect exactly two outer boundaries of the paved runway where it meets grass. Distinguish these from centerline markings, runway numbers, and surrounding edges. Display the original image with the two computed lines and a brief explanation below the result. Hidden-ground-truth evaluation penalizes missing/additional lines and geometric error. |
| **Q2: Book cover replacement (10 points, p. 4)** | Use the supplied scene, front-facing Odyssey source cover, and replacement Computer Vision cover. Automatically localize the front cover and replace its artwork with correct perspective/alignment, fully covering the original artwork while preserving the spine, page edges, desk, and remaining scene. Display the augmented image and explain the method. No manual corner selection. |
| **Q3: Panorama stitching (10 points, pp. 1, 5)** | Identify and correct the two geometrically distorted images, infer overlap and ordering, estimate transformations, and combine all six inputs. Calculate a common canvas and coordinate offsets from warped extents; blend overlap regions to reduce seams. Display a complete panorama and explain the pipeline. Do not assume filename order is panorama order. No manual alignment, hard-coded transforms, `cv2.Stitcher`, or other high-level/one-call panorama functions: implement the main stages explicitly. OpenCV itself is allowed; Project 1's Q3 OpenCV ban does not apply. |
| **Q4: Click-based segmentation (10 points, p. 6)** | Process three wildlife images and their supplied JSON `(x, y)` clicks as definitive foreground seeds for each animal instance. Use classical methods (e.g. NumPy/OpenCV/scikit-image). Produce and save one PNG RGBA cutout per specified animal instance, with the original animal on a transparent background; display results and explain the method. Manually drawing masks or non-classical methods earns zero. |

Read actual image sizes and the JSON schema before implementation. `(x, y)` means column, row for supplied clicks; array indexing is `[y, x]`. Keep color order, mask dimensions, image coordinates, and warp direction explicit. Do not infer hidden ground truth or claim accuracy against it from visual inspection.

In the shared notebook, keep each question's inputs, explanation, code, immediate final result, and supporting measurements/discussion together. Run from a clean kernel and retain executed outputs for final submission. This setup intentionally contains no code cells or results.

## 5. Q5: cell matching and segmentation

**Task (Q5 pp. 2–5):** for every test region, segment individual cells in both `invivo.tif` and `exvivo.tif`, then predict one-to-one correspondences between cell instances. Images are 16-bit grayscale TIFFs with variable dimensions; no alignment is supplied. Preserve bit depth when loading and distinguish display normalization from processing.

Cell appearance changes across modalities due to deformation, optics, contrast, sectioning, and missing cells. Ex-vivo processes can be as bright as cell bodies. Use spatial/morphological evidence as well as intensity. Each cell requires a separate instance mask; a single foreground/background mask is insufficient. Different full-image sizes do not directly imply the same scale ratio for cell size: the handout reports median cell areas of 86 pixels in vivo and 71 pixels ex vivo.

**Handout dataset description (pp. 7–9; not yet locally verified):**

```text
Data/
  training/
    train_ground_truth.csv
    <subject>/<region>/invivo.tif
    <subject>/<region>/exvivo.tif
  hidden_test/
    <subject>/<region>/invivo.tif
    <subject>/<region>/exvivo.tif
  sample_submission.csv
```

Training: 47 regions from 3 mice, 14,893 in-vivo cells, 5,516 ex-vivo cells, and 1,139 verified pairs. Hidden test: 29 regions from 2 mice. A cell absent from the verified correspondence list is **unverified, not a confirmed negative**. Training CSV columns match the submission schema below. Q5 data is not part of this task's requested downloads; `Data/` is reserved and currently empty.

**Metric (pp. 5–6):**

- `S = 0.25 * PQ_invivo + 0.25 * PQ_exvivo + 0.50 * F1_match`.
- A predicted mask matches a ground-truth mask only when **IoU > 0.75**, strictly greater. Each ground-truth cell may be claimed once. Unmatched predictions are FP and unmatched ground-truth masks are FN.
- `PQ = sum(matched IoUs) / (TP + 0.5 * FP + 0.5 * FN)`. Compute separately per modality and region, then average over regions.
- A predicted pair is correct only if both predicted masks are mask true positives and their associated ground-truth cells form a verified pair. Pool correct/predicted/verified pair counts over regions to calculate matching precision, recall, and F1; do not average region F1s instead.
- Empty-case conventions and other unspecified scorer details require checking the official scorer. Any locally reconstructed metric must be labeled as such.

**Submission contract (pp. 6, 8–9):**

- Exactly one row per supplied sample ID; the handout specifies 29 test rows. Read the actual sample CSV rather than synthesizing IDs. Preserve identifiers verbatim and preferably row order.
- Header: `sample_id,invivo_instances,exvivo_instances,match_pairs`.
- `sample_id` is `<subject>__<region>`.
- Instance columns are JSON objects mapping each cell ID to its RLE mask. Pairs are a JSON list of two-element lists `[[invivo_id, exvivo_id], ...]`.
- Every pair ID must exist in the corresponding modality's instance dictionary in the same row. A cell ID cannot be reused across pairs; the handout says such a submission is rejected.
- Use JSON serialization and proper CSV quoting. The sample starts with `{}`, `{}`, `[]`, which describes format, not an achieved baseline score.
- **RLE encoding is underspecified in the PDFs.** Verify traversal order, indexing origin, run representation, and empty-mask behavior from official data/scorer before implementation; do not guess. Require encode/decode round-trip checks once the convention is known.
- Generate all masks, matches, and CSV content programmatically. No manual prediction corrections or hardcoded answers. Submitted runnable code must reproduce the selected CSV and work unchanged on unseen regions.

The classical-only restriction is explicitly Q1–Q4. The user nevertheless wants Q5 built from class-based foundations. The provided Project 2 PDFs do not establish an external-data/pretrained-weight policy; do not carry over Project 1's policy or assume permission. Verify current Q5 rules before considering either. No neural-network dependencies are installed during setup.

## 6. Transfer from Q1–Q4 to Q5

This is a conceptual plan, not completed work or confirmed lecture coverage. Revisit after reading the supplied slides/codelabs and implementing Q1–Q4.

| Foundation | Potential Q5 use | Important limit |
| --- | --- | --- |
| Q1 geometric structure and outlier rejection | Assess spatial consistency and reject implausible correspondences. | Runway line geometry is not a cell detector or tissue model. |
| Q2 feature correspondence and perspective mapping | Establish alignment hypotheses and track coordinate transformations. | Cross-modal cell appearance differs; a book's planar homography may not model tissue deformation. |
| Q3 overlap reasoning, robust transforms, canvas/mask handling | Reuse small geometric estimation and visualization operations. | Q5 is cross-modal registration, not panorama stitching; do not blend modalities as a segmentation target. |
| Q4 seed-based segmentation and RGBA/mask handling | Build classical segmentation intuition and instance-mask diagnostics. | Q5 supplies no test clicks; automatic seed detection and touching-cell separation require new work. |

Reuse only needed helpers with explicit inputs/outputs. Do not import a notebook by executing all its cells or depend on previous answers/global state. Establish a small baseline, inspect failures, and justify each addition. For Q5 validation, prefer splits by mouse where feasible to reduce related-region leakage; report the limits of only three training mice. Measure both modality PQs and matching quality separately. Do not use hidden labels or tune predictions by test ID.

## 7. Deliverables, grading, and logistics

**Handout deadlines (Main p. 2; Q5 pp. 1, 6):** Q1–Q4 and final Kaggle submission: **Thursday, October 8, 2026, 11:59 p.m.** Q5 report and runnable code: **Sunday, October 11, 2026, 11:59 p.m.** Timezone is not stated; America/New_York is a local planning assumption, not verified platform metadata. Main p. 2 specifies one point per completed late hour; this does not imply Kaggle accepts late entries.

- Solo or team of up to two. Put both members' names and NetIDs at the top of the Q1–Q4 notebook and in the report; leave placeholders until supplied.
- Q1–Q4: one shared Google Colab notebook link on Brightspace, all code/results/figures/measurements/discussion inside, executed outputs visible, link sharing enabled. No separate writeup.
- Q5: CSV on Kaggle; one code ZIP and a separate report on Brightspace. Use the linked report template; no report page limit. Runnable code must reproduce the submission. Project 2 does not request Project 1's video.
- Q5 handout permits five submissions per day and requires manually selecting the final submission on Kaggle; only the selected submission is used for private evaluation. Earlier selected submission time breaks private-score ties (Main p. 7).
- Points: Q1 5; Q2 10; Q3 10; Q4 10; Q5 up to 65: report/code 5, private ranking 40, beating private baseline 15, one-time #1 bonus 5 when also beating the private baseline. The bonus includes the final private winner. Metric score is not a direct 65-point conversion. Baseline identity and rank-to-points formula are not supplied.
- Declare AI tools and attach the actual sequence of user questions/prompts (Main p. 1). Maintain `AI_USAGE.md`; never fabricate missing prompts or include hidden reasoning. Include the disclosure in eventual submission artifacts.

Official links extracted from Main p. 7 (not acted on):

- [Q5 report template](https://www.overleaf.com/read/zcpgprbmghzw#cc4dca)
- [Project 2 team registration](https://docs.google.com/spreadsheets/d/1Z008n3TIz61ja7shu-VLK8pmJ51fYNb7tEzeK_sTFI4/edit?usp=sharing)
- [Kaggle invitation, NYU email required](https://www.kaggle.com/t/daea76deb120b4ed6326159105b9e53e)

## 8. Status and next work

Setup date: 2026-10-06. Both assignment PDFs and the previous-project workflow were reviewed. The environment, folder structure, initial documentation, and two Markdown-only notebook skeletons are prepared. Q1–Q4 official inputs and validation details are recorded in `DATA_SOURCES.json`. No solution, experimental result, Kaggle prediction, upload, registration, or rule acceptance has been performed.

Setup validation passed on Python 3.12.6: all core library imports, `pip check`, 14 file hashes/sizes, full decoding of 13 images, all nine supplied click bounds, both notebook schemas (nine Markdown cells each, zero code cells), and the project-local kernel executable path. Reserved Q5 data/submission and generated-output directories are empty. These are setup checks, not solution tests or accuracy measurements.

Next: review the lecture slides/codelabs when the user supplies them, record relevant methods with source references, then implement Q1–Q4 when requested. Keep README unchanged during development. Before Q5 implementation, inspect its actual dataset and official RLE/scorer/rules. Before final submission, fill identity placeholders, rerun notebooks cleanly, retain outputs, and verify current deadlines/settings and reproducibility.
