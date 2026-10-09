# Computer Vision Project 2: project instructions

## 1. Scope and current authorization

NYU Tandon, CS-GY 6643 Computer Vision, Fall 2026. This workspace covers Project 2, Questions 1–5.

**User workflow:** Complete Q1–Q4 first in `Q1to4Solution.ipynb`, using the reviewed lecture slides and local codelab snippets described in §6. Then develop Q5 in `CellMatchingSegmentation.ipynb`, building on the relevant concepts and small reusable snippets from Q1–Q4.

**Current authorization includes Q1–Q5 implementation:** after source review and an approach overview, the user requested “great, let's code it in the notebook” on 2026-10-08. Q1–Q4 have executable pipelines in `Q1to4Solution.ipynb`; Q4 segmentation remains imperfect; the user subsequently authorized a traditional-only Q5 solution with a target combined score of 0.70. Preserve the completed Q1 work and the user's minimal-code preference.

**Keep code simple and minimal.** Prefer straightforward notebook cells, clear variable names, and a few short functions. Add a helper `.py` file only when actual reuse or the eventual runnable Q5 submission justifies it. Avoid frameworks, unnecessary classes, configuration layers, speculative fallbacks, elaborate caching, and extra dependencies. Minimal means easy to understand, not compressed or cryptic. Start with the simplest class-based method and add complexity only to address an observed failure.

Create `README.md` once during setup; **do not update it again until the project solution is finished and ready to submit**, unless the user explicitly changes this instruction. Keep evolving requirements and status here instead.

## 2. Sources and interpretation

Handouts reviewed on **2026-10-06**; Lectures 3–6 reviewed on **2026-10-08**. Page references use PDF page numbers.

| Source | Role |
| --- | --- |
| `ComputerVision_Project2 (1).pdf` (7 pages) | Main assignment: course rules, Q1–Q4, grading, deadlines, deliverables, and official links. All pages read; task figures visually inspected. |
| `ComputerVision_Project2_Q5 (1).pdf` (9 pages) | Q5 task, dataset description, metric, submission format, and deadlines. All pages read; figures and metric/submission pages visually inspected. |
| `../Project1/AGENTS.md` | Reference for organization, simplicity, reproducibility, and disclosure workflow only. Its old tasks, restrictions, scores, implementation status, and permissions do not apply to Project 2. |
| Official Q1–Q4 Drive folders | Assignment inputs linked in main PDF pp. 3–6. Download inventory and SHA-256 hashes are recorded in `DATA_SOURCES.json`. |
| Lecture slides 3–6 | Reviewed supplied PDFs; exact paths, covered concepts, and page references are in §6. These are conceptual sources, not runtime dependencies. |
| Local codelab notebooks | Four relevant notebooks in `../Lectures/CodeLabs/` reviewed on 2026-10-08; snippet locations and reuse limits are in §6.1. Review used local files, not a claim that live Colab versions were checked. |

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

In the shared notebook, keep each question's inputs, explanation, code, immediate final result, and supporting measurements/discussion together. Run from a clean kernel and retain executed outputs for final submission. Q1–Q5 contain executable code/results. Q4 mask quality requires further work.

## 5. Q5: cell matching and segmentation

**Task (Q5 pp. 2–5):** for every test region, segment individual cells in both `invivo.tif` and `exvivo.tif`, then predict one-to-one correspondences between cell instances. Images are 16-bit grayscale TIFFs with variable dimensions; no alignment is supplied. Preserve bit depth when loading and distinguish display normalization from processing.

Cell appearance changes across modalities due to deformation, optics, contrast, sectioning, and missing cells. Ex-vivo processes can be as bright as cell bodies. Use spatial/morphological evidence as well as intensity. Each cell requires a separate instance mask; a single foreground/background mask is insufficient. Different full-image sizes do not directly imply the same scale ratio for cell size: the handout reports median cell areas of 86 pixels in vivo and 71 pixels ex vivo.

**Dataset description (handout pp. 7–9; locally verified 2026-10-09):**

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

Training: 47 regions from 3 mice, 14,893 in-vivo cells, 5,516 ex-vivo cells, and 1,139 verified pairs. Hidden test: 29 regions from 2 mice. A cell absent from the verified correspondence list is **unverified, not a confirmed negative**. Training CSV columns match the submission schema below. The user supplied Q5 data under `Data/`; 152 TIFFs and two CSVs were checked and hashed in `DATA_SOURCES.json`.

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
- **RLE convention clarified by the supplied L5 codelab, cell 18:** zero-based start indices, row-major (`order="C"`) flattening, space-separated `start length` pairs, and `""` for an empty mask. Use `shape=(height, width)`. Its encoder/decoder passed five small round-trip checks, including image endpoints and a run crossing a row boundary. Verify against actual released masks/scorer when Q5 data is available; the decoder itself does not validate malformed input.
- Generate all masks, matches, and CSV content programmatically. No manual prediction corrections or hardcoded answers. Submitted runnable code must reproduce the selected CSV and work unchanged on unseen regions.

The classical-only restriction remains Q1–Q4; Lecture 5 p. 4 explicitly permits deep learning for Q5, and Lecture 5 p. 111 / Lecture 6 p. 70 recommend Cellpose/pretrained-model fine-tuning. These are supported Q5 options, not a requirement to use them or authorization to start training now. Preserve the user's class-based, minimal-code workflow. The general external-data policy remains unspecified; do not infer permission for arbitrary additional datasets from the pretrained-model advice. No neural-network dependencies are installed during setup.

## 6. Reviewed lecture concepts and use in solutions

Lecture sources (reviewed 2026-10-08; key equations, warping/graph diagrams, and project updates also visually checked):

| Source | Confirmed coverage |
| --- | --- |
| L3: `../Lectures/Slides/Computer Vision Fall 26 Lecture 3.pdf` (98 pages) | Least-squares fitting and outlier sensitivity, pp. 11–26; RANSAC sampling, inlier thresholds, trial counts, and refitting, pp. 27–52; scaling/rotation/shear, homogeneous coordinates, affine/projective transforms and composition order, pp. 71–96. |
| L4: `../Lectures/Slides/Fall'26 Computer-Vision Lecture 4.pdf` (118 pages) | Correspondence-based affine/homography fitting, pp. 24–41; forward/inverse warping and bilinear interpolation, pp. 42–56; threshold/edge segmentation, color-position features, superpixels, mean shift, graph affinities, normalized cuts and interactive graph cuts, pp. 60–104; segmentation metrics, pp. 105–110. |
| L5: `../Lectures/Slides/CV_2026_Lecture5.pdf` (117 pages) | Graph/spectral segmentation and evaluation, pp. 11–38; learned representations, activations, losses, gradient descent, regularization, overfitting and validation, pp. 39–108; Q5 segmentation/alignment advice, p. 111. |
| L6: `../Lectures/Slides/CV2026_lecture6.pdf` (77 pages; moved from Downloads) | CNN filters/channels, padding/stride/dilation, pooling and receptive fields, pp. 15–48; VGG, backpropagation, batch normalization and residual learning, pp. 49–56; training, initialization, weight decay, augmentation and transfer learning, pp. 59–66; Q5-specific advice, p. 70. |

**Use these concepts first, keeping code minimal.** The following are candidate applications, not implemented solutions. Cite the relevant lecture/page and codelab cell in each eventual notebook explanation; distinguish class code from any new adaptation.

| Question | Class-based starting point | Important limit |
| --- | --- | --- |
| Q1 | Fit boundary-line candidates robustly with RANSAC and refit on inliers (L3 pp. 27–52), using image-derived evidence to select the two outer edges. | A strong line fit alone cannot distinguish runway borders from internal markings; no fixed coordinates. |
| Q2 | Feature correspondences, RANSAC, planar homography, inverse warping and bilinear image interpolation (L3 pp. 23–26, 93–96; L4 pp. 34–56). | Composite only inside the computed cover mask; track transform direction and homogeneous-coordinate normalization. |
| Q3 | Infer pairwise overlap from robust matches, fit the simplest supported transforms, compose into one frame, then warp/blend explicitly (L3 pp. 54, 91; L4 pp. 26–56). | Detect/correct distortion from evidence; calculate canvas offsets, include all six views, and keep the one-call-stitcher ban. |
| Q4 | Use supplied foreground seeds with color/spatial affinities and interactive graph-cut concepts (L4 pp. 73–104; L5 pp. 12–27); consider mean shift/superpixels only if helpful. | Preserve all supplied foreground clicks. Unsupervised normalized cuts alone need not isolate the clicked animal; avoid a dense all-pixel affinity matrix. |
| Q5 | Transfer robust geometry from Q1–Q3 and segmentation/mask reasoning from Q4; use the specific lecture advice below. | No test clicks; automatic instance separation is new work. A book's homography and panorama blending are not a complete tissue-registration solution. |

**Q5 guidance (L5 p. 111; L6 p. 70):** normalize intensity by percentiles without discarding original 16-bit inputs; evaluate modalities separately and inspect mask boundaries/size/elongation, particularly bright ex-vivo fibers. If a learned approach is later chosen, start with the suggested Cellpose/pretrained baseline and modality-specific fine-tuning; consider cell-centered plus background patches, Dice/focal loss, and hard-negative patches when observed failures justify them. CNN architecture coverage is background, not a reason to build a large model from scratch. Use training/validation separation and regularization (L5 pp. 94–108; L6 pp. 61–66).

For matching, use coarse rotation/scale/translation alignment, robust affine fitting, then a few ICP iterations using close mutual centroid matches; inspect warped overlays and use distance-gated one-to-one assignment (e.g. Hungarian) while allowing unmatched cells. The slide's approximate image-size, rotation and match-fraction ranges are reported guidance, not hardcoded answers, guaranteed transform scales, or quotas. Keep the handout distinction between unverified cells and true negatives. Prefer simpler baselines before optional ensembling or added model complexity.

Reuse only needed helpers with explicit inputs/outputs. Do not import a notebook by executing all its cells or depend on previous answers/global state. Establish a small baseline, inspect failures, and justify each addition. For Q5 validation, prefer splits by mouse where feasible to reduce related-region leakage; report the limits of only three training mice. Measure both modality PQs and matching quality separately. Do not use hidden labels or tune predictions by test ID.

### 6.1 Reviewed codelab code and reuse limits

Sources below are in `../Lectures/CodeLabs/`. Cell numbers are **1-based positions including Markdown**, not execution counts. Reviewed code and explanatory cells; notebooks were not run end-to-end, and their downloads, installs, Drive mounts, and model-loading commands were not executed.

| Notebook | Reusable code and intended use |
| --- | --- |
| `CV_2026_Lecture_3.ipynb` | Cells 6/8: normalized implicit lines, perpendicular residuals, SVD total-least-squares refitting and `ransac_line` for Q1. Cell 11: trial-count reasoning. Cell 13: `apply_transform(points, H)` and composition helpers for Q2/Q3. Cell 21: similarity fitting/RANSAC as a simple geometric baseline for Q5. |
| `Lecture4_Fitting_Transformations_and_Segmentation_Codelab_5.ipynb` | Cell 10: SIFT, two-neighbor descriptor matching, ratio test, `findHomography(..., RANSAC)` and inlier visualization for Q2/Q3. Cells 12/14/15: homography equations, bilinear sampling, forward/inverse warp concepts. Cells 20/22/24/26/29/32: Otsu/Canny/components, k-means, SLIC, mean-shift filtering, feature affinities and normalized cuts for classical segmentation baselines. |
| `CV_2026_lecture5_codelab.ipynb` | Cells 6–9: color-position mean shift, connected-region cleanup and normalized cuts. Cells 12–14: MLP training/validation and error inspection as learning-workflow examples. Cells 18–20: Q5 RLE, CSV/JSON parsing and dataset paths; cells 25–27: instance masks, centroids, matched-cell plots and alignment overlays; cells 30–31: Cellpose inference. Cells 33–36 demonstrate MARCO and label-assisted Cellpose alignment, not an autonomous test pipeline. |
| `CV_Lecture_5_yolo_codelab.ipynb` | The local YOLO lab corresponding in topic to L6's YOLO session despite its filename. Cells 8–23: pretrained detection, multiscale grids, box coordinates, confidence thresholds and NMS; cells 24–30: model comparisons, Faster R-CNN and video inference. Useful for detection concepts/diagnostics, not a replacement for Q5 instance masks or permitted Q1–Q4 methods. Its hardcoded comparison tables are not measured project results. |

**Adapt carefully before copying:**

- L3's Scorpius demo supplies known correspondences and fixed pattern edges; reuse the solvers, not these answers or its thresholds. Its RANSAC helpers need a clear failure path for insufficient/fully degenerate candidates (`best` may remain `None`).
- L4 cell 12 references **undefined `fit_affine`, `reprojection_error`, and `apply_transform`** in the saved notebook. Supply only the needed helpers when implementing. L4 expects `apply_transform(T, points)`, opposite L3's argument order. Its homography solver uses unnormalized `A.T @ A`; prefer a numerically stable fit for real coordinates, sufficient nondegenerate matches, and an explicit failed-fit check. The demo warp assumes a channel axis; adapt it for 2-D masks/TIFFs and use nearest-neighbor sampling for discrete labels.
- L4's `pyrMeanShiftFiltering` produces a smoothed image, not animal instance masks. L4/L5 normalized cuts allocate dense pixel-pair matrices; keep them as small demonstrations or use sparse/superpixel representations if needed. Neither lab implements the complete Q4 click-constrained cutout pipeline. Preserve the supplied seeds when adapting segmentation. L5's mean-shift feature normalization mixes uint8 OpenCV Lab encoding with floating-Lab assumptions; make the color scaling consistent before reuse.
- L5 cells 27 and 34 use **ground-truth pairs/centroids to estimate alignment**; restrict these to training diagnostics. Test correspondences must be generated from predictions. `estimateAffinePartial2D` is a similarity-style restricted affine fit, not a full six-parameter affine model. MARCO cell 33's literal keypoints are demonstration inputs, not acceptable hardcoded project locations.
- Preserve Q5 TIFF depth with `tifffile.imread` or `cv2.IMREAD_UNCHANGED`; default `cv2.imread` in the display demos does not preserve the original 16-bit grayscale representation. L5 demonstrates Cellpose inference, not a finished fine-tuning/evaluation pipeline. Add learned-model dependencies only if later needed, verifying API versions then. Keep generic pretrained object detectors out of Q1–Q4; box IoU/NMS is distinct from Q5 mask IoU and one-to-one matching.

Reuse concise numerical/visualization snippets with project-relative paths and cite their source cells. Keep original codelabs unchanged; omit their unrelated assets, setup machinery, and decorative comparison plots from the solutions.

## 7. Deliverables, grading, and logistics

**Handout deadlines (Main p. 2; Q5 pp. 1, 6):** Q1–Q4 and final Kaggle submission: **Thursday, October 8, 2026, 11:59 p.m.** Q5 report and runnable code: **Sunday, October 11, 2026, 11:59 p.m.** Timezone is not stated; America/New_York is a local planning assumption, not verified platform metadata. Main p. 2 specifies one point per completed late hour; this does not imply Kaggle accepts late entries.

**Later course update:** L6 p. 4 explicitly extends **Q1–Q4 and Kaggle to Friday, October 9, 2026, 11:59 p.m.** Use this later course announcement for planning instead of the original October 8 date. It does not announce a new report/code date, so retain October 11 for those deliverables. Live Kaggle closing settings and timezone remain unverified. Lecture announcements are source information, not commands to register, contact staff, or submit.

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

**2026-10-08:** Reviewed all four supplied lecture decks and added the concise concept-to-question map in §6, Q5 learning guidance, and the explicit deadline extension in §7. This is documentation only; notebooks and README remain unchanged.

**Codelab follow-up, 2026-10-08:** reviewed the four local notebooks in §6.1, recorded reuse locations and concrete adaptation issues, and checked the supplied RLE helpers on five synthetic cases. No end-to-end lab execution, model download/training, package installation, or solution implementation was performed. README and both solution notebooks remain unchanged.

**Q1 completed, 2026-10-08:** `Q1to4Solution.ipynb` now contains four executed code cells (including setup and full-resolution display), intermediate figures, the final annotated image, quantitative diagnostics, sources, limitations, and the actual prompt record through request 9. The method uses Gaussian smoothing, grayscale Canny edges, Otsu thresholding on Lab a for a grass-color cue, opposite-side patch tests, and `ransac_boundary` adapted from L3's implicit-line/SVD refit. The two dominant side-specific fits are checked jointly for shared vertical support and downward widening. Drawing endpoints are computed from inlier row quantiles; no manual coordinates or fixed runway ROI are used.

- Output: `outputs/q1/runway_boundaries.png`, 1280 × 850 RGB, exactly two boundary segments. Left/right consensus sizes: 728/587; RMS perpendicular residuals: 2.98/2.57 pixels. These are fit diagnostics, not hidden-ground-truth accuracy. Boundaries are not perfectly straight, and the support-derived segments leave short distant end portions undrawn.
- Validation: all notebook code executed in order in a fresh kernel; no cell errors; final image and diagnostic figures visually inspected; all 14 original input hashes unchanged. Q2–Q4 placeholder cells, Q5 notebook, and README match their previous versions. No new dependencies, learned models, Kaggle predictions, or submissions.
- Reuse: `ransac_boundary` and its normalized point-to-line residual/SVD consensus pattern support later geometric reasoning; the grass-color cue and image-specific thresholds do not transfer directly to cell images. Q1 notebook locators are its section headings and helper name, not execution numbers.

Next: improve Q2 boundary alignment and Q4 mask quality before submission; develop Q5 when requested using the reviewed foundations. Keep README unchanged during development. Q5 input inventory and released-mask RLE checks are complete; the official scorer remains unavailable. Before final submission, fill identity placeholders, rerun notebooks cleanly, retain outputs, and verify current deadlines/settings and reproducibility.

**Q1 simplification:** user requested minimum code. Reduced 133 to 67 nonblank code lines, retaining class-based RANSAC, pair checks, essential figures and RMS diagnostics. Fresh-kernel rerun passed; output PNG is byte-identical. Prefer this concise style for subsequent questions.

**Q1 display follow-up:** original and annotated images are embedded directly at 1280 × 850, preserving native pixel data independently of viewer scaling. Q2 approach overview is authorized; its implementation remains pending.

**Q2 implementation, 2026-10-08:** user authorized a minimal solution. Added two code cells adapting L4 cell 10: SIFT on 2400-pixel working images, 0.75 ratio test, RANSAC homography in original coordinates, bilinear artwork warp and nearest-neighbor mask compositing. Output: `outputs/q2/replaced_cover.png`, 4284 × 5712. Fit: 84/286 inliers; median reprojection residual 2.83 scene pixels. The physical cover is curved, so narrow original-artwork strips remain along some boundaries; this is documented rather than claimed as pixel-perfect replacement. No manual coordinates. README and Q3–Q5 remain unchanged. Prompt record updated through request 10.

Q2 validation: full notebook executed in a fresh kernel without errors; saved PNG dimensions verified; unchanged pixels outside the mask asserted during execution. The warped cover was visually inspected.

**Q2 intermediate displays:** added ratio-test/inlier match comparisons (up to 40 best descriptor matches shown per panel), an automatically computed close-up of the projected outline, the compositing mask, and the masked warped artwork before the final result. Display selections do not alter estimation. Prompt record includes request 11.

**Q3 implementation, 2026-10-08:** user authorized the solution, intermediate plots, and native-resolution final image. Added explicit SIFT/ratio-test/RANSAC matching across all 15 pairs, an automatically rooted maximum-inlier overlap tree, composed homographies, cylindrical inverse mapping, and distance-weighted feather blending. Near-black exterior masks automatically identify inputs 1/4 as distorted; correspondence-derived mappings absorb their distortions. A flat-canvas trial crossed projective horizons in outer views, so the final cylinder is a documented project adaptation of inverse-warping concepts. Focal length is estimated from the sole reliable undistorted pair (2/5), assuming shared intrinsics and approximate rotational capture; this is a limitation, not ground-truth calibration.

- Reference: image 6. Inferred order: 5, 2, 4, 6, 1, 3. Tree: 3–1–6–4–2–5. Median tree-edge residuals: 0.13–0.25 px. Focal estimate: 425.55 px. Computed canvas: 1921 × 627, offset (1092, 257). No hard-coded order, coordinates, transforms, or high-level stitcher.
- `outputs/q3/panorama.png` is embedded at its full computed pixel dimensions. Includes input/distortion panels, overlap inlier matrix, cylindrical footprints, isolated corrected views, and overlap coverage. Black exterior regions preserve the complete observed field without cropping valid content. Feathering reduces seams; residual parallax/exposure differences and interpolation limits remain.
- Validation: full fresh-kernel execution passed; all six images contribute; native embedded dimensions verified; original 14 input hashes, Q1/Q2 result PNGs, README, and Q5 notebook unchanged. Prompt record updated through request 13. No new dependencies or submissions.

**Q4 baseline, 2026-10-08:** user authorized code, intermediate figures and RGBA outputs. Added a shared 640-pixel working-resolution GrabCut pipeline using the JSON clicks. Initialization uses distances scaled by click span and conservative outer/background assumptions. After observed RGB-only leakage, a second cut uses Lab L/a plus normalized click distance; the intersection reduces distant leakage. Keep seed-containing components, upsample binary alpha with nearest-neighbor interpolation, preserve exact original click pixels and original RGB values, and export one full-size RGBA PNG per instance under `outputs/q4/`. Display initialization, RGB mask, constrained mask, boundary, checkerboard and native PNGs. These are classical graph-cut/color-position adaptations of L4/L5, not a verbatim codelab pipeline.

**Quality limitation:** Q4 is not submission-ready. Lizard/bird masks retain bark; the leopard retains rocks; tails, legs and feather detail can be missed. Working-resolution masks do not imply native-resolution boundary accuracy. Foreground heuristics were compared by visual inspection without ground-truth masks or added annotations; no semantic accuracy score is claimed. Do not mark Q4 fully solved merely because execution, transparency or click checks pass. Q5 remains unimplemented and README remains frozen. Actual prompt record now includes requests 14–15.

Q4 validation: fresh-kernel execution and notebook schema passed; all three PNGs are native-size RGBA with binary alpha and unchanged original color channels; all nine clicks are opaque. Native embedded image dimensions also verified. All 14 input hashes, Q1–Q3 result PNGs, README and Q5 notebook remain unchanged. Final diagnostic figures were visually inspected; quality limitations above remain.

**Q4 refinement, 2026-10-09:** replaced the color-cut intersection with Lab + oriented Gabor texture averaged over SLIC regions, sparse appearance-weighted geodesic label propagation, and original-resolution GrabCut boundary refinement. Consulted [Szeliski, February 2010 draft](https://www.cs.jhu.edu/~ayuille/courses/Stat238-Winter10/SzeliskiBook_20100227_draft.pdf), §§5.4–5.5, printed pp. 295–301. This is a project adaptation of texture affinities/geodesic and graph-cut concepts, with one shared parameter set and only the supplied clicks. No new dependencies. More leopard face/rear and less owl/lizard bark are retained, but missing tails and rock/bark leakage remain; Q4 is still not submission-ready. Native refinement cannot correct locked-in coarse interior mistakes. Preserve these limitations and do not interpret file/seed checks as semantic accuracy. Prompt record includes requests 16–17.

Refinement validation: full notebook executed in a fresh kernel (28.5 seconds), with no cell errors and valid notebook schema. Visually inspected all three final diagnostic figures. Verified native-size embedded/exported RGBA PNGs, binary alpha, identical source colors and all nine opaque clicks; all 14 input hashes, Q1–Q3 output hashes, README and Q5 notebook remain unchanged. These checks establish reproducibility/file correctness, not semantic segmentation accuracy.

**Q5 authorization/access, 2026-10-09:** user now requests the competition solution at https://www.kaggle.com/competitions/testingpj-2 using **only traditional computer vision**, minimal code and a target score of 0.70. This overrides optional learned-method suggestions for this task. Read target as the handout's combined PQ/F1 score, not pixel accuracy; do not promise it. Data remains empty; official Kaggle data-list API returned HTTP 401, no local Kaggle credentials were found, no browser was available, and inspected Downloads archives contained no Q5 data. Requested a local dataset path/download. Q5 notebook now contains an executable dataset/schema check and codelab-derived RLE helpers; actual data verification, segmentation/matching implementation, validation and CSV generation remain pending. Do not reuse training ground-truth pairs for test-time registration. README and Q1–Q4 remain unchanged. Prompt record includes request 18.

**Q5 classical baseline, 2026-10-09:** implemented `cell_matching.py` (shared modality parameters, no sample-specific predictions), executable `CellMatchingSegmentation.ipynb`, six contract tests, and `Kaggle Submissions/classical_submission.csv` with all 29 official sample IDs. Pipeline: percentile normalization, DoG peaks, ex-vivo Hessian blob filtering, watershed, regional threshold/area checks, in-vivo one-pixel expansion; coarse/fine correlation registration, predicted-centroid RANSAC refinement and gated Hungarian assignment. No new dependencies or learned methods. Input bounds/dimensions/dtype, 20,409 masks and 1,139 verified pairs were checked; 154 Q5 input hashes are recorded in DATA_SOURCES.json. The released RLEs align visually with the codelab's zero-based row-major convention. Live rules/scorer remain unverified.

Parameters were selected on 29 regions from the two development mice and frozen before evaluation on 18 regions from `subject_db6b8b`. Development combined score: 0.1393. **Held-out local score: 0.1600; PQ in vivo 0.4089, PQ ex vivo 0.2309, matching F1 0.0000. The requested 0.70 target was NOT achieved.** Only 99/562 verified held-out pairs are mask-recoverable; with these fixed masks even perfect matching is bounded by approximately 0.310. This is an unsuccessful accuracy target, not a submission-ready competition solution or a leaderboard result. Preserve the honest measurements. Do not tune on this held-out mouse and continue calling it untouched; use a new nested/grouped validation design if further development uses its results.

Artifacts: per-region/summary metrics, development selection tables, example native-size label TIFFs and diagnostic figures under `outputs/q5/`; source code and notebook run from project-relative Data paths. Full notebook fresh-kernel execution passed (83.3 seconds). Six tests passed for RLE bounds/round trips, strict IoU, pair scoring, pooled F1, unmatched assignment and synthetic registration. No Kaggle submission/selection, code push, README update or Q1–Q4 modification. Prompt record includes request 19.

Final Q5 verification: standalone CLI regenerated a byte-identical candidate CSV; all 29 rows passed native-dimension RLE round trips, pair-reference/uniqueness and CSV schema checks. All 168 Q1–Q5 original file hashes remain unchanged. Notebook outputs are retained; no code cells have errors.

**Lecture 6 refinement, 2026-10-09 (requests 20–21):** user requested applying the supplied tips slide while keeping code/tokens minimal. The traditional-only constraint remains. Added post-segmentation major/minor-axis filtering (ratio ≤3), initialization that searches unequal x/y scales and shear as well as rotation/translation, and up to five mutual-neighbor ICP iterations with full-affine RANSAC (reject reflected/ill-conditioned fits). Added computed-footprint red/green warp overlays. Keep Hungarian matching with unmatched options. No neural fine-tuning, new dependencies or manual object locations.

New shape/gate choices use leave-one-mouse-out selection. Every fold chose ratio ≤3; matching gates/cutoffs were (8, 0.14), (10, 0.14), and (8, 0.12) for the three sorted mice. The base detector settings and method design reflect prior data inspection: call this **retrospective grouped development validation**, not a new independent test. Do not continue calling the old 18-region mouse untouched. All-region descriptive comparison: previous fixed baseline **0.1468**, new grouped result **0.1661**; PQ in vivo **0.3746**, PQ ex vivo **0.2005**, pooled matching F1 **0.0447** (previous 0.0092). Per-mouse combined scores: 0.1730, 0.1265, 0.1864. The 0.70 target remains unmet.

After grouped evaluation, all training data select a final gate of 8 pixels and correlation cutoff 0.14 for hidden-test inference. That fitted-to-training score (0.1685) is not the grouped estimate. Write the new candidate to `Kaggle Submissions/classical_lecture6.csv`; preserve the previous `classical_submission.csv`. Notebook contains grouped evaluation, diagnostic figures, source/limitation notes and actual prompts through request 21. Nine synthetic/contract tests now include affine shear/scale recovery, full-affine ICP and blank-image registration. Q1–Q4, original inputs and README remain unchanged. No Kaggle upload or leaderboard score.

Lecture 6 final verification: full fresh-kernel notebook run passed in 364.4 seconds; all outputs retained and grouped metrics reproduced exactly. Nine tests passed. The 29-row new CSV passed RLE/ID/uniqueness/schema checks and was byte-identical to an independent standalone CLI run. Previous CSV and all 168 source hashes remain unchanged. High/low-correlation warped-footprint overlays were visually inspected.

**Authorized U-Net experiment, 2026-10-09 (request 22):** the user explicitly requested a small CNN in a separate notebook. This authorizes neural methods **for this isolated Q5 experiment**, overriding the prior traditional-only constraint within that scope. Keep the classical pipeline/notebook/CSVs, Q1–Q4 and README unchanged. `SmallUNetExperiment.ipynb` trains separate two-level, 16/32/64-channel U-Nets from scratch on 29 regions from two mice; `subject_db6b8b` (18 regions) is excluded from weight training and used for final development evaluation. It was previously inspected during classical work, so do not call it a new untouched test. Fixed budget: 12 epochs × 60 batches × 8 patches per modality, 128-pixel patches, AdamW/cosine schedule, foreground plus eroded-interior BCE/Dice supervision, paired rotation/flip and intensity augmentation. No evaluation-based checkpoint or threshold selection. Native tiled inference and watershed recover instance masks; area quantiles use training masks only. Classical registration/matching is reused unchanged.

PyTorch 2.14.1 is installed only in the project environment and pinned in optional `requirements-unet.txt`; existing core requirements remain unchanged and pip check passed. Apple MPS is available. No pretrained weights or external dataset was downloaded. Artifacts are isolated under `outputs/q5/unet/`; no hidden-test CSV or Kaggle upload is part of this experiment. Record measured results after execution; a training-loss decrease is not evidence of 0.70 competition performance.
