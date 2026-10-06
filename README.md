# Computer Vision — Project 2

NYU Tandon · CS-GY 6643 · Fall 2026

Feature matching, geometric transformations, and image segmentation.

This workspace is initialized for development. The notebooks currently contain Markdown headings and descriptions only. Q1–Q4 will be solved using the lecture slides and codelabs supplied later; their relevant methods will then support the Q5 cell matching and segmentation solution.

## Layout

- `Q1to4Solution.ipynb`: runway boundaries, book-cover replacement, panorama stitching, and click-based segmentation.
- `CellMatchingSegmentation.ipynb`: future Q5 segmentation and correspondence solution.
- `Q1 Data/` through `Q4 Data/`: official assignment inputs.
- `Data/`: reserved for Q5 data.
- `outputs/q1/` through `outputs/q5/`: generated results.
- `Kaggle Submissions/`: future prediction CSVs.
- `AGENTS.md`: requirements and development workflow.
- `DATA_SOURCES.json`: download sources and input inventory.
- `AI_USAGE.md`: AI-assistance disclosure and prompt record.

## Environment

Python 3.12 with a project-local virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To recreate the environment, first run `python3.12 -m venv .venv`. Select `.venv/bin/python` or **Python (CV Project 2)** as the notebook kernel in VS Code. Optional PDF-inspection and dataset-download tools are in `requirements-setup.txt`.

**Documentation policy:** this is the initial README. Leave it unchanged until the complete solution is finished and ready to submit; track interim work in `AGENTS.md`.
