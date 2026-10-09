# AI Assistance Record — Project 2

## Tools and scope

During initial setup on 2026-10-06, OpenAI Codex assisted with reading the previous project's workflow and both Project 2 handouts, preparing project instructions, downloading official Q1–Q4 data, initializing the Python environment, and creating Markdown-only notebook scaffolds. No question was solved during setup. Later source review and Q1 implementation are recorded chronologically below.

Keep actual user prompts in chronological order. Add subsequent assistance here and include the required disclosure/prompt chain in final deliverables. Do not reconstruct unavailable prompts or include hidden model reasoning.

## Request 1 — 2026-10-06

The user referenced `../Project1/AGENTS.md` and attached a screenshot of the previous project's directory structure. The request text was:

> Go through the AGENTS.md file for the previous project of this course, we will be following similar workflow for this project as well. Understand the requirements for this project by going through the project material present in the two pdfs present in the project root. Create a AGENTS.md file similar to one uploaded from the previous project with emaphasis on simple and minila code, we will follow the similar project structure as the project 1. For this task create the directories and download the data for Q1-4 we will solve this in subsequent tasks and the solutions and concepts used in it will the foundation upon which will we solve the kaggle competition problem. Also initialize the virtual python environment and install the required libraries. Add basic README.md we wont update it  until after the project solution is finished and ready to submit. Just create the solution notebooks and just add Heading markdown cells with boilerplate information about what it contains don't add code yet. In next tasks I will provide the lecture slides and the codelab notbooks the concepts and code from will we will use to solve this assignment.

## Request 2 — 2026-10-06

> Commit and push the repository to [https://github.com/rohitfarfad/-CV-Cell-Matching-x-Segmentation.git](https://github.com/rohitfarfad/-CV-Cell-Matching-x-Segmentation.git)

Assistance: inspect the local and remote Git state, prepare the initial setup commit, and push it to the user-specified repository. Existing dataset, environment, and generated-output exclusions are retained.

## Request 3 — 2026-10-08

The user supplied Lecture 3, 4, 5, and 6 PDFs; their exact paths are recorded in `AGENTS.md` §6.

> Go through this slides and understand the concepts covered  in the lectures. Update the AGENTS.md briefly to use these concepts in the solutions

Assistance: review the four slide decks, map their concepts to Q1–Q5, and record the later deadline extension and Q5-specific guidance. No solution code or notebook changes.

## Request 4 — 2026-10-08

The message listed the four lecture PDFs; the corresponding local code notebooks were found in `../Lectures/CodeLabs/` and are identified in `AGENTS.md` §6.1.

> Go through these codelabs and understand the code and concepts discussed. We will use this code to solve the porblems.

Assistance: review four relevant local codelabs, record reusable snippets and adaptation issues, and check the supplied RLE functions on five synthetic masks. The solution notebooks and README were not changed; no models were downloaded or trained.

## Request 5 — 2026-10-08

> Using the concepts covered in the class, how do you solve the Question 1, give the overview of the approach before we code it

Assistance: explain a classical approach combining edge detection, pavement–grass color evidence, RANSAC line fitting, and pair geometry checks. No code was added in that turn.

## Request 6 — 2026-10-08

> great, let's code it in the notebook

Assistance: implement Q1 in `Q1to4Solution.ipynb`, adapting class RANSAC/SVD, Canny, thresholding and color-feature concepts; inspect the programmatically annotated result and retain diagnostics and explanations. Q2–Q5 and README remain unchanged. No hidden ground truth, manual coordinates, or learned models are used.

Validation: the notebook executed in a fresh kernel without cell errors, saved the annotated image and visible figures, and passed original-input hash and unchanged-placeholder checks. Residuals describe the fitted edge consensus, not hidden-ground-truth accuracy.

## Request 7 — 2026-10-08

> Use minimum code

## Request 8 — 2026-10-08

> Continue, use minimal tokens

Assistance: reduced Q1 from 133 to 67 nonblank code lines, removed redundant plots/table dependencies, and reran the notebook. The generated boundary image is byte-identical to the previous result.

## Request 9 — 2026-10-08

> For question1, add the final solution images in origianl image resolution and after that give the overview of the solution for question 2 using concepts covered in the class

Assistance: embed the original and annotated Q1 images at native 1280 × 850 resolution; outline SIFT matching, RANSAC homography estimation, perspective warping and mask-based compositing for Q2. No Q2 solution code added.

## Request 10 — 2026-10-08

> Great, add the solution for question 2 in the notebook with minimum code and optimizing the token usage

Assistance: implement a concise SIFT/ratio-test/RANSAC homography solution, warp and composite the replacement at original scene resolution, inspect the result, and document the small boundary mismatch caused by the curved cover. No manual coordinates or learned methods; README and Q3–Q5 remain unchanged.

## Request 11 — 2026-10-08

> You can add the code for question 2 to show important intermediate steps of the solution

Assistance: add concise visualizations of ratio-test matches, RANSAC inliers, the projected cover boundary, compositing mask and warped artwork. Display subsets/crops are labeled and computed automatically; fitting and final compositing remain unchanged.

## Request 12 — 2026-10-08

> Great, similary give the overview of solution for question 3 before coding

Assistance: inspect all six inputs and outline feature-based distortion correction, overlap inference, transform composition, canvas calculation and blending. No solution code added in that turn.

## Request 13 — 2026-10-08

> Great, let's add the solution to the notebook, keep the code minimum but include the important intermediate steps to the final solution, add final solution image in original dimentions

Assistance: implement Q3 with automatic padding-based distortion identification, SIFT/RANSAC overlap estimation, transform composition, image-derived cylindrical projection and feather blending. The cylinder addresses an observed projective-horizon failure in a flat-canvas trial. Include intermediate figures and the full-resolution panorama; document calibration assumptions and residual limitations. No hard-coded transforms/order, manual alignment, or high-level stitcher.

## Request 14 — 2026-10-08

> similarly give the overview of solution for question 4

Assistance: inspect the wildlife images and JSON clicks, and outline click-initialized GrabCut, conservative component cleanup, intermediate displays and original-resolution RGBA export.

## Request 15 — 2026-10-08

> Great let's code this into the notebook like previous questions

Assistance: implement classical click-guided GrabCut, compare initialization/spatial and edge-aware alternatives on the supplied images, and retain a common RGB-plus-spatial-cut baseline. Add diagnostic figures, original-resolution RGBA outputs, and explicit caveats about retained background and missing animal parts. No extra manual annotations or learned semantic models; image quality remains insufficient for a claim of complete animal isolation.
