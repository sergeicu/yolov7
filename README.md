# YOLOv7 Bone Fracture Detection

YOLOv7-based bone fracture detection for pediatric wrist and elbow X-rays,
fine-tuned on institutional data from Boston Children's Hospital.

## Project Structure

- `yolov7/` -- Customized YOLOv7 training code with BCH-specific configs
- `YOLOv7-Bone-Fracture-Detection/` -- Forked reference implementation (GRAZPEDWRI-DX)
- `create_wrist_fracture_dataset/` -- Pipeline for wrist fracture dataset creation
- `create_elbow_fracture_dataset/` -- Pipeline for elbow fracture dataset creation
- `dicom-to-png/` -- DICOM to PNG conversion utilities

## Fracture Classes

- Distal radius fracture
- Distal ulna fracture
- Scaphoid fracture
- Ulna styloid fracture

## Data

Training data is institutional and not included in this repository.
The model was pre-trained on the [GRAZPEDWRI-DX](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)
public dataset and fine-tuned on BCH pediatric wrist/elbow X-rays.

## Requirements

See `yolov7/requirements.txt` for dependencies.

---

## PHI Notice

This repository has been sanitized. The following data types have been excluded:

- Patient MRN, names, and demographics (xlsx/csv files)
- Clinical radiology report text files
- DICOM images with embedded patient metadata
- PNG images with accession numbers in filenames
- LLM-processed report annotations
- Model weight files (.pt, .onnx)
- Training run outputs and logs

**Original path:** `/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/`

See `GIT_DO_NOT_COMMIT_LINKS/` (in the parent directory, outside this repo) for
symlinks to excluded data. See `GIT_DO_NOT_COMMIT_MAPPINGS/` for sanitization details.
