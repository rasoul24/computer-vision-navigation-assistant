# Computer Vision Navigation Assistant

A computer vision prototype designed to assist blind and visually impaired pedestrians by identifying relevant objects and interpreting their surroundings.

## Features

- **Semantic segmentation:** SegFormer-based scene understanding.
- **Object detection:** DETR-based detection of relevant objects.
- **Trajectory analysis:** Cython-based processing for navigation-related geometry.
- **Audio feedback:** Generation of spoken descriptions and navigation cues.

## Project structure

- `Core/`: main processing pipeline.
- `Navigation/`: geometric analysis and Cython modules.
- `Outputs/`: visual and audio output components.
- `Data/`: sample inputs and test data.

## Requirements

- Python (use a version compatible with the project's dependencies).
- PyTorch and the required computer vision libraries.
- A C compiler compatible with the Python environment for building Cython extensions.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

Build the Cython extensions:

```bash
python setup.py build_ext --inplace
```

## Usage

Example:

```bash
python main.py vision image_test1
python main.py text image_test1
```

Ensure the sample image is available at the expected location under `Data/`.

## Limitations

This project is a prototype. Object detection and scene interpretation do not guarantee that a path is safe. Outputs should not be relied upon as the sole basis for real-world navigation.

## Author

Rasoul Islangueriev