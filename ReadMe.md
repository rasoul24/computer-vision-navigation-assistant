# Computer Vision Navigation Assistant

**An AI-powered prototype for environmental perception and navigation assistance for blind and visually impaired pedestrians.**

The project combines semantic segmentation, object detection and geometric trajectory analysis to interpret a pedestrian's surroundings and generate navigation-related feedback.

> **Project status:** Research prototype — not intended for standalone real-world navigation.

## Overview

The system processes visual input to identify relevant elements in the environment and generate scene descriptions or navigation cues.

<!-- Replace this placeholder with a screenshot or GIF of your system -->
<p align="center">
  <em>Demo coming soon — add a screenshot or GIF of the perception pipeline here.</em>
</p>

## Key Features

| Component | Description |
|---|---|
| **Semantic Segmentation** | SegFormer-based scene understanding |
| **Object Detection** | DETR-based detection of relevant objects |
| **Trajectory Analysis** | Geometric processing implemented partly in Cython |
| **Audio Feedback** | Spoken scene descriptions and navigation-related cues |

## Architecture

```text
Camera / Image
      |
      v
Visual Perception
  ├── Semantic Segmentation (SegFormer)
  └── Object Detection (DETR)
      |
      v
Geometric Analysis
        (Cython)
      |
      v
Scene Interpretation
      |
      v
Text / Audio Feedback
```

## Project Structure

```text
.
├── Core/                 # Main processing pipeline
├── Navigation/           # Trajectory and geometric processing
├── Outputs/              # Text and audio output
├── Data/                 # Sample inputs and test data
├── Segmentation_Model/   # Segmentation model configuration and weights
├── Vision_Process/       # Visual processing components
├── main.py               # Application entry point
├── setup.py              # Cython extension build configuration
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container configuration
```

## Getting Started

### Prerequisites

- Python version compatible with the project dependencies
- A C/C++ compiler supported by your Python environment
- Model weights required by the perception pipeline
- PyTorch and the project's computer vision dependencies

### 1. Clone the repository

```bash
git clone https://github.com/rasoul24/computer-vision-navigation-assistant.git
cd computer-vision-navigation-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in **Windows Git Bash**:

```bash
source .venv/Scripts/activate
```

On Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Build the Cython extensions

```bash
python setup.py build_ext --inplace
```

### 5. Configure the models and input data

Download or place the required model weights in the locations expected by the application. Make sure the sample image exists under `Data/` and that its filename matches the one used in the command.

### 6. Run the application

Example commands:

```bash
python main.py vision image_test1
python main.py text image_test1

python main.py video video_demo
```

The exact inputs and available commands depend on the application's current implementation.

## Models and Dependencies

This project uses SegFormer for semantic segmentation and DETR for object detection.

Large model-weight files may be hosted separately from the source code. Follow the repository's model download instructions and place the files in the expected directories before running inference.

## Limitations and Safety

This project is an experimental prototype. Object detection, semantic segmentation and generated navigation cues can be inaccurate or incomplete. The system does not guarantee that a detected path is safe.

**Do not rely on this prototype as the sole means of navigation in real-world environments.**

## Author

**Rasoul Islangueriev**

[GitHub](https://github.com/rasoul24)