# Face Recognition System

This repository contains a face recognition system implemented in Python. It uses the `face_recognition` library to encode faces from a training dataset and recognize faces in new images. This system is designed with a command-line interface for training, validating, and testing the face recognition model.

## Features

- **Training Mode**: Processes a training dataset to encode faces.
- **Validation Mode**: Validates the model against a separate dataset.
- **Testing Mode**: Tests the model's recognition capabilities on individual images.
- **Model Selection**: Supports choosing between 'hog' and 'cnn' models for face detection.

## Getting Started

### Prerequisites

- Python 3.x
- face_recognition library
- Additional libraries as listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recognize_faces.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd recognize_faces
   ```
3. Install packages
   ```bash
   pip install requirements.txt
   ```
### Directory Structure
   ```scss
   recognize_faces/
   ├── training/
   │   └── [individual folders with images]
   ├── validation/
   │   └── [images for validation]
   ├── output/
   │   └── encodings.pkl
   ├── detector.py
   ├── requirements.txt
   └── unknown.jpg (optional test image)
   ```

### Usage
1. Training the Model:
      Run the training process by using the `--train` flag. Ensure you have a `training` directory with images organized by person names!
      ```bash
      python detector.py --train
      ```
2. Validating the Model:
      Validate the model's performance using the `--validate` flag. Validation images should be placed in the `validation` directory.
      ```bash
      python detector.py --validate
      ```
3. Testing the Model:
      Test the model on an unknown image using the `--test` flag and providing an image path with `-f`.
      ```bash
      python detector.py --test -f path/to/your/image.jpg
      ```
