# Import necessary libraries
import argparse
import math
import os
import pickle
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np

import cv2
import face_recognition

# Define the path for storing face encodings
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)

# Add argument for specifying the file path for testing
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
args = parser.parse_args()

# Create necessary directories if they do not exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


# Function to encode faces from the training dataset
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []  # List to store names of individuals
    encodings = []  # List to store face encodings

    # Get all image filepaths from the training directory
    image_filepaths = list(Path("training").glob("*/*"))
    total_images = len(image_filepaths)  # Total number of images

    print("[INFO] quantifying faces...")

    for idx, filepath in enumerate(image_filepaths, start=1):
        print(f"[INFO] processing image {idx}/{total_images}")
        name = filepath.parent.name  # Extract the name from the directory structure
        image = face_recognition.load_image_file(filepath)  # Load the image
        face_locations = face_recognition.face_locations(image, model=model)  # Detect faces in the image
        face_encodings = face_recognition.face_encodings(image, face_locations)  # Encode the detected faces

        # Append the name and encoding for each face found
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # Save the names and encodings as a dictionary
    name_encodings = {"names": names, "encodings": encodings}

    # Write the encodings to a file
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    print("[INFO] serializing encodings...")


# Helper function to recognize a single face
def _recognize_face(unknown_encoding, loaded_encodings):
    # Compare the unknown face with known encodings
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    # Count the number of matches for each name
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

    # Return the name with the highest number of votes (most likely match)
    if votes:
        return votes.most_common(1)[0][0]


# Function to recognize faces in an image
def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    try:
        # Try to load the input image
        input_image = face_recognition.load_image_file(image_location)
    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"Error: The file '{image_location}' was not found.")
        return
    # Load the saved encodings
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Load the input image and detect faces
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    # Recognize each face found in the input image
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        print(name, bounding_box)


# Function to validate face recognition on a set of images
def validate(model: str = "hog"):
    """
    This function iterates through all files in the 'validation' directory
    and uses the 'recognize_faces' function to perform face recognition on each file.
    It allows testing the face recognition system on a separate dataset (validation dataset).

    Args:
    model (str): The model used for face detection. Defaults to "hog".
                 Other options can be "cnn", which might be more accurate but is slower.
    """
    # Iterate over all files in the 'validation' directory and its subdirectories
    for filepath in Path("validation").rglob("*"):
        # Check if the current path is a file
        if filepath.is_file():
            # Call the recognize_faces function on the file
            # 'filepath.absolute()' is used to get the absolute path of the file
            recognize_faces(image_location=str(filepath.absolute()), model=model)


def estimate_face_orientation(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    left_eye_center = tuple(map(lambda x: int(sum(x) / len(x)), zip(*left_eye)))
    right_eye_center = tuple(map(lambda x: int(sum(x) / len(x)), zip(*right_eye)))

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = math.atan2(dy, dx) * 180 / math.pi
    return angle


def process_image(image_path):
    image = face_recognition.load_image_file(str(image_path))
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        angle = estimate_face_orientation(face_landmarks)
        if not -30 <= angle <= 30:
            return image_path
    return None


def get_face_detections_dnn(image_path, prototxt, caffemodel):
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at path: {image_path}")

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_regions = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_regions.append(box.astype("int").tolist())
        return image_path, face_regions

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, []


def encode_faces(image_path, face_regions):
    image = face_recognition.load_image_file(image_path)
    encodings = []
    for x1, y1, x2, y2 in face_regions:
        face_image = image[y1:y2, x1:x2]
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            encodings.append(face_encodings[0])
    return image_path, encodings


def find_duplicates(face_encodings, threshold=0.4):
    duplicates = []
    for path1, encodings1 in face_encodings.items():
        for path2, encodings2 in face_encodings.items():
            if path1 != path2:
                for encoding1 in encodings1:
                    for encoding2 in encodings2:
                        distance = face_recognition.face_distance([encoding1], encoding2)
                        if distance < threshold:
                            duplicates.append((path1, path2))
    return duplicates


def process_folder_parallel(folder_path, prototxt, caffemodel):
    start_time = time.time()
    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))

    face_data = {}  # Store face encodings for each image
    images_without_faces_count = 0  # Counter for images without faces

    with Pool(cpu_count()) as pool:
        # Get face regions
        face_regions_results = pool.starmap(
            get_face_detections_dnn, [(str(image_path), prototxt, caffemodel) for image_path in image_paths]
        )

        # Get face encodings and count images without faces
        for image_path, regions in face_regions_results:
            if regions:  # Proceed only if faces are detected
                _, encodings = encode_faces(image_path, regions)
                face_data[image_path] = encodings
            else:
                images_without_faces_count += 1  # Increment counter if no faces are detected

    # Find duplicates
    duplicates = find_duplicates(face_data, threshold=0.3)

    end_time = time.time()
    return len(duplicates), images_without_faces_count, end_time - start_time


# Specify the correct paths to your .prototxt and .caffemodel files
prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"

# Process the folders
folders = ["test 100", "test 1000", "test 5000", "test 13000"]
for folder in folders:
    duplicate_count, no_face_count, duration = process_folder_parallel(folder, prototxt, caffemodel)
    print(
        f"Folder {folder}: Found {duplicate_count} potential duplicates and {no_face_count} images without faces in {duration:.2f} seconds"
    )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
