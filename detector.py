# Import necessary libraries
import pickle
from collections import Counter
from pathlib import Path

import face_recognition

# Define the path for storing face encodings
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Create necessary directories if they do not exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


# Function to encode faces from the training dataset
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []  # List to store names of individuals
    encodings = []  # List to store face encodings

    # Loop through each file in the training directory
    for filepath in Path("training").glob("*/*"):
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


# Example usage
# encode_known_faces()
recognize_faces("unknown.jpg")
