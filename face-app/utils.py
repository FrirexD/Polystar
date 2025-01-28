import cv2 as cv
from insightface.app import FaceAnalysis
import os

def format_number(number: int) -> str:
    """
    ### Converts an integer to a conventional string to search for images in celebA DB

    Args:
        number: The integer to convert

    Returns:
        formatted_number: Formatted string to according rules
    """

    # Convert the number to a string with leading zeros if necessary
    formatted_number = str(number).zfill(6)
    
    # Append the "jpg" extension
    return formatted_number + ".jpg"

def detect_and_draw_faces(input_image_path: str, output_directory: str, output_filename: str = "output.jpg") -> tuple:
    """
    ### Detects faces on an image, and returns a new image with box around detected faces

    Args:
        input_image_path: Path to image file to detect image
        output_directory: Directory in which to store modified image
        output_filename: Name of the output image.

    Returns:
        tuple: (image, output_path) The modified image and its full path with name
    """

    # Chargement de l'image d'entrée
    image = cv.imread(input_image_path)
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return None, None

    # Initialisation de l'application InsightFace
    app = FaceAnalysis(providers=["CPUExecutionProvider"], model = "buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Détection des visages
    faces = app.get(image)

    # Dessin des contours des visages sur l'image
    for face in faces:
        bbox = face.bbox.astype(int)  # Convertit les coordonnées en entiers
        cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Génération du chemin de sortie
    output_path = os.path.join(output_directory, output_filename)

    return (image, output_path)

def extract_embedding(image_path: str, app):
    """
    ### Gets the embedding of an image

    Args:
        image_path: Path of the image to get the embedding
        app: The FaceAnalysis object already initialized from InsightFace

    Returns:
        embedding object
    """
    # Read the image
    img = cv.imread(image_path)

    # Detect faces and extract embeddings
    faces = app.get(img)
    
    if len(faces) == 0:
        print("No face detected!")
        return None

    # Get the largest detected face if multiple
    face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    
    # Return the embedding (which is in 'embedding' field)
    return face.embedding