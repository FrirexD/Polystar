import os
import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis
from zipfile import ZipFile
import pickle
from progress.bar import Bar
from constants import *
import shutil
import random

def initialize_face_analyzer() -> FaceAnalysis:
    """
    ### Initialize the InsightFace face analyzer in a global variable "app"
    """
    face_analyzer = FaceAnalysis()
    # Use 0 for GPU, -1 for CPU
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer


def copy_random_images(source_folder: str = DATA_DIR+"img_align_celeba", destination_folder: str = DATA_DIR+"celebA", max_images: int = 1000):
    """
    Copies up to `max_images` random images from the source folder to the destination folder.
    Clears the destination folder before copying.

    Args:
        source_folder: Path to the source folder containing images.
        destination_folder: Path to the destination folder where images will be copied.
        max_images: Maximum number of images to copy.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    else:
        # Clear the destination folder
        for filename in os.listdir(destination_folder):
            file_path = os.path.join(destination_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # List all valid image files in the source folder
    image_files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Shuffle the list of image files
    random.shuffle(image_files)

    print("Copying random images...")
    # Copy up to `max_images` files
    for i, image_file in enumerate(image_files):
        if i >= max_images:
            break
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy2(source_path, destination_path)
        print(f"Copied {image_file} to {destination_folder+'/'+image_file}")

def preprocess_folder(folder_path: str = CELEBA_DIR, embedding_file: str = PREPOC_DIR, batch_size: int = DEFAULT_BATCH_SIZE, app : FaceAnalysis = None):
    """
    Preprocess all images in the folder to extract embeddings and store them in file.

    Args:
        folder_path: Path to the folder containing images.
        embedding_file: Path to save the embeddings and metadata.
        batch_size: Number of images to process in a batch.
    """

    embeddings = []
    metadata = []

    # List all valid image files in the folder
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Initialize the progress bar
    

    print("Getting face embeddings...")
    # Process images in batches
    bar = Bar('Processing Images', max=len(image_files))
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        
        for image_path in batch:
            embedding, _ = get_face_embedding(app, image_path)
            if embedding is not None:
                embeddings.append(embedding)
                metadata.append(image_path)
            
            # Update the progress bar
            bar.next()

    # Finish the progress bar
    bar.finish()

    # Save embeddings and metadata
    with open(embedding_file, 'wb') as f:
        pickle.dump((np.array(embeddings), metadata), f)
    print(f"Embeddings saved to {embedding_file}")

def get_face_embedding(face_analyzer : FaceAnalysis, image_path : str = CELEBA_DIR+"000001.jpg") -> tuple:
    """
    ### Extract face embedding from an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (face embedding, face bbox) or (None, None) if no face detected
    """

   

    img = cv.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None, None
    
    # BGR to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_analyzer.get(img)
    
    if not faces:
        print(f"No face detected in: {image_path}")
        return None, None
    
    # Get the largest face if multiple faces are detected
    face = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]
    return face.embedding, face.bbox

# Run the function
copy_random_images(max_images=200)
app = initialize_face_analyzer()
preprocess_folder(CELEBA_DIR, PREPOC_DIR+"embeddings.pk1", DEFAULT_BATCH_SIZE, app)
