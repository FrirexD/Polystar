import os
from typing import List
import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis
import pickle
from constants import *
import shutil
import random
from tqdm import tqdm
from pathlib import Path

def initialize_face_analyzer(device: str = 'cuda:0') -> FaceAnalysis:
    """
    ### Initialize the InsightFace face analyzer in a global variable "app"
    """
    face_analyzer = FaceAnalysis(model = "buffalo_l")
    face_analyzer.prepare(ctx_id=0 if device.startswith('cuda') else -1)  # 0 for GPU, -1 for CPU
    return face_analyzer

def copy_random_images(source_folder: str = DATA_DIR+"img_align_celeba", destination_folders: List[str] = DATA_DIR+"celebA", max_images: int = 1000):
    """
    ### Copies up to `max_images` random images from the source folder to the destination folder with the tmp folder.
    ### Clears the destination folder before copying.

    Args:
        source_folder: Path to the source folder containing images.
        destination_folder: Path to the destination folder where images will be copied.
        max_images: Maximum number of images to copy.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folders[0]):
        os.makedirs(destination_folders[0])
    else:
        for destination_folder in destination_folders:
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
            print(f"Cleared files of folder {destination_folder}")

    # List all valid image files in the source folder
    image_files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Copy up to `max_images` files
    for destination_folder in destination_folders:
        for i, image_file in enumerate(image_files):
            if i >= max_images:
                break
            source_path = os.path.join(source_folder, image_file)
            destination_path = os.path.join(destination_folder, image_file)
            shutil.copy2(source_path, destination_path)
            print(f"Copied {image_file} to {destination_folder+'/'+image_file}")

    print("Finished copying images")

def preprocess_folder(folder_path: str = CELEBA_DIR, embeddings_dir: str = PREPOC_DIR, batch_size: int = DEFAULT_BATCH_SIZE, max_image : int = 1000, app : FaceAnalysis = None):
    """
    ### Preprocess all images in the folder to extract embeddings and store them in file.
    ### The size of the file depends on the size of the batch

    Args:
        folder_path: Path to the folder containing images.
        embeddings_dir: Directory in which to save all embeddings.
        batch_size: Number of images to process in a batch.
        max_image: Number of images to process in total
    """


    if not os.path.exists(folder_path):
        print(f"ERROR : {folder_path} does not exist.")
        return
    
    # Clear the preprocessed directory
    for item in os.listdir(embeddings_dir):
        item_path = os.path.join(embeddings_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove file

    # Going through the folder batch by batch
    for i in range(0, (max_image//batch_size)):
        embeddings, metadata = process_batch(folder_path, batch_size, app)
        # Save embeddings and metadata
        with open(embeddings_dir+"embeddings"+str(i+1)+".pk1", 'wb') as f:
            pickle.dump((np.array(embeddings), metadata), f)
            print(f"Batch {i+1} saved to {embeddings_dir+str(i+1)}.pk1")

    # Get the remaining images of the folder if it's a smaller batch
    embeddings, metadata = process_batch(folder_path, (max_image%batch_size), app)
    with open(embeddings_dir+"embeddings"+str(max_image//batch_size)+".pk1", 'wb') as f:
        pickle.dump((np.array(embeddings), metadata), f)
        print(f"Batch {max_image//batch_size} saved to {embeddings_dir+str(max_image//batch_size)}.pk1")

    print("Preprocessing done !")


def process_batch(folder_path: str = DATA_DIR+"tmp/", batch_size: int = DEFAULT_BATCH_SIZE, app : FaceAnalysis = None):
    """
    ### Gets embeddings and metadata of images in a temporary folder by batch.

    #### Note : All images from the temporary folder are deleted

    Args:
        folder_path: The path of the temporary folder in which are stored images
        batch_size: The amount of images to get embeddings.
        app: FaceAnalysis component app.
    """
    embeddings = []
    metadata = []
    
    image_names = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.name.lower().endswith(("png","jpg","jpeg")):  # Check if it's an image
                image_names.append(entry.name)
            if len(image_names) >= batch_size:
                break  # Stops when batch of image is stored
        
    # Process images in batches
    for image_name in image_names:

        # Get the embedding of current face
        embedding, _ = get_face_embedding(app, DATA_DIR+"tmp/"+image_name)
        if embedding is not None:
            embeddings.append(embedding)
            metadata.append(CELEBA_DIR+image_name)
        else:
            print(f"Could not find any embedding for file {image_name}")
        Path(DATA_DIR+"tmp/" + image_name).unlink()
        # os.remove(CELEBA_DIR+image_name) # Once treated, delete the file
        print(f"{DATA_DIR+image_name} succesfully deleted from tmp folder")
        

    return embeddings, metadata


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

def preprocess_folder_gpu(
    folder_path: str,
    embedding_file: str,
    batch_size: int,
    app: FaceAnalysis,
    device: str = 'cuda:0'
):
    """
    Preprocess all images in the folder using GPU to extract embeddings.

    Args:
        folder_path: Path to the folder containing images.
        embedding_file: Path to save the embeddings and metadata.
        batch_size: Number of images to process in a batch.
        app: Pre-initialized FaceAnalysis object with GPU support.
        device: GPU device to use for processing.
    """
    # Collect all image files
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"Found {len(image_files)} images in {folder_path}")
    embeddings = []
    metadata = []

    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch = image_files[i:i + batch_size]
        batch_embeddings, batch_metadata = process_images_gpu(batch, app, device)
        
        # Extend our results
        embeddings.extend(batch_embeddings)
        metadata.extend(batch_metadata)
        
        # Print progress information
        print(f"Processed batch {i//batch_size + 1}/{len(image_files)//batch_size + 1}")
        print(f"Current total embeddings: {len(embeddings)}")

    # Verify we have results before saving
    if len(embeddings) == 0:
        print("Warning: No embeddings were generated!")
        return

    # Convert embeddings to numpy array and save
    try:
        embeddings_array = np.array(embeddings)
        print(f"Final embeddings shape: {embeddings_array.shape}")
        
        with open(embedding_file, 'wb') as f:
            pickle.dump((embeddings_array, metadata), f)
        
        print(f"Successfully saved {len(embeddings)} embeddings to {embedding_file}")
        
        # Verify the save
        with open(embedding_file, 'rb') as f:
            loaded_emb, loaded_meta = pickle.load(f)
            print(f"Verified save: loaded {len(loaded_emb)} embeddings")
            
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def process_images_gpu(batch, app: FaceAnalysis, device: str):
    embeddings = []
    metadata = []

    for image_path in batch:
        img = cv.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Process the image on the GPU
        try:
            faces = app.get(img)  # Process one image at a time
            if faces:
                face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                embeddings.append(face.embedding)
                metadata.append(image_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    return embeddings, metadata

def chunked(iterable, size):
    """
    Yield successive n-sized chunks from the iterable.

    Args:
        iterable: Iterable to chunk.
        size: Chunk size.

    Yields:
        Chunks of the iterable.
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# Run the function
max_images = 10000
copy_random_images(DATA_DIR+"img_align_celeba", [DATA_DIR+"tmp/", DATA_DIR+"celebA/"] ,max_images=max_images)
app = initialize_face_analyzer()
preprocess_folder(DATA_DIR+"tmp/", PREPOC_DIR, DEFAULT_BATCH_SIZE, max_images, app)
#preprocess_folder_gpu(CELEBA_DIR, PREPOC_DIR+"embeddings.pk1",DEFAULT_BATCH_SIZE,app,"cuda:0")