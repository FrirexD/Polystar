import insightface
import numpy as np
import cv2 as cv
from insightface.app import FaceAnalysis
from utils import detect_and_draw_faces, extract_embedding
from constants import *
import pickle
import os
from morphing import morph_images

def initialize_face_analyzer(model:str = 'buffalo_l') -> FaceAnalysis:
    """
    ### Initialize the InsightFace face analyzer in a global variable "app"
    
    Args:
        model: Name of the model to be used for init (default 'buffalo_l')
    """
    global app

    face_analyzer = FaceAnalysis(providers=[ "CPUExecutionProvider"], model = model)
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    app = face_analyzer

def calculate_similarity(embedding1, embedding2) -> float:
    """
    ### Calculate cosine similarity between two face embeddings
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        
    Returns:
        float: Similarity score between 0 and 1
    """
    embedding1 = embedding1.ravel()
    embedding2 = embedding2.ravel()
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity


def find_best_match(source_embedding, embedding_folder_path = PREPOC_DIR) -> tuple:
    """
    ### Compares source face with all faces of a folder and determines the best match
    
    Args:
        source_embedding: Embedding of source image
        matching_embeddings_path: Path to pk1 file of all embeddings to compare
        
    Returns:
        tuple: (path of best match, similarity score)
    """
    best_match_score = 0.0
    best_match_path = ""
    batch_counter = 1

    if source_embedding is None:
        print("Could not find face on source embedding")
        return "", best_match_score
    
    print("Finding best match...")

    # Get all the embeddings by batch
    for matching_embedding in os.listdir(embedding_folder_path):
        if matching_embedding.endswith("pk1"):
            file_path = os.path.join(embedding_folder_path, matching_embedding)

            # Load the embeddings and metadata from the pickle file
            with open(file_path, 'rb') as f:
                embeddings, metadata = pickle.load(f)
    
        # Store the embeddings in a variable
        embeddings_array = np.array(embeddings)

        # Calculate similarity for each embedding
        for i, (embedding, path) in enumerate (zip(embeddings_array, metadata)):
            score = calculate_similarity(source_embedding, embedding)
            
            # Find index and score of best matching image
            if(score > best_match_score):
                best_match_score = score
                best_match_path = path
            i+=1
        
        print(f"Treated batch {batch_counter}")
        batch_counter += 1
    return best_match_path, best_match_score

def visualize_match(image_path1 :str = CELEBA_DIR+"000001.jpg", image_path2 : str = CELEBA_DIR+"000002.jpg", similarity : float = 0.) -> np.ndarray:
    """
    ### Create a visualization of the face matching result
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        similarity: Similarity score
    """
    img1 = cv.imread(image_path1)
    img2 = cv.imread(image_path2)
    
    if(img1 is None or img2 is None):
        print("Error reading images for visualization")
        return None
    
    # Resize images to same height
    height = 300
    ratio1 = height / img1.shape[0]
    ratio2 = height / img2.shape[0]
    img1 = cv.resize(img1, (int(img1.shape[1] * ratio1), height))
    img2 = cv.resize(img2, (int(img2.shape[1] * ratio2), height))
    
    # Create result visualization
    result = np.hstack([img1, img2])

    # BGR color channels
    color = (215, 193, 0)
    text = f"Similarity: {min(100,100*(similarity+0.5)):.0f}%"
    
    cv.putText(result, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return result

    
def main(source_image_path):
    # Initialize the app
    initialize_face_analyzer()

    # Reading source image
    source_embedding = extract_embedding(source_image_path, app)

    detected_faces_img , _= detect_and_draw_faces(source_image_path,OUTPUT_DIR, "detected_faces.jpg")
    cv.imwrite(OUTPUT_DIR+"detected_faces.jpg",detected_faces_img)

    comparing_image_path = SAMPLES_DIR+"Renan1.JPG"
    compare_embedding = extract_embedding(comparing_image_path, app)
    compared_images = visualize_match(source_image_path, comparing_image_path, calculate_similarity(source_embedding, compare_embedding))
    cv.imwrite(OUTPUT_DIR+"matching_samples.jpg",compared_images)

    # Find best score of matching star with source
    matching_image_path, score = find_best_match(source_embedding)
    result = {"path" : matching_image_path , "score" : score}

    # Visualize score with images side by side
    score_img = visualize_match(source_image_path, matching_image_path, score)
    # Store image in output folder
    if score_img is None:
        print("Error while writing similarity file image")
    cv.imwrite(OUTPUT_DIR+"best_match.jpg", score_img)
    print(f"Result saved to /app/{OUTPUT_DIR}best_match.jpg")


    # morphed_image = morph_images(source_image_path, matching_image_path, 0.5, app)
    # cv.imwrite(OUTPUT_DIR+"morphed.jpg", morphed_image)
    # print(f"Result saved to /app/{OUTPUT_DIR }morphed.jpg")
    
    return result