import insightface
import numpy as np
import cv2 as cv
from insightface.app import FaceAnalysis
from utils import detect_and_draw_faces, extract_embedding, format_number
from constants import *
import pickle


def initialize_face_analyzer(model:str = 'buffalo_l') -> FaceAnalysis:
    """
    ### Initialize the InsightFace face analyzer in a global variable "app"
    
    Args:
        model: Name of the model to be used for init (default 'buffalo_l')
    """
    global app
    face_analyzer = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"], model = model)
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



def find_best_match(source_embedding, matching_embeddings_path = PREPOC_DIR+"embeddings.pk1") -> tuple:
    """
    ### Compare source face with all faces of a folder and determine the best match
    
    Args:
        source_embedding: Embedding of source image
        matching_embeddings_path: Path to pk1 file of all embeddings to compare
        
    Returns:
        tuple: (index of image, similarity score)
    """
    best_match = 0.0
    i = 0
    index = 0

    if source_embedding is None:
        print("Could not find face on source embedding")
        return 0, best_match
    
    print("Finding best match...")
    # Load the embeddings and metadata from the pickle file
    with open(matching_embeddings_path, 'rb') as f:
        embeddings, metadata = pickle.load(f)

    # Store the embeddings in a variable
    embeddings_array = np.array(embeddings)
    
    # Calculate similarity
    for i, (embedding, path) in enumerate (zip(embeddings_array, metadata)):
        similarity = calculate_similarity(source_embedding, embedding)
        print(f"sim {similarity}, ind : {i}, path :{path}")

        # Find index and score of best matching image
        if(similarity > best_match):
            best_match = similarity
            index = i
        i+=1
    
    return index+1, best_match

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
    color = (0, 0, 255)
    text = f"Similarity: {similarity:.2f}"
    
    cv.putText(result, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return result


def morph_faces(image_path1 : str, image_path2 : str, model : str ='buffalo_l') -> np.ndarray:
    """
    ### Morph two images using embeddings from InsightFace.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        model (str): The model name to use for InsightFace (default: 'buffalo_l').

    Returns:
        numpy.ndarray: The morphed image as a numpy array.
    """

    # Load the images
    img1 = cv.imread(image_path1)
    img2 = cv.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {image_path1}, {image_path2}")
    
    # Detect faces in both images
    faces1 = app.get(img1)
    faces2 = app.get(img2)
    
    if not faces1 or not faces2:
        raise ValueError("No faces detected in one or both images.")
    
    print("probleme ici ?")
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download = False, download_zip = False)

    print("ou probleme là ?")
    # Image of user
    source_face = faces1[0]
    # Image of star
    target_face = faces2[0]
    
    morphed_image = img1.copy()
    morphed_image = swapper.get(morphed_image, target_face, source_face, paste_back=True)

    return morphed_image

def main():
    try:
        print(f"Insight version : {insightface.__version__}")
        print(f"Numpy version : {np.__version__}")
    except Exception as e:
        print(f"An error has occured : {e}")

    # Initialize the app
    initialize_face_analyzer()

    # Reading source image
    source_image_path = SAMPLES_DIR+"Renan2.JPG"
    source_embedding = extract_embedding(SAMPLES_DIR+"Renan2.JPG", app)

    # Find best score of matching star with source
    ind_img, score = find_best_match(source_embedding)
    matching_image_path = CELEBA_DIR+format_number(ind_img)

    # Visualize score with images side by side
    score_img = visualize_match(source_image_path, matching_image_path, score)
    # Store image in output folder
    cv.imwrite(OUTPUT_DIR+"best_match.jpg", score_img)
    print(f"Result saved to /app/{OUTPUT_DIR}best_match.jpg")


    # morphed_image = morph_faces(image2_path, image2_path, "buffalo_l")
    # cv.imwrite(OUTPUT_DIR+morphed_image_name,morphed_image)
    # print(f"Result saved to /app/{OUTPUT_DIR + morphed_image_name}")

    return 0

# Example usage
if __name__ == "__main__":

    main()

    while True:
        print("aa")
