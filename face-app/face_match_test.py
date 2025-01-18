import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

def initialize_face_analyzer():
    """### Initialize the InsightFace face analyzer with default configs"""
    face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer

def get_face_embedding(face_analyzer, image_path):
    """
    ### Extract face embedding from an image
    
    Args:
        face_analyzer: InsightFace face analyzer instance
        image_path: Path to the image file
        
    Returns:
        tuple: (face embedding, face bbox) or (None, None) if no face detected
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None, None
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_analyzer.get(img)
    
    if not faces:
        print(f"No face detected in: {image_path}")
        return None, None
    
    # Get the largest face if multiple faces are detected
    face = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]
    return face.embedding, face.bbox

def calculate_similarity(embedding1, embedding2):
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

def match_faces(image_path1, image_path2, threshold=0.5):
    """
    ### Compare two faces and determine if they match
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        threshold: Similarity threshold (default: 0.5)
        
    Returns:
        tuple: (bool indicating match, similarity score)
    """
    # Initialize face analyzer
    face_analyzer = initialize_face_analyzer()
    
    # Get embeddings for both images
    embedding1, bbox1 = get_face_embedding(face_analyzer, image_path1)
    embedding2, bbox2 = get_face_embedding(face_analyzer, image_path2)
    
    if embedding1 is None or embedding2 is None:
        return False, 0.0
    
    # Calculate similarity
    similarity = calculate_similarity(embedding1, embedding2)
    
    # Determine if faces match based on threshold
    is_match = similarity > threshold
    
    return is_match, similarity

def visualize_match(image_path1, image_path2, is_match, similarity):
    """
    ### Create a visualization of the face matching result
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        is_match: Boolean indicating if faces match
        similarity: Similarity score
    """
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Resize images to same height
    height = 300
    ratio1 = height / img1.shape[0]
    ratio2 = height / img2.shape[0]
    img1 = cv2.resize(img1, (int(img1.shape[1] * ratio1), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * ratio2), height))
    
    # Create result visualization
    result = np.hstack([img1, img2])
    color = (0, 255, 0) if is_match else (0, 0, 255)
    text = f"Match: {is_match}, Similarity: {similarity:.2f}"
    
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your image paths
    image1_path = "person1.jpg"
    image2_path = "person2.jpg"
    
    # Perform face matching
    is_match, similarity = match_faces(image1_path, image2_path)
    
    # Print results
    print(f"Face Match: {is_match}")
    print(f"Similarity Score: {similarity:.2f}")
    
    # Visualize results
    result_image = visualize_match(image1_path, image2_path, is_match, similarity)
    cv2.imshow("Face Matching Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()