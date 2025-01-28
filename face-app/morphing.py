"""import cv2 as cv
import numpy as np
from facemorpher import morpher

def morph_images(image_path1: str, image_path2: str, alpha: float = 0.5) -> np.ndarray:
    
    Returns the face morphing of two images.

    Args:
        image_path1 (str): The path of image 1.
        image_path2 (str): The path of image 2.
        alpha (float): Merging factor (0.5 default = 50% of each image).

    Returns:
        np.ndarray: The morphed image object as array.
    
    # Load the images
    img1 = cv.imread(image_path1)
    img2 = cv.imread(image_path2)

    # Morph the images
    morphed_image = morpher.morph(img1, img2, alpha=alpha)

    return morphed_image

"""