import cv2
import numpy as np
from insightface.app import FaceAnalysis

def get_landmarks(image_path: str, app: FaceAnalysis) -> np.ndarray:
    """
    ### Extract facial landmarks from an image using InsightFace.

    Args:
        image_path (str): Path to the image.
        app (FaceAnalysis): The InsightFace application instance.

    Returns:
        numpy.ndarray: The facial landmarks as a numpy array.
    """

    image = cv2.imread(image_path)
    faces = app.get(image)
    if len(faces) == 0:
        raise ValueError("No face detected")
    landmarks = faces[0]['landmark_2d_106']
    return np.array(landmarks, dtype=np.float32)

def calculate_delaunay_triangles(rect: tuple, points: np.ndarray) -> list:
    """
    ### Calculate Delaunay triangles for a set of points within a rectangle.

    Args:
        rect (tuple): The rectangle (x, y, width, height).
        points (numpy.ndarray): The points to calculate triangles for.

    Returns:
        list: A list of triangles, each represented as a list of three points.
    """

    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()
    triangles = []
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            triangles.append([pt1, pt2, pt3])
    return triangles

def rect_contains(rect: tuple, point: tuple) -> bool:
    """
    ### Check if a point is within a rectangle.

    Args:
        rect (tuple): The rectangle (x, y, width, height).
        point (tuple): The point (x, y).

    Returns:
        bool: True if the point is within the rectangle, False otherwise.
    """
        
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

def apply_affine_transform(src: np.ndarray, src_tri: list, dst_tri: list, size: tuple) -> np.ndarray:
    """
    ### Apply an affine transformation to a source image.

    Args:
        src (numpy.ndarray): The source image.
        src_tri (list): The source triangle points.
        dst_tri (list): The destination triangle points.
        size (tuple): The size of the destination image (width, height).

    Returns:
        numpy.ndarray: The transformed image.
    """
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1: np.ndarray, img2: np.ndarray, img: np.ndarray, t1: list, t2: list, t: list, alpha: float) -> None:
    """
    ### Morph a triangle from two images.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        img (numpy.ndarray): The output image.
        t1 (list): The first triangle points.
        t2 (list): The second triangle points.
        t (list): The morphed triangle points.
        alpha (float): The morphing factor (0.0 to 1.0).
    """

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask

def morph_images(image_path1: str, image_path2: str, alpha: float=0.5, app: FaceAnalysis = None) -> np.ndarray:
    """
    ### Returns the face morphing of two images.

    Args:
        image_path1: The path of image 1
        image_path2: The path of image 2
        alpha: Mergir factor (0.5 default = 50% of each image)
        app: FaceAnalysis object

    Returns:
        img_morph: The morphed image object as array
    """

    points1 = get_landmarks(image_path1, app)
    points2 = get_landmarks(image_path2, app)

    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    rect = (0, 0, img1.shape[1], img1.shape[0])
    triangles = calculate_delaunay_triangles(rect, points1)

    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)
    for t in triangles:
        x = t[0][0]
        y = t[0][1]
        t1 = [(x, y), (t[1][0], t[1][1]), (t[2][0], t[2][1])]
        x = t[0][0]
        y = t[0][1]
        t2 = [(x, y), (t[1][0], t[1][1]), (t[2][0], t[2][1])]
        x = (1 - alpha) * t1[0][0] + alpha * t2[0][0]
        y = (1 - alpha) * t1[0][1] + alpha * t2[0][1]
        t = [(x, y),
             ((1 - alpha) * t1[1][0] + alpha * t2[1][0], (1 - alpha) * t1[1][1] + alpha * t2[1][1]),
             ((1 - alpha) * t1[2][0] + alpha * t2[2][0], (1 - alpha) * t1[2][1] + alpha * t2[2][1])]
        morph_triangle(img1, img2, img_morphed, t1, t2, t, alpha)

    return img_morphed