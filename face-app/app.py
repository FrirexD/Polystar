import insightface
import pkg_resources
import numpy as np
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def main():
    try:
        # Obtenir la version de insightface
        print(f"Version de insightface : {insightface.__version__}")
        print(f"Version de numpy : {np.__version__}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

    model = "buffalo_l"
    face_detection(model)

def face_detection(model):
    app = FaceAnalysis(providers = ["CUDAExecutionProvider", "CPUExecutionProvider"], name=model)
    app.prepare(ctx_id=0, det_size=(640,640))
    img = ins_get_image('t1')
    faces = app.get(img)
    rimg = app.draw_on(img, faces)
    cv2.imwrite("./t1.jpg",rimg)

    print("ça marche j'ai juré")



if __name__ == "__main__":
    main()

