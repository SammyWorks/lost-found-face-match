from mtcnn import MTCNN
import numpy as np
import cv2

detector = MTCNN()

def detect_face(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img)

    if not results:
        return None

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return results[0]
