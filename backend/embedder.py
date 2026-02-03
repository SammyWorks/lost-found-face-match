import streamlit as st
from keras_facenet import FaceNet
import numpy as np
import cv2

@st.cache_resource
def load_model():
    return FaceNet()

embedder = load_model()

def get_face_embedding(image, box):
    img = np.array(image)
    x, y, w, h = box

    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)[0]

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding
