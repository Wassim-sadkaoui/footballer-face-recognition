import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis


# LOAD MODEL

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# LOAD DATABASE

with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]


# COSINE SIMILARITY

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# PREDICT FUNCTION

def predict(image_path, threshold=0.35):
    img = cv2.imread(image_path)

    if img is None:
        return "Invalid image"

    faces = app.get(img)

    if len(faces) == 0:
        return "No face detected"

    query_embedding = faces[0].embedding

    best_score = -1
    best_name = "Unknown"

    for emb, name in zip(known_embeddings, known_names):
        score = cosine_similarity(query_embedding, emb)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "Unknown"

    return f"{best_name} ({best_score:.2f})"


# TEST

if __name__ == "__main__":
    result = predict("test.jpg")
    print("[RESULT]:", result)