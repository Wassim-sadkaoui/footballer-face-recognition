import cv2
import pickle
import numpy as np

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


app = FaceAnalysis()
app.prepare(ctx_id=0)


with open("encodings/encodings.pkl", "rb") as f:
    database = pickle.load(f)

THRESHOLD = 0.35


def recognize_face(image_path):

    image = cv2.imread(image_path)

    faces = app.get(image)

    if len(faces) == 0:
        return image, "No face detected"

    best_result = "Unknown"

    for face in faces:

        embedding = face.embedding

        best_match = "Unknown"
        best_score = -1

        for player_name, embeddings in database.items():

            for db_embedding in embeddings:

                similarity = cosine_similarity(
                    [embedding],
                    [db_embedding]
                )[0][0]

                if similarity > best_score:
                    best_score = similarity
                    best_match = player_name

        if best_score < THRESHOLD:
            label = "Unknown"
        else:
            label = f"{best_match} ({best_score:.2f})"

        # Face box coordinates
        x1, y1, x2, y2 = map(int, face.bbox)

        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        best_result = label

    return image, best_match, best_score