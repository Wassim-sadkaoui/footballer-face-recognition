import os
import pickle
import cv2

from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0)

DATASET_PATH = "dataset"

database = {}

# Loop through each player folder
for player_name in os.listdir(DATASET_PATH):

    player_folder = os.path.join(
        DATASET_PATH,
        player_name
    )

    if not os.path.isdir(player_folder):
        continue

    embeddings = []

    # Loop through images
    for image_name in os.listdir(player_folder):

        image_path = os.path.join(
            player_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            continue

        # Take first detected face
        embedding = faces[0].embedding

        embeddings.append(embedding)

    database[player_name] = embeddings

# Save database
with open("encodings/encodings.pkl", "wb") as f:
    pickle.dump(database, f)

print("Encodings saved successfully!")