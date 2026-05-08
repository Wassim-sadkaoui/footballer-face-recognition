import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# -----------------------------
# INIT MODEL
# -----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

DATASET_PATH = "dataset"
OUTPUT_FILE = "encodings/encodings.pkl"

known_embeddings = []
known_names = []

print("[INFO] Starting encoding process...")

# -----------------------------
# LOOP DATASET
# -----------------------------
for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing: {person_name}")

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            continue

        # take first face
        embedding = faces[0].embedding

        known_embeddings.append(embedding)
        known_names.append(person_name)

# -----------------------------
# SAVE DATABASE
# -----------------------------
data = {
    "embeddings": np.array(known_embeddings),
    "names": known_names
}

os.makedirs("encodings", exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding complete!")
print(f"[INFO] Saved {len(known_embeddings)} faces.")