#  Football Player Face Recognition

An AI-powered football player face recognition system built using Python, InsightFace, OpenCV, and Streamlit.

The application detects a human face from an uploaded image, extracts facial embeddings, compares them with known football player embeddings, and predicts the player identity.

---

#  Features

- Face detection using InsightFace
- Facial embedding extraction
- Cosine similarity matching
- Football player identification
- Bounding box visualization
- Confidence score display
- Interactive Streamlit web interface

---

#  AI Concepts Used

This project includes several Artificial Intelligence and Computer Vision concepts:

- Computer Vision
- Face Detection
- Face Recognition
- Deep Learning Embeddings
- Feature Extraction
- Similarity Matching
- Pattern Recognition

---

#  Technologies Used

## Programming Language

- Python 3

## Libraries & Frameworks

- InsightFace
- OpenCV
- NumPy
- Scikit-learn
- Pillow
- Streamlit

---

#  Project Structure

```bash
football-face-recognition/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
├── dataset_builder/
│   └── datasetbuilder.py
│
├── encodings/
│   ├── encode_faces.py
│   └── encodings.pkl
│
├── recognition/
│   └── recognizer.py
│
└── uploads/
