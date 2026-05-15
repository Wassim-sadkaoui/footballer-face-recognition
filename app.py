import streamlit as st
from PIL import Image
import os
import cv2

from recognition.recognizer import recognize_face

st.set_page_config(
    page_title="Football Face Recognition",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #00FFAA; font-size: 3rem;'>
    Football Player Face Recognition
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align: center; color: #CCCCCC; font-size: 2rem;'>
    Upload a football player image and let the AI identify the player
    </h3>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    save_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    image.save(save_path)

    with st.spinner("Analyzing face..."):

        result_image, player_name, score = recognize_face(save_path)

    # Convert similarity to percentage
    confidence = int(score * 100)

    # Convert image BGR → RGB
    result_image = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    st.divider()

    st.subheader("Prediction Result")

    st.success(f"Player Identified: {player_name}")

    st.write(f"Confidence: {confidence}%")

    # Progress bar
    st.progress(min(confidence, 100))

    st.divider()

    st.subheader("Detected Face")

    st.image(
        result_image,
        use_container_width=True
    )