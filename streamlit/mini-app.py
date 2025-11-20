# === ENVIRONMENTS ===
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === LIBS GENERAL ===
import io
import json
import streamlit as st

from PIL import Image

# === PROJECT SCRIPTS ===
from processing import (
    process_tws,
    process_cellpose,
    process_dlgram
)

# === DEFAULT SESSION ===
def init_session_state():
    st.session_state.mode = None
    st.session_state.uploaded_file = None
    st.session_state.processed_file = None
    
if "mode" not in st.session_state:
    init_session_state()
    
st.set_page_config(layout="wide")
left, right = st.columns([1, 2])

# === LEFT PANEL ===
with left:
    st.header("Настройки")
    mode = st.radio(
        "Выберите режим обработки:",
        ["TWS", "Cellpose", "DLgram"],
        index=None,
        key="mode"
    )
    if st.session_state.mode == "TWS":
        st.session_state.uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["png", "jpg", "jpeg", "tiff", "bmp"]
        )
    if st.session_state.mode == "Cellpose":
        st.session_state.uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["png", "tiff"]
        )
    if st.session_state.mode == "DLgram":
        st.session_state.uploaded_file = st.file_uploader(
            "Загрузите JSON файл DLgram",
            type=["json"]
        )
        
    if not st.session_state.uploaded_file:
        st.info("Выберите режим и загрузите файл.")

    processingClicked = st.button("Start processing", disabled=st.session_state.uploaded_file is None)

# === RIGHT PANEL ===
with right:
    st.header("Результат обработки")
    clearProcessingClicked = st.button("Clear processing result")
    if clearProcessingClicked:
        st.session_state.processed_file = None

    if processingClicked:
        if st.session_state.mode == "DLgram":
            json_data = json.load(st.session_state.uploaded_file)
            st.session_state.processed_file = process_dlgram(json_data)

        else:
            image = Image.open(st.session_state.uploaded_file)

            if st.session_state.mode == "TWS":
                st.session_state.processed_file = process_tws(image)

            elif st.session_state.mode == "Cellpose":
                st.session_state.processed_file = process_cellpose(image)
                
                
    if st.session_state.processed_file is not None: 
        st.image(st.session_state.processed_file, caption="Обработанная маска", use_container_width=True)

        buffer = io.BytesIO()
        st.session_state.processed_file.save(buffer, format="PNG")
        st.download_button(
            label="Скачать результат",
            data=buffer.getvalue(),
            file_name="processed.png",
            mime="image/png"
        )
