﻿# === LIBRARIES GENERAL ===
import io

import streamlit as st

from PIL import Image

# === PROJECT SCRIPTS ===
from processing import (
    process_tws,
    process_cellpose,
    process_dlgram)

# === DEFAULT SESSION ===
def init_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None

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

    if st.session_state.mode:
        uploaded = st.file_uploader(
            "Загрузите изображение", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="uploaded_file"
        )

        if uploaded:
            st.session_state.uploaded_file = uploaded

# === RIGHT PANEL ===
with right:
    st.header("Результат обработки")

    if st.session_state.uploaded_file:
        image = Image.open(st.session_state.uploaded_file)

        if st.session_state.mode == "TWS":
            processed = process_tws(image)
        elif st.session_state.mode == "Cellpose":
            processed = process_cellpose(image)
        elif st.session_state.mode == "DLgram":
            processed = process_dlgram(image)
        else:
            processed = image

        st.session_state.processed_image = processed

        st.image(processed, caption="Обработанное изображение", use_column_width=True)

        buffer = io.BytesIO()
        processed.save(buffer, format="PNG")
        st.download_button(
            label="Скачать результат",
            data=buffer.getvalue(),
            file_name="processed.png",
            mime="image/png"
        )
    else:
        st.info("Выберите режим и загрузите изображение.")
