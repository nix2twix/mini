# === ENVIRONMENTS ===
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === LIBS GENERAL ===
import io
import json
import numpy as np
import cv2

import streamlit as st

from PIL import Image

# === PROJECT SCRIPTS ===
from processing import (
    process_tws,
    process_cellpose,
    process_dlgram
)

import tools


# === DEFAULT SESSION ===
def init_session_state():
    st.session_state.mode = None
    st.session_state.uploaded_file = None
    st.session_state.gt_file = None
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

    st.session_state.gt_file = st.file_uploader(
            "Загрузите разметку",
            type=["zip"]
        )

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

                mask_Cellpose = np.array(st.session_state.processed_file)

                BLOBs = []             
                unique_labels = np.unique(mask_Cellpose)
                unique_labels = unique_labels[(unique_labels > 0)]
            

                for label in unique_labels:
                    mask = (mask_Cellpose == label).astype(np.uint8)
                
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                  
                    
                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            x = int(M["m10"] / M["m00"])
                            y = int(M["m01"] / M["m00"])
                
                            # Вычисляем все расстояния между центром и точками контура
                            points = contour.reshape(-1, 2)
                            distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
                        
                            # Различные метрики радиуса
                            radius_metrics = {
                                'average_radius': np.mean(distances),
                                'median_radius':  np.median(distances),
                                'min_radius':     np.min(distances),
                                'max_radius':     np.max(distances),
                                'std_radius':     np.std(distances),
                                'effective_radius': np.sqrt(cv2.contourArea(contour) / np.pi),
                                'bounding_circle_radius': cv2.minEnclosingCircle(contour)[1]
                            }
                
                            BLOBs.append([y, x, radius_metrics['average_radius']])
                            temp_blobs = np.asarray(BLOBs)

                
                if (st.session_state.gt_file is not None):
                    gt_blobs, _, _ = tools.ImportTaskFromCVAT(st.session_state.gt_file)
                    gt_blobs[:, 2] = gt_blobs[:, 2] / 2 # диаметры в радиусы

                    gt_roi = tools.blobs2roi(gt_blobs, 960, 1280)

                    # 6. Оценка качества
                    match, no_match, fake, no_match_gt_blobs, fake_blobs, match_blobs, truedetected_blobs = tools.accur_estimation2(gt_blobs, temp_blobs, gt_roi, 0.25)
                    accuracy = match / (match+fake+no_match)
    
                    
    if st.session_state.processed_file is not None: 
        
        if (st.session_state.gt_file is not None):
            st.write(f'{accuracy*100:.2f}% (TP): {match}; (FN): {no_match}; (FP): {fake}')

        st.image(st.session_state.processed_file, caption="Обработанная маска", use_container_width=True)

        buffer = io.BytesIO()
        st.session_state.processed_file.save(buffer, format="PNG")
        st.download_button(
            label="Скачать результат",
            data=buffer.getvalue(),
            file_name="processed.png",
            mime="image/png"
        )
