import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import os
import pandas as pd
import json

# Load models once
model_path = r"C:\Users\Rambali\Downloads\best (2).pt"
model = YOLO(model_path)

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(img_crop):
    img_tensor = transform(img_crop).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_tensor).squeeze().numpy()
    return embedding

def detect_and_embed(video_path, every_n_frames=30):
    cap = cv2.VideoCapture(video_path)
    player_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            results = model(frame)[0]
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    features = extract_features(crop)
                    player_data.append({
                        'frame': frame_idx,
                        'bbox': (x1, y1, x2, y2),
                        'features': features
                    })
        frame_idx += 1
    cap.release()
    return player_data

def match_players(broadcast_data, tactima_data, threshold=0.8):
    # Assign IDs based on best cosine similarity
    for tactima_player in tactima_data:
        best_score = -1
        best_match_id = None
        for broadcast_player in broadcast_data:
            score = cosine_similarity(
                [tactima_player['features']], [broadcast_player['features']]
            )[0][0]
            if score > best_score and score > threshold:
                best_score = score
                best_match_id = broadcast_player.get('player_id', None)
        if best_match_id is not None:
            tactima_player['player_id'] = best_match_id
        else:
            # Assign a new unique ID if no match
            tactima_player['player_id'] = f"player_{uuid.uuid4().hex[:8]}"
    return tactima_data

def draw_and_save_video(video_path, player_data, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_idx = 0
    data_dict = {}
    for p in player_data:
        data_dict.setdefault(p['frame'], []).append(p)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in data_dict:
            for p in data_dict[frame_idx]:
                x1, y1, x2, y2 = p['bbox']
                player_id = str(p.get('player_id', 'N/A'))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path

# -------------------- Streamlit UI --------------------

st.title("🎯 Cross-Camera Player Mapping App")
st.write("Upload two videos of the same match from different camera angles to map consistent player IDs.")

broadcast_file = st.file_uploader("Upload Broadcast Video", type=["mp4"])
tactima_file = st.file_uploader("Upload Tactima Video", type=["mp4"])

if st.button("🔍 Start Player Mapping") and broadcast_file and tactima_file:
    with st.spinner("Processing videos..."):
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as broadcast_temp:
            broadcast_temp.write(broadcast_file.read())
            broadcast_path = broadcast_temp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tactima_temp:
            tactima_temp.write(tactima_file.read())
            tactima_path = tactima_temp.name

        # Detect players
        st.info("Detecting players in Broadcast video...")
        broadcast_players = detect_and_embed(broadcast_path)

        st.info("Detecting players in Tactima video...")
        tactima_players = detect_and_embed(tactima_path)

        # Match players
        st.info("Matching players across both feeds...")
        matched_players = match_players(broadcast_players, tactima_players)

        # Save annotated videos
        out1 = "broadcast_annotated.mp4"
        out2 = "tactima_annotated.mp4"
        draw_and_save_video(broadcast_path, broadcast_players, out1)
        draw_and_save_video(tactima_path, matched_players, out2)

        # Display videos
        st.video(out1, format="video/mp4", start_time=0)
        st.download_button("Download Annotated Broadcast Video", open(out1, 'rb'), file_name="broadcast_annotated.mp4")
        st.video(out2, format="video/mp4", start_time=0)
        st.download_button("Download Annotated Tactima Video", open(out2, 'rb'), file_name="tactima_annotated.mp4")

# Export data
def export_player_data(broadcast_data, tactima_data):
    records = []
    for p in broadcast_data:
        records.append({
            "frame": p["frame"],
            "video": "broadcast",
            "player_id": p.get("player_id", "N/A"),
            "bbox": p["bbox"]
        })
    for p in tactima_data:
        records.append({
            "frame": p["frame"],
            "video": "tactima",
            "player_id": p.get("player_id", "N/A"),
            "bbox": p["bbox"]
        })
    df = pd.DataFrame(records)
    csv_path = "player_mapping.csv"
    json_path = "player_mapping.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    return csv_path, json_path

# Download data
if 'broadcast_players' in locals() and 'matched_players' in locals():
    csv_path, json_path = export_player_data(broadcast_players, matched_players)
    st.success("Player data saved!")
    with open(csv_path, "rb") as f:
        st.download_button("📥 Download CSV", f, file_name="player_mapping.csv", mime="text/csv")
    with open(json_path, "rb") as f:
        st.download_button("📥 Download JSON", f, file_name="player_mapping.json", mime="application/json")
