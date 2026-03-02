import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#  EXTRACT FRAMES FUNCTION
def extract_function(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

   
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25
    frame_interval = max(1, int(round(fps)))  

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames.")
    return output_folder



