import cv2
import os
import glob
import numpy as np

def extract_frames(video_path, output_dir="frames", step=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            filename = f"{output_dir}/frame_{saved:04d}.jpg"
            cv2.imwrite(filename, frame)
            saved += 1
        i += 1

    cap.release()
    print(f"Extracted {saved} frames into '{output_dir}'.")

def stitch_frames(frames_dir="frames", output_file="minecraft_scan.jpg"):
    images = []
    files = sorted(glob.glob(f"{frames_dir}/*.jpg"))  

    for filename in files:
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
            images.append(img)

    if len(images) < 2:
        print("Not enough frames to stitch!")
        return

    print(f"Stitching {len(images)} frames...")

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_file, pano)
        print(f"[OK] Panorama saved as '{output_file}'")
    else:
        print(f"[ERROR] Stitching failed with code {status}")

if __name__ == "__main__":
    video_path = "Minecraft_stitch_test.mp4"  
    extract_frames(video_path, step=30)
    stitch_frames()
