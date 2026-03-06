import cv2
import numpy as np
import os
import glob

def process_frame_v3(img):
    if img is None:
        return None

    h, w = img.shape[:2]

    # --- Step 1: Cropping ---
    y_start, y_end = int(0.02 * h), int(0.98 * h)
    x_start, x_end = int(0.02 * w), int(0.95 * w)
    img_cropped = img[y_start:y_end, x_start:x_end]

    # --- Step 2: Enhancement ---
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # --- Step 3: Binary Segmentation ---
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Step 4: Final Contour with Holes ---
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy()
    
    if contours:
        # Find largest contour (Parent)
        max_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        max_contour = contours[max_idx]
        
        # Draw Parent
        epsilon = 0.0001 * cv2.arcLength(max_contour, True)
        refined_main = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Restore coordinates
        corrected_main = refined_main.copy()
        corrected_main[:, 0, 0] += x_start
        corrected_main[:, 0, 1] += y_start
        cv2.drawContours(result_img, [corrected_main], -1, (0, 0, 255), 2)

        # Find and Draw Holes (Children)
        for i, cnt in enumerate(contours):
            if i == max_idx:
                continue
                
            # Check if parent is the max_contour (direct child) or just any hole
            # Using hierarchy[0][i][3] == max_idx is a good heuristic
            parent_idx = hierarchy[0][i][3]
            
            if parent_idx == max_idx and cv2.contourArea(cnt) > 100:
                 epsilon_hole = 0.0001 * cv2.arcLength(cnt, True)
                 refined_hole = cv2.approxPolyDP(cnt, epsilon_hole, True)
                 
                 corrected_hole = refined_hole.copy()
                 corrected_hole[:, 0, 0] += x_start
                 corrected_hole[:, 0, 1] += y_start
                 
                 # Draw holes in Yellow to distinguish
                 cv2.drawContours(result_img, [corrected_hole], -1, (0, 255, 255), 2)
    
    return result_img

def process_video_v3(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video parsing stream: {video_path}")
        return

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    
    output_video_path = os.path.join(output_dir, f"{name}_v3.avi")
    print(f"Processing video V3: {video_path} -> {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame_v3(frame)
        if processed_frame is not None:
             out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Finished processing {filename}.")

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    if not os.path.exists(os.path.join(base_dir, "1", "results_v3")):
        os.makedirs(os.path.join(base_dir, "1", "results_v3"))
    if not os.path.exists(os.path.join(base_dir, "2", "results_v3")):
        os.makedirs(os.path.join(base_dir, "2", "results_v3"))

    videos1 = glob.glob(os.path.join(base_dir, "1", "*.avi"))
    for vid in videos1:
        process_video_v3(vid, os.path.join(base_dir, "1", "results_v3"))

    videos2 = glob.glob(os.path.join(base_dir, "2", "*.avi"))
    for vid in videos2:
        process_video_v3(vid, os.path.join(base_dir, "2", "results_v3"))
