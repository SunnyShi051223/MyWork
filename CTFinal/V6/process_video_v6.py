import cv2
import numpy as np
import os
import glob

def process_frame_v6(img):
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

    # --- Step 3: K-Means --
    pixel_values = enhanced.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3 
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    sorted_indices = np.argsort(centers.flatten())
    lut = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        lut[sorted_indices[i]] = i
    sorted_labels = lut[labels.flatten()]
    
    # --- Step 4: Contours ---
    result_img = img.copy()
    labels_reshaped = sorted_labels.reshape(enhanced.shape)
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # Blue(Air), Green(Tissue), Red(Bone)

    for class_idx in range(3):
        mask = np.uint8(labels_reshaped == class_idx) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                 epsilon = 0.001 * cv2.arcLength(cnt, True)
                 refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                 
                 corrected_cnt = refined_cnt.copy()
                 corrected_cnt[:, 0, 0] += x_start
                 corrected_cnt[:, 0, 1] += y_start
                 
                 cv2.drawContours(result_img, [corrected_cnt], -1, class_colors[class_idx], 2)
    
    return result_img

def process_video_v6(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video parsing stream: {video_path}")
        return

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    
    output_video_path = os.path.join(output_dir, f"{name}_v6.avi")
    print(f"Processing video V6: {video_path} -> {output_video_path}")

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
            
        processed_frame = process_frame_v6(frame)
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
    
    if not os.path.exists(os.path.join(base_dir, "1", "results_v6")):
        os.makedirs(os.path.join(base_dir, "1", "results_v6"))
    if not os.path.exists(os.path.join(base_dir, "2", "results_v6")):
        os.makedirs(os.path.join(base_dir, "2", "results_v6"))

    videos1 = glob.glob(os.path.join(base_dir, "1", "*.avi"))
    for vid in videos1:
        process_video_v6(vid, os.path.join(base_dir, "1", "results_v6"))

    videos2 = glob.glob(os.path.join(base_dir, "2", "*.avi"))
    for vid in videos2:
        process_video_v6(vid, os.path.join(base_dir, "2", "results_v6"))
