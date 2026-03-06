import cv2
import numpy as np
import os
import glob

# --- GLOBAL THRESHOLDS ---
THRESH_LOW = 55
THRESH_HIGH = 200

def process_frame_v8(img):
    if img is None:
        return None

    h, w = img.shape[:2]

    # --- Step 1: Cropping ---
    y_start, y_end = int(0.02 * h), int(0.98 * h)
    x_start, x_end = int(0.02 * w), int(0.95 * w)
    img_cropped = img[y_start:y_end, x_start:x_end]
    cropped_h, cropped_w = img_cropped.shape[:2]

    # --- Step 2: Enhancement ---
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    # 双边滤波：保留超声组织边界，去除斑点噪声
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # --- Fix: Normalize Intensity ---
    # Video frames (Mean=37) are much darker than PNGs (Mean=51).
    # We normalize the enhanced image to 0-255 to match the PNG brightness distribution.
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # --- Step 3: Helper for Smooth Mask ---
    def get_smooth_mask(m):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.GaussianBlur(closed, (21, 21), 0)
        _, smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return smooth

    result_img = img.copy()

    # --- Step 4: Topological Contours ---
    # 4.1 Body (Green=Outer, Blue=Inner)
    mask_body = np.uint8(enhanced >= THRESH_LOW) * 255
    mask_body_smooth = get_smooth_mask(mask_body)
    
    contours_body, hierarchy_body = cv2.findContours(mask_body_smooth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy_body is not None:
        for i, cnt in enumerate(contours_body):
            if cv2.contourArea(cnt) < 200:
                continue
            
            parent_idx = hierarchy_body[0][i][3]
            if parent_idx == -1:
                color = (0, 255, 0) # Green (Outer)
            else:
                color = (255, 0, 0) # Blue (Inner Hole)

            # ROI Filter
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if (cX < 0.05 * cropped_w) or (cX > 0.95 * cropped_w) or \
                   (cY < 0.05 * cropped_h) or (cY > 0.95 * cropped_h):
                    continue
                if (cX < 0.15 * cropped_w) and (cY < 0.15 * cropped_h):
                    continue

            # Smooth Approximation
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            corrected_cnt = refined_cnt.copy()
            corrected_cnt[:, 0, 0] += x_start
            corrected_cnt[:, 0, 1] += y_start
            
            cv2.drawContours(result_img, [corrected_cnt], -1, color, 2)

    # 4.2 Bone (Red)
    mask_bone = np.uint8(enhanced >= THRESH_HIGH) * 255
    mask_bone_smooth = get_smooth_mask(mask_bone)
    
    contours_bone, _ = cv2.findContours(mask_bone_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_bone:
        if cv2.contourArea(cnt) < 100:
            continue
            
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        
        corrected_cnt = refined_cnt.copy()
        corrected_cnt[:, 0, 0] += x_start
        corrected_cnt[:, 0, 1] += y_start
        
        cv2.drawContours(result_img, [corrected_cnt], -1, (0, 0, 255), 2)
    
    return result_img

def process_video_v8(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video parsing stream: {video_path}")
        return

    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)
    
    output_video_path = os.path.join(output_dir, f"{name}_v8.avi")
    print(f"Processing video V8: {video_path} -> {output_video_path}")

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
            
        processed_frame = process_frame_v8(frame)
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
    
    if not os.path.exists(os.path.join(base_dir, "1", "results_v8")):
        os.makedirs(os.path.join(base_dir, "1", "results_v8"))
    if not os.path.exists(os.path.join(base_dir, "2", "results_v8")):
        os.makedirs(os.path.join(base_dir, "2", "results_v8"))

    videos1 = glob.glob(os.path.join(base_dir, "1", "*.avi"))
    for vid in videos1:
        process_video_v8(vid, os.path.join(base_dir, "1", "results_v8"))

    videos2 = glob.glob(os.path.join(base_dir, "2", "*.avi"))
    for vid in videos2:
        process_video_v8(vid, os.path.join(base_dir, "2", "results_v8"))
