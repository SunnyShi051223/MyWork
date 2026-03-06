import cv2
import numpy as np
import os
import glob

# --- GLOBAL THRESHOLDS (Physics Based) ---
# Adjust these based on medical ultrasound standards or user preference
THRESH_LOW = 55   # Below this is Air/Fluid (Blue)
THRESH_HIGH = 200 # Above this is Bone/Calcification (Red)
# Everything in between is Soft Tissue (Green)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v8(image_path, output_dir):
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)
    
    image_output_dir = os.path.join(output_dir, name)
    ensure_dir(image_output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    h, w = img.shape[:2]

    # --- Step 1: Cropping ---
    y_start, y_end = int(0.02 * h), int(0.98 * h)
    x_start, x_end = int(0.02 * w), int(0.95 * w)
    img_cropped = img[y_start:y_end, x_start:x_end]
    cropped_h, cropped_w = img_cropped.shape[:2]
    
    cv2.imwrite(os.path.join(image_output_dir, "1_cropped.png"), img_cropped)

    # --- Step 2: Enhancement ---
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    # 使用双边滤波替代高斯模糊，以保留组织边界并去除斑点噪声
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    cv2.imwrite(os.path.join(image_output_dir, "2_enhanced.png"), enhanced)

    # --- Step 3: Helper for Smooth Mask ---
    def get_smooth_mask(m):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.GaussianBlur(closed, (21, 21), 0)
        _, smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return smooth

    # --- Step 4: Topological Contours ---
    result_img = img.copy()
    
    # 4.1 Body Mask (> THRESH_LOW) -> Green (Outer) & Blue (Inner)
    mask_body = np.uint8(enhanced >= THRESH_LOW) * 255
    mask_body_smooth = get_smooth_mask(mask_body)
    
    # Use RETR_CCOMP to get 2-level hierarchy (External + Holes)
    contours_body, hierarchy_body = cv2.findContours(mask_body_smooth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy_body is not None:
        for i, cnt in enumerate(contours_body):
            if cv2.contourArea(cnt) < 200:
                continue
            
            # Hierarchy: [Next, Prev, First_Child, Parent]
            parent_idx = hierarchy_body[0][i][3]
            
            # Determine Color: External=Green, Internal=Blue
            if parent_idx == -1:
                color = (0, 255, 0) # Green (Tissue)
            else:
                color = (255, 0, 0) # Blue (Air/Hole)

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

            # High Precision Smoothing
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            corrected_cnt = refined_cnt.copy()
            corrected_cnt[:, 0, 0] += x_start
            corrected_cnt[:, 0, 1] += y_start
            
            cv2.drawContours(result_img, [corrected_cnt], -1, color, 2)

    # 4.2 Bone Mask (> THRESH_HIGH) -> Red
    mask_bone = np.uint8(enhanced >= THRESH_HIGH) * 255
    mask_bone_smooth = get_smooth_mask(mask_bone)
    
    contours_bone, _ = cv2.findContours(mask_bone_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_bone:
        if cv2.contourArea(cnt) < 100: # Bone might be smaller
            continue
            
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        
        corrected_cnt = refined_cnt.copy()
        corrected_cnt[:, 0, 0] += x_start
        corrected_cnt[:, 0, 1] += y_start
        
        cv2.drawContours(result_img, [corrected_cnt], -1, (0, 0, 255), 2) # Red

    cv2.imwrite(os.path.join(image_output_dir, "4_final_v8.png"), result_img)

def process_directory_v8(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v8(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    if not os.path.exists(os.path.join(base_dir, "1", "results_v8")):
        os.makedirs(os.path.join(base_dir, "1", "results_v8"))
    if not os.path.exists(os.path.join(base_dir, "2", "results_v8")):
        os.makedirs(os.path.join(base_dir, "2", "results_v8"))

    process_directory_v8(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v8"))
    process_directory_v8(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v8"))
