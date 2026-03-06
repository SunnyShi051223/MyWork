import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v7(image_path, output_dir):
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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    cv2.imwrite(os.path.join(image_output_dir, "2_enhanced.png"), enhanced)

    # --- Step 3: K-Means Clustering ---
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
    
    # Visualization (Step 3)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    segmented_img = colors[sorted_labels.flatten()]
    segmented_img = segmented_img.reshape(img_cropped.shape)
    cv2.imwrite(os.path.join(image_output_dir, "3_kmeans_map.png"), segmented_img)

    # --- Step 4: Refined Multi-class Contours (ROI + Smooth) ---
    result_img = img.copy()
    labels_reshaped = sorted_labels.reshape(enhanced.shape)
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # Blue, Green, Red
    
    ignored_count = 0

    for class_idx in range(3):
        # Create binary mask
        mask = np.uint8(labels_reshaped == class_idx) * 255
        
        # --- FEATURE 1: Mask Smoothing (Softer Lines) ---
        # Instead of morphological closing which can still leave sharp edges,
        # we used Gaussian Blur on the binary mask then threshold again.
        # This rounds off the corners ("organic" look).
        
        # Standard closing first to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Blur the mask to smooth edges
        mask_blurred = cv2.GaussianBlur(mask_closed, (21, 21), 0)
        _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue

            # --- FEATURE 2: ROI Filtering (Edge Noise Removal) ---
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Filter 1: Too close to edges (Left/Right/Top/Bottom)
                # Keep only if centroid is within [5%, 95%] of width/height
                if (cX < 0.05 * cropped_w) or (cX > 0.95 * cropped_w) or \
                   (cY < 0.05 * cropped_h) or (cY > 0.95 * cropped_h):
                    ignored_count += 1
                    continue
                
                # Filter 2: Specific Top-Left Corner (often has "P" logo)
                # If in top-left 15% box
                if (cX < 0.15 * cropped_w) and (cY < 0.15 * cropped_h):
                    ignored_count += 1
                    continue

            # --- FEATURE 3: Smooth Approximation ---
            # High Precision Epsilon for exact tracing of the smooth mask
            epsilon = 0.0001 * cv2.arcLength(cnt, True)
            refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            corrected_cnt = refined_cnt.copy()
            corrected_cnt[:, 0, 0] += x_start
            corrected_cnt[:, 0, 1] += y_start
            
            cv2.drawContours(result_img, [corrected_cnt], -1, class_colors[class_idx], 2)

    cv2.imwrite(os.path.join(image_output_dir, "4_final_refined.png"), result_img)
    print(f"Processed V7 {filename}: Ignored {ignored_count} edge/noise contours.")

def process_directory_v7(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v7(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    if not os.path.exists(os.path.join(base_dir, "1", "results_v7")):
        os.makedirs(os.path.join(base_dir, "1", "results_v7"))
    if not os.path.exists(os.path.join(base_dir, "2", "results_v7")):
        os.makedirs(os.path.join(base_dir, "2", "results_v7"))

    process_directory_v7(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v7"))
    process_directory_v7(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v7"))
