import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v4(image_path, output_dir):
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
    
    cv2.imwrite(os.path.join(image_output_dir, "1_cropped.png"), img_cropped)

    # --- Step 2: Enhancement ---
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    cv2.imwrite(os.path.join(image_output_dir, "2_enhanced.png"), enhanced)

    # --- Step 3: Binary Segmentation ---
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cv2.imwrite(os.path.join(image_output_dir, "3_binary.png"), opened)

    # --- Step 4: Final Contour with Intensity Validation ---
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy() 
    
    # We will use the 'gray' image for intensity checks
    # The 'gray' image corresponds to the cropped region
    
    if contours:
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

        # Process Holes
        hole_count = 0
        ignored_holes = 0

        for i, cnt in enumerate(contours):
            if i == max_idx:
                continue
            
            # Heuristic: Must be a child of the max contour
            parent_idx = hierarchy[0][i][3]
            if parent_idx == max_idx and cv2.contourArea(cnt) > 100:
                
                # --- Intensity Check ---
                # Create a mask for this specific contour (hole)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Calculate mean intensity in the original grayscale image (cropped)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # Threshold for "Black": < 40 (Adjustable)
                # Tissue varies, but usually > 50-60 in enhanced/gray images
                # Air is usually < 10-20
                DARK_THRESHOLD = 40 
                
                if mean_val < DARK_THRESHOLD:
                    # It's dark -> Real Hole (Air)
                    epsilon_hole = 0.0001 * cv2.arcLength(cnt, True)
                    refined_hole = cv2.approxPolyDP(cnt, epsilon_hole, True)
                    
                    corrected_hole = refined_hole.copy()
                    corrected_hole[:, 0, 0] += x_start
                    corrected_hole[:, 0, 1] += y_start
                    
                    cv2.drawContours(result_img, [corrected_hole], -1, (0, 255, 255), 2) # Yellow
                    hole_count += 1
                else:
                    # It's bright -> Tissue (Gray) -> Ignore
                    ignored_holes += 1
                    # Optional: Draw ignored holes in Blue for debugging? 
                    # refined_hole = cv2.approxPolyDP(cnt, 0.0001 * cv2.arcLength(cnt, True), True)
                    # refined_hole[:, 0, 0] += x_start
                    # refined_hole[:, 0, 1] += y_start
                    # cv2.drawContours(result_img, [refined_hole], -1, (255, 0, 0), 1)

    cv2.imwrite(os.path.join(image_output_dir, "4_contour.png"), result_img)
    print(f"Processed V4 {filename}: {hole_count} valid holes, {ignored_holes} ignored (tissue).")

def process_directory_v4(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v4(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    process_directory_v4(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v4"))
    process_directory_v4(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v4"))
