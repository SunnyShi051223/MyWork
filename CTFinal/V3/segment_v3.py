import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v3(image_path, output_dir):
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)
    
    # Create a subfolder for this image's steps
    image_output_dir = os.path.join(output_dir, name)
    ensure_dir(image_output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    h, w = img.shape[:2]

    # --- Step 1: Cropping (1_cropped) ---
    y_start, y_end = int(0.02 * h), int(0.98 * h)
    x_start, x_end = int(0.02 * w), int(0.95 * w)
    
    img_cropped = img[y_start:y_end, x_start:x_end]
    
    step1_path = os.path.join(image_output_dir, "1_cropped.png")
    cv2.imwrite(step1_path, img_cropped)

    # --- Step 2: Enhancement (2_enhanced) ---
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    step2_path = os.path.join(image_output_dir, "2_enhanced.png")
    cv2.imwrite(step2_path, enhanced)

    # --- Step 3: Binary Segmentation (3_binary) ---
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    step3_path = os.path.join(image_output_dir, "3_binary.png")
    cv2.imwrite(step3_path, opened)

    # --- Step 4: Final Contour (4_contour) with Internal Holes ---
    # Use RETR_TREE to get hierarchy
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy() 
    
    if contours:
        # Hierarchy: [Next, Previous, First_Child, Parent]
        # Find the largest contour (likely the main outer organ)
        max_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        max_contour = contours[max_idx]
        
        contours_to_draw = []
        
        # 1. Process Main Contour
        epsilon = 0.0001 * cv2.arcLength(max_contour, True) # High precision as requested
        refined_main = cv2.approxPolyDP(max_contour, epsilon, True)
        contours_to_draw.append(refined_main)
        
        # 2. Process Internal Holes (Children of the main contour)
        # Iterate through all contours to find those that have max_contour as parent (or ancestor)
        # For simplicity, we look for direct children or just any internal contour that is significant
        # A robust way is to check if a contour is inside the max_contour
        
        for i, cnt in enumerate(contours):
            if i == max_idx:
                continue
                
            # Check intersection or hierarchy. 
            # Simple check: hierarchy[0][i][3] is the parent index.
            # If parent is the max_contour, it's a direct hole.
            
            parent_idx = hierarchy[0][i][3]
            if parent_idx == max_idx and cv2.contourArea(cnt) > 100: # Filter small noise holes
                 epsilon_hole = 0.0001 * cv2.arcLength(cnt, True)
                 refined_hole = cv2.approxPolyDP(cnt, epsilon_hole, True)
                 contours_to_draw.append(refined_hole)

        # Restore coordinates and draw
        for cnt in contours_to_draw:
            cnt[:, 0, 0] += x_start
            cnt[:, 0, 1] += y_start
            cv2.drawContours(result_img, [cnt], -1, (0, 0, 255), 2)
    
    step4_path = os.path.join(image_output_dir, "4_contour.png")
    cv2.imwrite(step4_path, result_img)
    print(f"Processed V3 {filename} -> {image_output_dir}")

def process_directory_v3(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v3(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    process_directory_v3(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v3"))
    process_directory_v3(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v3"))
