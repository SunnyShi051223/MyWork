import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v2(image_path, output_dir):
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
    # Crop 2% from edges as per Test2.py
    y_start, y_end = int(0.02 * h), int(0.98 * h)
    x_start, x_end = int(0.02 * w), int(0.95 * w)
    
    img_cropped = img[y_start:y_end, x_start:x_end]
    
    step1_path = os.path.join(image_output_dir, "1_cropped.png")
    cv2.imwrite(step1_path, img_cropped)

    # --- Step 2: Enhancement (2_enhanced) ---
    # Convert to grayscale
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur (3x3)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    step2_path = os.path.join(image_output_dir, "2_enhanced.png")
    cv2.imwrite(step2_path, enhanced)

    # --- Step 3: Binary Segmentation (3_binary) ---
    # Otsu Thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology: Close (Ellipse 5x5) -> Open (Ellipse 5x5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    step3_path = os.path.join(image_output_dir, "3_binary.png")
    cv2.imwrite(step3_path, opened)

    # --- Step 4: Final Contour (4_contour) ---
    # Find Contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy() # Draw on original image
    
    if contours:
        # Keep only the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour
        epsilon = 0.0001 * cv2.arcLength(max_contour, True) # Sligthly looser than 0.0001 for smoother look
        refined_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Restore coordinates to original image space
        # We cropped by x_start and y_start
        refined_contour[:, 0, 0] += x_start
        refined_contour[:, 0, 1] += y_start
        
        # Draw contour (Red, thickness 2)
        cv2.drawContours(result_img, [refined_contour], -1, (0, 0, 255), 2)
    
    step4_path = os.path.join(image_output_dir, "4_contour.png")
    cv2.imwrite(step4_path, result_img)
    
    print(f"Processed {filename} -> {image_output_dir}")

def process_directory_v2(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v2(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    # Process dataset 1
    process_directory_v2(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v2"))
    
    # Process dataset 2
    process_directory_v2(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v2"))
