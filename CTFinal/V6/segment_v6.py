import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def segment_image_v6(image_path, output_dir):
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

    # --- Step 3: K-Means Clustering (Multi-class) ---
    # Reshape image to a 1D array of float32
    pixel_values = enhanced.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # K-Means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3 # 3 Classes: Air (Dark), Tissue (Gray), Bone (Bright)
    
    # Run K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Sort centers to ensure:
    # Index 0 = Darkest (Air)
    # Index 1 = Medium (Tissue)
    # Index 2 = Brightest (Bone)
    centers = np.uint8(centers)
    sorted_indices = np.argsort(centers.flatten())
    centers_sorted = centers[sorted_indices]
    
    # Map old labels to new sorted labels
    lut = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        lut[sorted_indices[i]] = i
        
    sorted_labels = lut[labels.flatten()]
    
    # Visualize the 3 classes
    # Class 0 (Air) -> Blue
    # Class 1 (Tissue) -> Green
    # Class 2 (Bone) -> Red
    colors = np.array([
        [255, 0, 0],   # Blue
        [0, 255, 0],   # Green
        [0, 0, 255]    # Red
    ], dtype=np.uint8)
    
    segmented_img = colors[sorted_labels.flatten()]
    segmented_img = segmented_img.reshape(img_cropped.shape)
    
    cv2.imwrite(os.path.join(image_output_dir, "3_kmeans_map.png"), segmented_img)

    # --- Step 4: Extract Contours for Each Class ---
    result_img = img.copy()
    
    # Reshape labels back to image shape for masking
    labels_reshaped = sorted_labels.reshape(enhanced.shape)

    class_names = ["Air/Background", "Software Tissue", "Bone/HighDensity"]
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # Blue, Green, Red

    for class_idx in range(3):
        # Create binary mask for this class
        mask = np.uint8(labels_reshaped == class_idx) * 255
        
        # Clean up mask (optional closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw significant contours
        for cnt in contours:
            if cv2.contourArea(cnt) > 200: # Filter small noise
                 epsilon = 0.001 * cv2.arcLength(cnt, True)
                 refined_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                 
                 # Restore coordinates
                 corrected_cnt = refined_cnt.copy()
                 corrected_cnt[:, 0, 0] += x_start
                 corrected_cnt[:, 0, 1] += y_start
                 
                 cv2.drawContours(result_img, [corrected_cnt], -1, class_colors[class_idx], 2)

    cv2.imwrite(os.path.join(image_output_dir, "4_multiclass_contour.png"), result_img)
    print(f"Processed V6 {filename}: Segmented into {class_names}")

def process_directory_v6(input_dir, output_dir):
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        segment_image_v6(img_path, output_dir)

if __name__ == "__main__":
    base_dir = r"d:/HuaweiMoveData/Users/32874/Desktop/ZJU/week3"
    
    process_directory_v6(os.path.join(base_dir, "1"), os.path.join(base_dir, "1", "results_v6"))
    process_directory_v6(os.path.join(base_dir, "2"), os.path.join(base_dir, "2", "results_v6"))
