import os
import cv2
import numpy as np
import glob

def draw_multiple_bounding_boxes(image_path, txt_file_path):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Read the bounding box coordinates from the txt file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            label = data[0]
            x_center = float(data[1])
            y_center = float(data[2])
            bbox_width = float(data[3])
            bbox_height = float(data[4])
            
            # Convert normalized coordinates back to pixel coordinates
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)
            
            # Draw a rectangle around the detected bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return image

def save_image_with_reference(labeled_image, reference_image_path, output_path):
    # Read the reference image
    reference_image = cv2.imread(reference_image_path)
    
    # Ensure both images have the same height
    height = max(reference_image.shape[0], labeled_image.shape[0])
    width_ref = reference_image.shape[1]
    width_lbl = labeled_image.shape[1]
    
    # If the heights differ, resize the images
    if reference_image.shape[0] != height:
        reference_image = cv2.resize(reference_image, (width_ref, height))
    if labeled_image.shape[0] != height:
        labeled_image = cv2.resize(labeled_image, (width_lbl, height))
    
    # Concatenate the two images side by side
    concatenated_image = np.hstack((reference_image, labeled_image))
    
    # Save the concatenated image
    cv2.imwrite(output_path, concatenated_image)

def label_image_with_txt(input_image_path, txt_file_path, output_image_path, reference_image_path=None, output_reference_path=None):
    # Draw bounding boxes on the image based on the txt file
    labeled_image = draw_multiple_bounding_boxes(input_image_path, txt_file_path)
    
    # Save the labeled image
    cv2.imwrite(output_image_path, labeled_image)
    
    # If a reference image path is provided, create a side-by-side comparison image
    if reference_image_path and output_reference_path:
        save_image_with_reference(labeled_image, reference_image_path, output_reference_path)

if __name__ == "__main__":
    # Example usage
    rootd='/home/ch215616/w/code/llm/experiments/yolov7/yolov7/elbow_fracture_dataset/'
    f_="21_*___2_Oblique.jpg"
    f = glob.glob(f_)[0]
    input_image = rootd + "images_unlabelled/" +f 
    txt_file = rootd + "labels/" +f.replace('.jpg', '.txt')
    output_image = rootd + "images_remove/" +f 
    reference_image = rootd + "images_labelled/" +f 
    output_reference_image = rootd + "images_side_by_side/" +f 
    
    # Label image and save the labeled image
    label_image_with_txt(input_image, txt_file, output_image)
    
    # If you want to compare with a reference image
    label_image_with_txt(input_image, txt_file, output_image, reference_image, output_reference_image)
