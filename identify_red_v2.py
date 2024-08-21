import os
import cv2
import numpy as np

def extract_red_bounding_box_coordinates(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the bounding box
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)
    else:
        return None

def save_coordinates_to_file(coordinates, image_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    x1, y1, x2, y2 = coordinates
    
    # Convert coordinates to the required format
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height
    
    # Save to file
    with open(output_path, 'w') as file:
        file.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
    
    # Return normalized coordinates for drawing
    return x1, y1, x2, y2

def draw_red_bounding_box(image_path, coordinates):
    # Read the image
    image = cv2.imread(image_path)
    
    if coordinates:
        x1, y1, x2, y2 = coordinates
        # Draw a red rectangle around the detected bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return image

def process_images(input_dir, output_labels_dir, output_images_dir, unlabelled_images_dir):
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            image_path = os.path.join(input_dir, filename)
            coordinates = extract_red_bounding_box_coordinates(image_path)
            
            if coordinates:
                # Create and save the text file with coordinates
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_output_path = os.path.join(output_labels_dir, label_filename)
                x1, y1, x2, y2 = save_coordinates_to_file(coordinates, image_path, label_output_path)
                
                # Use the coordinates to draw on the image from /images_unlabelled/
                unlabelled_image_path = os.path.join(unlabelled_images_dir, filename)
                labelled_image = draw_red_bounding_box(unlabelled_image_path, (x1, y1, x2, y2))
                
                # Save the labelled image in /images_labelled/
                labelled_image_output_path = os.path.join(output_images_dir, filename)
                cv2.imwrite(labelled_image_output_path, labelled_image)

                # Commented out matplotlib display
                # plt.imshow(labelled_image)
                # plt.axis('off')
                # plt.show()
            else:
                print(f"No red bounding box found in {filename}.")

if __name__ == "__main__":
    rootd='/home/ch215616/w/code/llm/experiments/yolov7/yolov7/elbow_fracture/'
    input_directory = rootd+ "images_labelled/"
    output_labels_directory = rootd+  "labels/"
    output_images_directory = rootd+ "images_relabelled/"
    unlabelled_images_directory = rootd+ "images_unlabelled/"
    
    process_images(input_directory, output_labels_directory, output_images_directory, unlabelled_images_directory)
