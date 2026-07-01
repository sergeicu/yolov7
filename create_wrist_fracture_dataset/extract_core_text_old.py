import os
import re

def extract_core_text(input_folder, output_folder, log_file):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize log file
    with open(log_file, 'w') as log:
        log.write("Files with missing markers:\n")
    
    # Process all text files in the input folder
    for filename in os.listdir(input_folder):
        
        if filename.endswith('.txt'):
            print(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r') as file:
                content = file.read()
            
            # Extract core text using regex
            match = re.search(r'IMPRESSION:\s*(.*?)\s*END OF IMPRESSION', content, re.DOTALL)
            
            if match:
                core_text = match.group(1).strip()
                # Write core text to output file
                with open(output_path, 'w') as out_file:
                    out_file.write(core_text)
            else:
                # Log files with missing markers
                with open(log_file, 'a') as log:
                    log.write(f"{filename}\n")

#Example usage
input_folder = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/reports/'
output_folder = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/reports_impressions/'
log_file = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/missing_impressions.txt'
extract_core_text(input_folder, output_folder, log_file)