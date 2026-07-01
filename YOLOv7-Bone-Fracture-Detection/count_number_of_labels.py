
#https://chatgpt.com/share/a573621e-90af-4f78-9f30-89f80d01e994


"""
Reads a text file line by line and identifies the first number 
between 0 and 8. The count for the identified number is incremented 
in a global counter.

Args:
    file_path (str): The path to the .txt file to be processed.
"""

import os
import re
from collections import Counter

# Path to the directory containing the .txt files
directory = '/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/labels/valid/'

# Initialize a Counter for the numbers 0-8
counter = Counter({str(i): 0 for i in range(9)})

# Function to process each file
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'\b[0-8]\b', line)
            if match:
                number = match.group(0)
                counter[number] += 1
                break

# Continuously process the files and print the counter
while True:
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            process_file(os.path.join(directory, filename))
    
        # Print the counter in the desired format
        print(' '.join([f'{i}:{counter[str(i)]}' for i in range(9)]))
