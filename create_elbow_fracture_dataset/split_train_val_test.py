import os
import shutil
import random

# Paths
rootd='/home/ch215616/w/code/llm/experiments/yolov7/yolov7/elbow_fracture/'
image_folder = rootd+'images_unlabelled/'
label_folder = rootd+'labels'
output_train_folder = rootd + 'final2/images/' +'train/'
output_val_folder = rootd + 'final2/images/' + 'validation/'
output_test_folder = rootd + 'final2/images/' + 'test/'

# Create directories if they don't exist
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_val_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

os.makedirs(output_train_folder.replace('/images/', '/labels/'), exist_ok=True)
os.makedirs(output_val_folder.replace('/images/', '/labels/'), exist_ok=True)
os.makedirs(output_test_folder.replace('/images/', '/labels/'), exist_ok=True)


# Get list of all images
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Shuffle the list to randomize
random.shuffle(image_files)

# Split into train, validation, and test
train_files = image_files[:180]
val_files = image_files[180:205]
test_files = image_files[205:]

# Function to move files
def copy_files(file_list, dest_folder):
    for file_name in file_list:
        image_path = os.path.join(image_folder, file_name)
        label_name = file_name.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_name)
        
        dest_folder2 = dest_folder.replace('/images/', '/labels/')
        
        shutil.copyfile(image_path, os.path.join(dest_folder, file_name))
        shutil.copyfile(label_path, os.path.join(dest_folder2, label_name))

# Move files to respective folders
copy_files(train_files, output_train_folder)
copy_files(val_files, output_val_folder)
copy_files(test_files, output_test_folder)

print("Files have been successfully copied.")
print(output_train_folder)
print(output_val_folder
print(output_test_folder)
