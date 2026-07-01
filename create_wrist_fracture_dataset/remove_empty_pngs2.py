import os
import glob 

root_dir = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm/'


folders = glob.glob(root_dir + "[0-9]*/")
print("got folders")
for folder in folders:
    files = glob.glob(folder + "/*.png")
    for file in files:
        if file.endswith('.png'):

            if os.path.getsize(file) == 0:
                
                print(f"{file}")
                # from IPython import embed; embed()
                os.remove(file)
