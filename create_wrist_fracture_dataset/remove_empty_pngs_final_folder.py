import os
import glob 

root_dir = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs/'



files = glob.glob(root_dir + "/*.png")
for file in files:
    if os.path.getsize(file) == 0:
        print(f"{file}")
        os.remove(file)
