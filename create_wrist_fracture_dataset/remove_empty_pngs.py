import os

root_dir = '/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/dcm/'

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.png'):
            # print(file)

            file_path = os.path.join(subdir, file)
            # print(file_path)
            if os.path.getsize(file_path) == 0:
                
                print(f"Removed empty file: {file_path}")
                # from IPython import embed; embed()
                os.remove(file_path)
