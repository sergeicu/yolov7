# download wrist fracture data en-masse and process it. 

Steps are the following: 
1. Download images using montage 
2. Preprocess the reports 
3. Label images... 
4. 
5. 
6.
7. 



###################################
# download dicoms using dcmtk and convert to png 
###################################


# convert dicom to png 
python /home/ch215616/w/code/llm/experiments/yolov7/dicom-to-png/mritopng.py DX.1.3.46.670589.30.966169714234.3228.1641171169492 test.png

# test image for bounding box detection
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd yolov7/
f=/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/4912771/4912771/4087204/4_PA/
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f
