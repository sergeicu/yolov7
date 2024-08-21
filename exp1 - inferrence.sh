############################################
# inferrence - hello world 
############################################

# init 
ssh rayan 
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd YOLOv7-Bone-Fracture-Detection/



# test on our dataset 
weights=yolov7-p6-bonefracture.onnx
bchd=bch_data/one/1_5655171_images/
python inference_onnx.py --model-path $weights --img-path $bchd/1_AP.jpg --dst-path predictions/

# test on validation 
weights=yolov7-p6-bonefracture.onnx
dd=GRAZPEDWRI-DX_dataset/yolov5/images/test/
python inference_onnx.py --model-path $weights --img-path $dd/1879_0909421726_01_WRI-L2_F011.png --dst-path predictions/

# test on training 
weights=yolov7-p6-bonefracture.onnx
dd=GRAZPEDWRI-DX_dataset/yolov5/images/train/
python inference_onnx.py --model-path $weights --img-path $dd/0002_0354485735_01_WRI-R1_F012.png --dst-path predictions/


# compare ground truth - files
dd2=GRAZPEDWRI-DX_dataset/yolov5/labels/test/
cat predictions/1879_0909421726_01_WRI-L2_F011.txt
cat $dd2/1879_0909421726_01_WRI-L2_F011.txt



# compare ground truth - train - labels  
dd2=GRAZPEDWRI-DX_dataset/yolov5/labels/train/
cat predictions/0002_0354485735_01_WRI-R1_F012.txt
cat $dd2/0002_0354485735_01_WRI-R1_F012.txt


# run gui 



############################################
# INSTALL 
############################################
# enable the following 
ssh izmir 
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
python -m venv venv 
source venv/bin/activate 

# 
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd YOLOv7-Bone-Fracture-Detection/


# download the weights ls 
    #wget https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/download/trained-models/yolov7-p6-bonefracture.onnx
    cd ~/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/
    
    wget https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/download/trained-models/yolov7-p6-bonefracture.pt

# prepare data 
    # descriotion
    https://www.nature.com/articles/s41597-022-01328-z
    # download here 
    https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193

    # Use curl to download the file with a custom name
    curl -L -o downloaded_file.zip "https://figshare.com/ndownloader/files/34268891"

    curl -L -o downloaded_file2.zip https://figshare.com/ndownloader/files/34268864

    curl -L -o downloaded_file3.zip https://figshare.com/ndownloader/files/34268849

    curl -L -o downloaded_file4.zip https://figshare.com/ndownloader/files/34268828


# converting the dataset ()
# lots of manual steps - unzip the files in data_zip and move to GRAZPEDWRI-DX_dataset. 
    # the images from 4 different zip files in data_zip go to GRAZPEDWRI-DX_dataset/yolov5/images 
    # labels go from data_zip/folder_structure/yolov5/labels/ to GRAZPEDWRI-DX_dataset/yolov5/labels 
# then run split.py in YOLOv7-Bone-Fracture-Detection to split into train test validation sets


# install onnx 
install onnx 
    pip install onnx onnxruntime
# download onnx weights
    wget https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection/releases/download/trained-models/yolov7-p6-bonefracture.onnx




# for gui - does not work 
pip install PySide6