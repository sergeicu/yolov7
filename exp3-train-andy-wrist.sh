## Setup Instructions
ssh rayan  
conda activate llava-med  
cd ~/w/code/llm/experiments/yolov7/  
source venv/bin/activate   
cd yolov7/  


## Test train - on elbow dataset 
                    hyp=data/hyp.scratch.p6_bch_2.yaml
                    name=yolov7-p6-bonefracture-finetune-bch-elbow-v5-copy
                    cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
                    data=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
                    img=640

                    python train_aux.py --workers 8 --device 0 --batch-size 1 \
                        --data $data --img $img $img --cfg $cfg \
                        --weights yolov7-p6-bonefracture.pt --name $name --hyp $hyp


## Create config for new dataset 
        
        # one thing to consider - maybe keep exactly the same class structure and names - but just update things? 
        # need to create an algorithm that detects the correct bounding boxes around images. 
        


## Test train - on elbow dataset 
hyp=data/hyp.scratch.p6_bch_2.yaml
cp $hyp data/hyp.scratch.p6_andylabels294.yaml
name=yolov7-p6-bonefracture-finetune-bch-andy294-v1
cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
cp $cfg cfg/training/yolov7-w6_ch9_bonefracture-bch-andy294.yaml
data=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
cp $data data/yolov7-p6-bonefracture-finetune_bch_andy294.yaml
img=640

python train_aux.py --workers 8 --device 0 --batch-size 1 \
    --data $data --img $img $img --cfg $cfg \
    --weights yolov7-p6-bonefracture.pt --name $name --hyp $hyp    