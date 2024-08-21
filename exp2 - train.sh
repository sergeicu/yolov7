ssh rayan 
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd yolov7/



########################################################
# conversation with Shaoju
########################################################

# only use large model - 1280 
# for augmentations - try to crop IN (never zoom out) to improve performance 
# the main author works at bch - reach out to him? 
# is the dataset sufficient? 
    # one thing we can do is jsut train from scratch on those images and overfit like crazy - see if it converges well 
# how much are we 'finetuning' vs 'transfer learning' - shall we freeze some layers? 
# how is resizing done inside the model? (if image is 800 x 1200 - how does it resize to 1280 x 1280 )
# shall we freeze some layers instead of finetuning whole network (since we have so few images?)


########################################################
# llava finetuning (with grounding)
########################################################


########################################################
# learn what do all the numbers represent in results.txt output...
########################################################

# 


########################################################
# double check that our labels (.txt files) yield the same bounding boxes using actual yolo code 
########################################################

# 


########################################################
# finetuning grazped on bch-elbow  - v4 - only define hyperparameters that will tell the model NOT to use augmentation 
########################################################

# does not work - we must specify ALL parameters...  (it gives errors otherwise)

# training again with batch of 1 and image size 640 
hyp=data/hyp.scratch.p6_bch_2.yaml
name=yolov7-p6-bonefracture-finetune-bch-elbow-v5
cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
bonefracturedata=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
img=640
python train_aux.py --workers 8 --device 0 --batch-size 1 --data $data --img $img $img --cfg $cfg --weights 'yolov7-p6-.pt' --name $name --hyp $hyp
    # results in: 
        yolov7-p6-bonefracture-finetune-bch-elbow-v5
        



########################################################
# finetuning grazped on bch-elbow  - v4 - no hyperparameters used at all 
########################################################
# no hyperparameters at all - all defaults 
# NOT good - it performs too much augmentation... 


# training again with batch of 1 and image size 640 
name=yolov7-p6-bonefracture-finetune-bch-elbow-v4
cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
data=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
img=640
python train_aux.py --workers 8 --device 0 --batch-size 1 --data $data --img $img $img --cfg $cfg --weights 'yolov7-p6-bonefracture.pt' --name $name #--hyp $hyp
    # results in: 
        yolov7-p6-bonefracture-finetune-bch-elbow-v4
        



########################################################
# finetuning grazped on bch-elbow  - v3 - no augmentations to dataset 
########################################################
# no augmentations 

# training again with batch of 1 and image size 640 
hyp=data/hyp.scratch.p6_bch_1.yaml
name=yolov7-p6-bonefracture-finetune-bch-elbow-v3
cfg=cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml
data=data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml
img=640
python train_aux.py --workers 8 --device 0 --batch-size 1 --data $data --img $img $img --cfg $cfg --weights 'yolov7-p6-bonefracture.pt' --name $name --hyp $hyp
    # results in: 
        yolov7-p6-bonefracture-finetune-bch-elbow-v3
        


########################################################
# finetuning grazped on bch-elbow  - v2 - separate train and valid and test sets 
########################################################

# NB # here we use the hyperparameter file with LOTS of augmentations. 
     # at the same time we fixed the issue with number of classes (just 1!). we also fixed the issue with train-val-test size. 
     # finally - we set the batch to 1.



# we separate our datset of 217 images into 180 (82%) train, 25 (11%) valid and 12 (5%) test images 
python /home/ch215616/w/code/llm/experiments/yolov7/split_train_val_test.py


# training again with batch of 1 and image size 640 
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_bch_elbow.yaml --img 640 640 --cfg cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-bch-elbow-v2 --hyp data/hyp.scratch.p6.yaml
    # results in: 
        yolov7-p6-bonefracture-finetune-bch-elbow-v23

    # here we use the hyperparameter file with LOTS of augmentations.

# here is what the results file shows: 
    # In the provided code, the terms represent the following:

    # Class: The class of objects being detected or classified.
    # Images: The number of images processed during testing or validation.
    # Labels: The number of object instances labeled in the dataset.
    # P (Precision): The precision of the model, which is the ratio of correctly predicted positive observations to the total predicted positive observations.
    # R (Recall): The recall of the model, which is the ratio of correctly predicted positive observations to all observations in the actual class.
    # mAP@.5: The mean Average Precision at an Intersection over Union (IoU) threshold of 0.5. It is a standard metric in object detection.
    # mAP@.5:.95: The mean Average Precision averaged over multiple IoU thresholds from 0.5 to 0.95 with a step of 0.05. This provides a more comprehensive evaluation of the model's performance across different IoU thresholds.
# example of results file: 
    cat /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v23/results.txt
    epoch       ? memory?                                                          precision    recall   mAP@.5     mAP@.5:.95   
    11/299      2.2G    0.0402  0.009244         0   0.04944         1       640 0.0005973      0.16     0.0001043  1.309e-05   0.06285   0.00716         0


########################################################
# run grazped-bch-elbow-v1 on test case 
########################################################

# WARNING: model was 

# inferrence on bch-elbow (no separate test set was saved)
# ZERO cases identified 
f=elbow_fracture/few_images_train_set/
python detect.py --weights runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v16/weights/best.pt --conf 0.25 --img-size 1280 --source $f
    # /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/detect/exp8


# inferrence with grazped (for comparison)
    # ONE out of 3 fractures identified. one is super obvious. 
    f=elbow_fracture/few_images_train_set/
    python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $f
    # /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/detect/exp9

    # Image size is half... same success rate 
    python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f
    # /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/detect/exp10
    # change conf to 1% (0.01) -> very visible what is fracture and what is not! 
    python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.01 --img-size 640 --source $f
    # /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/detect/exp10



# new models 
f=elbow_fracture/few_images_train_set/
python detect.py --weights runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v16/weights/best.pt --conf 0.25 --img-size 1280 --source $f
    # /home/ch215616/w/code/llm/experiments/yolov7/yolov7/runs/detect/exp8


########################################################
# finetuning grazped on bch-elbow  - 217 images in total 
########################################################

# NB this training script has the same train and valid cases - 
# so after every epoch it calculates accuracy on ALL training images....
# it takes around 11mins to perform validation (which is a long time) -> 724 sets... (why 724 - no one knows )


# prepare correct paths 
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/elbow_fracture/
mkdir -p final/images/ final/labels
ln -sf $PWD/images_unlabelled/ final/images/train
ln -sf $PWD/images_unlabelled/ final/images/valid
ln -sf $PWD/labels/ final/labels/train
ln -sf $PWD/labels/ final/labels/valid


# [FINAL WORKING] - with hyperparameters, image 1280, batch 6
python train_aux.py --workers 8 --device 0 --batch-size 6 --data data/yolov7-p6-bonefracture-finetune_bch_elbow_nosplit.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6_ch9_bonefracture-bch-elbow.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-bch-elbow-v1 --hyp data/hyp.scratch.p6.yaml
    # results saved to 
    # runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v16

    # weights saved to: 
    runs/train/yolov7-p6-bonefracture-finetune-bch-elbow-v16/



########################################################
# finetuning grazped on grazped
########################################################

# [FINAL WORKING] - with hyperparameters, image 1280, batch 6
python train_aux.py --workers 8 --device 0 --batch-size 6 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6_ch9_bonefracture.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-v2 --hyp data/hyp.scratch.p6.yaml


########################################################
# debugging finetuning on grazped vs original 
########################################################

# [works!!! FIXED v4] WITH hyperparameters, image 1280 AND max batch of 6
python train_aux.py --workers 8 --device 0 --batch-size 6 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6_ch9_bonefracture.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-v2 --hyp data/hyp.scratch.p6.yaml

# [works!!! FIXED v3] WITH hyperparameters defined (image size is 1280 1280)
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6_ch9_bonefracture.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-v2 --hyp data/hyp.scratch.p6.yaml

# [works!!! FIXED v2] WITH hyperparameters defined (image size is still 640)
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7-w6_ch9_bonefracture.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-v2 --hyp data/hyp.scratch.p6.yaml


# [works!!! FIXED] let's try to create a new cfg file for grazped for p6 model - and only change number of layers  -> without hyperparameters defined
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7-w6_ch9_bonefracture.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-v2 

# here is the reason why 
    # i.e. grazped people included p5 file for training, not p6 file (!) 
    # these files are exactly the same except for the number of channels  -> p5 model -> i.e. grazped people included p5 file, not p6 file 
yolov7/cfg/training/yolov7.yaml
yolov7/cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml 


# [DO NOT work]  grazped -> does not work -> the cfg file that grazped shared is incorrect - it is for p5 model, but their finetuned weights (on pediatric xray wrist fractures) shared are for p6 model 
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune #--hyp data/hyp.scratch.yolov7-p6-bonefracture.yaml

# [works] original p5 (with original yolov7 weights but grazped dataset ) -> works 
python train.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7-w5 #--hyp data/hyp.scratch.p5.yaml

# [works]  original p5 (modified as similar as possible to grazped - v1) (with original yolov7 weights and coco dataset ) -> works 
python train.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7-w5 #--hyp data/hyp.scratch.p5.yaml

# [works]  original p6 (modified as similar as possible to grazped - v1) (with original yolov7 weights and coco dataset) - works 
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7-w6.yaml --weights 'yolov7.pt' --name yolov7-w6 #--hyp data/hyp.scratch.p6.yaml

# [works]  original p6 - works (with original yolov7 weights and coco dataset)  - works
python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml

########################################################
# Prepare grazped dataset for finetuning 
########################################################

cd ~/w/code/llm/experiments/yolov7/yolov7
# create cfg file ref 
ln -sf /home/ch215616/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/yolov7_cfg.yaml cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml
# create data file ref 
cp /home/ch215616/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/yolov7_data.yaml data/yolov7-p6-bonefracture-finetune_data_BAD.yaml
# manually edit the paths in the file above and re-save it as data/yolov7-p6-bonefracture-finetune_data.yaml
... [manual edit of yaml file and paths]
# create symbolic links to model 
ln -sf /home/ch215616/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/yolov7-p6-bonefracture.pt yolov7-p6-bonefracture.pt


# for reference - here are the labels 
    # coco labels: 
    cat coco/labels/val2017/000000189698.txt
    # grazpedwri labels: 
    cat ../YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/labels/valid/1984_0326659031_05_WRI-L1_F011.txt

    # tasks 
        # 1. data/coco.yaml -> data/yolov7-p6-bonefracture-finetune_data.yaml
        # 2. cfg/training/yolov7-w6.yaml -> cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml
        # 3. data/coco.yaml -> data/yolov7-p6-bonefracture-finetune_data.yaml
        # 4. [use with defaults] data/hyp.scratch.p6.yaml -> data/hyp.scratch.yolov7-p6-bonefracture.yaml 
        # 5. 1280 1280 -> 640 640  -> this may or may not be necessay 

# NB: when running the finetunning for the first time - we need to create cache files for our dataset 
# train without hyp.scratch.p6.yaml
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune #--hyp data/hyp.scratch.yolov7-p6-bonefracture.yaml

# train WITH hyp.scratch.p6.yaml (default from coco)
python train_aux.py --workers 8 --device 0 --batch-size 1 --data data/yolov7-p6-bonefracture-finetune_data.yaml --img 640 640 --cfg cfg/training/yolov7-p6-bonefracture-finetune_cfg.yaml --weights 'yolov7-p6-bonefracture.pt' --name yolov7-p6-bonefracture-finetune-with-hyp-scratch-p6 --hyp data/hyp.scratch.p6.yaml

# NOTE: we must wait for cache to finish 
/home/ch215616/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/labels/train.cache
# the original cache for coco is here: 




########################################################
# Basic YOLO inferrence ON BCH wrist fracture with GRAZPEDWRI model
########################################################

# [x] inferrence - pediatric xwrist at bch  - all 
d=more_wrist_fractures/
f=$d/
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f
    file runs/detect/exp6/260*_bilateral_wrist_fracture_7.png
    # detects fractures and 'periostealreaction' (low prob) and 'metal' (low prob)



# [x] inferrence - pediatric xwrist at bch  - one 
d=more_wrist_fractures/
f=$d/26080878_bilateral_wrist_fracture_1.png
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f
    file runs/detect/exp5/260*_bilateral_wrist_fracture_1.png
    # 592 x 1270


# how i found the files - 
https://bch.nuancempower.com/search/rad?q=wrist+fracture


########################################################
# Basic YOLO inferrence ON GRAZPEDWRI-DX data and our elbow data
########################################################

# [x] inferrence - pediatric xwrist 
f=../YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/images/test/6060_0587040349_04_WRI-L2_M015.png
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f #inference/images/horses.jpg
    # > runs/detect/exp2/6060_0587040349_04_WRI-L2_M015.png
    file runs/detect/exp2/6060_0587040349_04_WRI-L2_M015.png
    # 592 x 1270

# [x] inferrence - elbow
f=../YOLOv7-Bone-Fracture-Detection/bch_data/one/1_*_images/1_AP.jpg
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 640 --source $f
    # > runs/detect/exp3/1_AP.jpg
    file runs/detect/exp3/1_AP.jpg
    # 822x1014

# note the actual sizes of files: 
file ../YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/images/test/6060_0587040349_04_WRI-L2_M015.png
    # 592 x 1270,
file ../YOLOv7-Bone-Fracture-Detection/bch_data/one/1_*_images/1_AP.jpg
    # 822x1014

########################################################
# Basic YOLO RUN ON COCO DATA
########################################################


# [x] test 
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
#> 
            # Namespace(weights=['yolov7.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.001, iou_thres=0.65, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7_640_val', exist_ok=False, no_trace=False, v5_metric=False)
            # YOLOR 🚀 2024-7-17 torch 2.3.1+cu121 CUDA:0 (NVIDIA TITAN Xp, 12186.875MB)

            # Fusing layers... 
            # RepConv.fuse_repvgg_block
            # RepConv.fuse_repvgg_block
            # RepConv.fuse_repvgg_block
            # /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/venv/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3587.)
            # return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
            # Model Summary: 306 layers, 36905341 parameters, 36905341 gradients, 104.5 GFLOPS
            # Convert model to Traced-model... 
            # /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/venv/lib/python3.10/site-packages/torch/nn/modules/module.py:810: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
            # param_grad = param.grad
            # traced_script_module saved! 
            # model is traced! 

            # val: Scanning 'coco/val2017' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█| 5000/5
            # val: New cache created: coco/val2017.cache
            #             Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|█| 157/157 
            #                 all        5000       36335       0.724       0.635       0.691       0.497
            # Speed: 13.2/2.1/15.3 ms inference/NMS/total per 640x640 image at batch-size 32

            # Evaluating pycocotools mAP... saving runs/test/yolov7_640_val/yolov7_predictions.json...
            # pycocotools unable to run: No module named 'pycocotools'
            # /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/venv/lib/python3.10/site-packages/torch/nn/modules/module.py:810: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
            # param_grad = param.grad
            # Results saved to runs/test/yolov7_640_val


# [x] inferrence on image 
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
#> 
            # Namespace(weights=['yolov7.pt'], source='inference/images/horses.jpg', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
            # YOLOR 🚀 2024-7-17 torch 2.3.1+cu121 CUDA:0 (NVIDIA TITAN Xp, 12186.875MB)

            # Fusing layers... 
            # RepConv.fuse_repvgg_block
            # RepConv.fuse_repvgg_block
            # RepConv.fuse_repvgg_block
            # /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/venv/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3587.)
            # return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
            # Model Summary: 306 layers, 36905341 parameters, 6652669 gradients, 104.5 GFLOPS
            # Convert model to Traced-model... 
            # traced_script_module saved! 
            # model is traced! 

            # 5 horses, Done. (39.3ms) Inference, (531.3ms) NMS
            # The image with the result is saved in: runs/detect/exp/horses.jpg
            # Done. (2.102s)



# [x] train p6 models -> reduced batch size to 4 instead of 16 
python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
#> 
            # YOLOR 🚀 2024-7-17 torch 2.3.1+cu121 CUDA:0 (NVIDIA TITAN Xp, 12186.875MB)

            # Namespace(weights='', cfg='cfg/training/yolov7-w6.yaml', data='data/coco.yaml', hyp='data/hyp.scratch.p6.yaml', epochs=300, batch_size=16, img_size=[1280, 1280], rect=False, resume=False, nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='', cache_images=False, image_weights=False, device='0', multi_scale=False, single_cls=False, adam=False, sync_bn=False, local_rank=-1, workers=8, project='runs/train', entity=None, name='yolov7-w6', exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias='latest', v5_metric=False, world_size=1, global_rank=-1, save_dir='runs/train/yolov7-w62', total_batch_size=16)
            # tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
            # hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.3, cls_pw=1.0, obj=0.7, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, paste_in=0.15, loss_ota=1
            # wandb: Install Weights & Biases for YOLOR logging with 'pip install wandb' (recommended)


            #                 from  n    params  module                                  arguments                     
            # 0                -1  1         0  models.common.ReOrg                     []                            
            # 1                -1  1      7040  models.common.Conv                      [12, 64, 3, 1]                
            # 2                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
            # 3                -1  1      8320  models.common.Conv                      [128, 64, 1, 1]               
            # 4                -2  1      8320  models.common.Conv                      [128, 64, 1, 1]               
            # 5                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
            # 6                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
            # 7                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]
            # ... 


            # 122[114, 115, 116, 117, 118, 119, 120, 121]  1   1474420  models.yolo.IAuxDetect                  [80, [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542], [436, 615, 739, 380, 925, 792]], [256, 512, 768, 1024, 320, 640, 960, 1280]]
            # /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/venv/lib/python3.10/site-packages/torch/functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3587.)
            #   return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
            # Model Summary: 477 layers, 82312436 parameters, 82312436 gradients, 105.5 GFLOPS

            # Scaled weight_decay = 0.0005
            # Optimizer groups: 115 .bias, 115 conv.weight, 115 other
            # train: Scanning 'coco/train2017' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█| 50
            # train: New cache created: coco/train2017.cache
            # val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█| 

            # autoanchor: Analyzing anchors... anchors/target = 5.57, Best Possible Recall (BPR) = 0.9943
            # Image sizes 1280 train, 1280 test
            # Using 8 dataloader workers
            # Logging results to runs/train/yolov7-w62
            # Starting training for 300 epochs...

            #      Epoch   gpu_mem       box       obj       cls     total    labels  img_size
            #   0%|                     


#> 

            # had to make some changes in this file (on lines 1404 and 1557): /home/ch215616/w/code/llm/experiments/yolov7/yolov7/utils/loss.py
            # OLD 
            from_which_layer = from_which_layer[fg_mask_inboxes]
            # NEW 
            from_which_layer = from_which_layer.to(fg_mask_inboxes.device)[fg_mask_inboxes]             


# [-] finetune p6 models
    # will take us 90 hours to complete training - 300 epochs * 20 mins per epoch 
    # with batch size of 4 and VALIDATION dataset (not even training!...)
    # 
    # 
    # [todo - need to define custom yamls files properly]
    # here is simple edited version - data/ and cfg/train point to the same as original coco training. 
    # however we include path to weights! so that weights are starting not from scratch - --weights 'yolov7.pt'

    # here is original definition of finetuning as it should be instantiated
    # python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml    

    # here is proof of concept 
python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights 'yolov7.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.p6.yaml





















########################################################
# Finetune our dataset on GRAZPEDWRI model 
########################################################

# data yaml 
data/yolov7-p6-bonefracture-finetune_bch_data.yaml

########################################################
# Convert our ENTIRE dataset to YOLO format (with .text files)
########################################################

# here is the file that identifies the coordinates of red boxes by pixels... 

# here is example of coordinate file 
cat /home/ch215616/w/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/labels/test/4435_0479463126_02_WRI-R2_M012.txt
    # 8 0.938301 0.843496 0.123397 0.090244
    # 3 0.504006 0.272764 0.254808 0.078862
    # 3 0.532853 0.346341 0.120192 0.043902

# files should be in 
/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/elbow_fracture/final/

# images with the following
images/valid/
images/train 
images/test

# labels with the following 
labels/valid/
labels/train/
labels/test/

