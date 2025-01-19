
# checkout test.py before i made any changes to it to test_og2.py 
git show 55b90e111984dd85e7eed327e9ff271222aa8b82:test.py > test_og2.py

# checkout specific versions of utils/metrics.py and utils/loss.py
git checkout 55b90e111984dd85e7eed327e9ff271222aa8b82 -- utils/metrics.py 
git checkout 9cfa36dccd60dffea795de712d7984b1529574bc -- utils/loss.py 

# check when were these files changed 
git log --follow utils/metrics.py
git log --follow utils/loss.py




# now let's try a simple train. 

        srun -A bch -p bch-gpu -t 01:30:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --gres=gpu:1 --pty /bin/bash
        conda activate lisa   # activate lisa on e3 
        cd ~/w/code/llm/experiments/yolov7/  
        source venv/bin/activate   
        cd yolov7/         

        name=TEST_REMOVE
        data=data/andy294_4graz_all_histogram_matched_test_as_training.yaml
        # cfg=cfg/training/yolov7-w6_ch9_bonefracture.yaml
        cfg=cfg/training/yolov7-w6_ch9_bonefracture_fixed_0118.yaml
        img=640
        hyp=data/hyp.scratch.p6_andylabels294_v4_1.yaml # increased weight decay to original 0.0005
        grazval=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/YOLOv7-Bone-Fracture-Detection/GRAZPEDWRI-DX_dataset/yolov5/images/test_reduced
        # python train_aux.py --workers 1 --device 0 --batch-size 1 \
        python train_aux_0118_v2.py --workers 1 --device 0 --batch-size 1 \
            --data $data --img $img $img --cfg $cfg \
            --weights yolov7-p6-bonefracture.pt --name ${name}_bs16 --hyp $hyp \
            --project test_remove_mAP_tracking --entity sergeicu --finetune \
            --additional-val $grazval     


        # train without auxiliary lol 
        img=640
        name=TEST_REMOVE
        data=data/andy294_4graz_all_histogram_matched_test_as_training.yaml        
        hyp=data/hyp.scratch.p6_andylabels294_v4_1.yaml # increased weight decay to original 0.0005        
        cfg=cfg/training/yolov7_ch9_bonefracture.yaml
        python train_0118.py --workers 1 --device 0 --batch-size 1 \
            --data $data --img $img $img --cfg $cfg \
            --weights yolov7-p6-bonefracture.pt --name ${name}_bs16 --hyp $hyp \
            --project test_remove_mAP_tracking --entity sergeicu      

        # conclusion:
            # graz yolo paper incorrectly named the .pt file as yolov7-p6-bonefracture.pt
            # this suggested a model that had auxiliary detection layers, but it did not 
            # upon checking the config of the .pt file we realize that it is actually a standard yolov7 model
            # this explains the reason why our finetuning was not working. 
            # the model that we were trying to finetune was not loading correctly. 



        # lets try with pure test of test_og2.py 
        python test_og2.py --weights yolov7-p6-bonefracture.pt --data $data --img $img  --batch 1 --conf 0.001 --iou 0.65 --device 0 --name $name



# checkout the previous version of train_aux.py that was working before all the errors were discovered 
git show be08a22ec577d9efc9de6bf54bd21cd71b94895f:train_aux.py > train_aux_og.py
git show 55b90e111984dd85e7eed327e9ff271222aa8b82:test.py > test_og3.py

# 