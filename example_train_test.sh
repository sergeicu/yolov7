# init 
    srun -A bch -p bch-gpu -t 01:30:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --gres=gpu:1 --pty /bin/bash
    conda activate lisa   # activate lisa on e3 
    cd ~/w/code/llm/experiments/yolov7/  
    source venv/bin/activate   
    cd yolov7/         

# train example 
    img=640
    name=TEST_REMOVE
    data=data/andy294_4graz_all_histogram_matched_test_as_training.yaml        
    hyp=data/hyp.scratch.p6_andylabels294_v4_1.yaml # increased weight decay to original 0.0005        
    cfg=cfg/training/yolov7_ch9_bonefracture.yaml
    python train_0118.py --workers 1 --device 0 --batch-size 1 \
        --data $data --img $img $img --cfg $cfg \
        --weights yolov7-p6-bonefracture.pt --name ${name}_bs16 --hyp $hyp \
        --project test_remove_mAP_tracking --entity sergeicu      

# test example 
    name=TEST_REMOVE
    python test_0118.py --weights yolov7-p6-bonefracture.pt --data $data --img $img  --batch 1 --conf 0.001 --iou 0.65 --device 0 --name $name --project test_remove_mAP_tracking


# helper: look at git history 
    # check when were these files changed 
    git log --follow utils/metrics.py
    git log --follow utils/loss.py

    # checkout test.py before i made any changes to it to test_og2.py 
    git show 55b90e111984dd85e7eed327e9ff271222aa8b82:test.py > test_og2.py

    # checkout specific versions of utils/metrics.py and utils/loss.py
    git checkout 55b90e111984dd85e7eed327e9ff271222aa8b82 -- utils/metrics.py 
    git checkout 9cfa36dccd60dffea795de712d7984b1529574bc -- utils/loss.py 


