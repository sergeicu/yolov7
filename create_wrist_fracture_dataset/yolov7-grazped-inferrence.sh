ssh rayan 
conda activate llava-med
cd ~/w/code/llm/experiments/yolov7/
source venv/bin/activate 
cd yolov7/

# create link 
ln -sf /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs/ wrist_fracture_dataset

# run - basic 
f=wrist_fracture_dataset/
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $f


# run - save txt files with confidence score 
o=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v1
f=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/wrist_fracture_dataset/26375050-26375050-2.png
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $f --save-txt --save-conf --project $o --name run1


# run - save txt files with confidence score - 2 
o=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v1
fo=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/wrist_fracture_dataset/
cp $fo/26375050-26375050-2.png $o 
cp $fo/26503356-2_Lateral-1.png $o 
cp $fo/26737967-6_Lateral-2.png $o 
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $o/ --save-txt --save-conf --project $o --name run1


# explanation of input options 
https://chatgpt.com/share/770ece74-2490-4a9b-8f5c-7484b0dcb604

# run - full run 
o=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2
fo=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/wrist_fracture_dataset/
python detect.py --weights yolov7-p6-bonefracture.pt --conf 0.25 --img-size 1280 --source $fo/ --save-txt --save-conf --project $o --name run1
