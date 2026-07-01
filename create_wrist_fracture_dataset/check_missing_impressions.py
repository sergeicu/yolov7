import os 

f='/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/missing_impressions.txt'




with open(f, 'r') as file:
    lines = file.readlines()
    
d='/home/ch215616/w/code/llm/experiments/yolov7/wrist_fracture_dataset/reports/'
    
for i,line in enumerate(lines): 
    line = line.replace('\n','')
    
    filename= d+line  
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines2 = f.readlines()
        from IPython import embed; embed()    
    
    
    
    if i >10:
        break