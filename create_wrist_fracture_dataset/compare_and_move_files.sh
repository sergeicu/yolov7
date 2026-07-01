python3 compare_and_move_files.py --root-only --log-file remove.json 
python3 compare_and_move_files.py --root-only --log-file remove.json --delete

rpath="code/llm/experiments/yolov7/create_elbow_fracture_dataset"
python3 compare_and_move_files.py --path $rpath --root-only --log-file comparison.json



    # Run comparison and generate log
    directories=($(ls -d ~/w/code/llm/experiments/yolov7/*/ | sed 's|^/home/ch215616/w/||' | sed 's/\/$//'))
    echo ${directories[@]}
    for dir in "${directories[@]}"; do
        python3 compare_and_move_files.py --path "$dir" --root-only --log-file comparison.json
        python3 compare_and_move_files.py --path "$dir" --root-only --log-file comparison.json --delete 
    done 
    
    