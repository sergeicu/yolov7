import os
import shutil
from glob import glob
import logging

# Define paths
fixed_png_dir = '/home/ch215616/w/code/llm/experiments/yolov7/yolov7/ALL/run1_fixed_pngs_only'
fixed_label_dir = os.path.join(fixed_png_dir, 'labels')
run1_dir = '/home/ch215616/w/code/llm/experiments/yolov7/yolov7/ALL/run1'
run1_label_dir = os.path.join(run1_dir, 'labels')

# Set up logging
log_file = os.path.join(fixed_png_dir, 'fixing_inverted_pngs.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 1. Get all PNGs in run1_fixed_pngs_only
fixed_pngs = glob(os.path.join(fixed_png_dir, '*.png'))

for fixed_png_path in fixed_pngs:
    basename = os.path.splitext(os.path.basename(fixed_png_path))[0]

    # 2. Replace PNG in run1 with the fixed one
    run1_png_path = os.path.join(run1_dir, f'{basename}.png')
    if os.path.exists(run1_png_path):
        shutil.copy2(fixed_png_path, run1_png_path)
        logging.info(f'Replaced: {run1_png_path}')
    else:
        logging.warning(f'{run1_png_path} does not exist in run1, skipping.')

    # 3. Handle corresponding .txt label
    fixed_txt_path = os.path.join(fixed_label_dir, f'{basename}.txt')
    run1_txt_path = os.path.join(run1_label_dir, f'{basename}.txt')

    if os.path.exists(fixed_txt_path):
        # Copy label from fixed to run1/labels
        shutil.copy2(fixed_txt_path, run1_txt_path)
        logging.info(f'Copied label: {run1_txt_path}')
    else:
        # If label exists in run1/labels, remove it
        if os.path.exists(run1_txt_path):
            os.remove(run1_txt_path)
            logging.info(f'Removed label: {run1_txt_path}')

logging.info('Done.')