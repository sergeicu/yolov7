#!/usr/bin/env python3
"""
Specialized YOLO Test Script for Single-Class Fracture Detection
Evaluates YOLO model performance specifically for fracture detection (class ID 3)
"""

import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized, TracedModel


def test_fracture(data,
                 weights=None,
                 batch_size=32,
                 imgsz=640,
                 conf_thres=0.001,
                 iou_thres=0.6,
                 save_json=False,
                 augment=False,
                 verbose=False,
                 model=None,
                 dataloader=None,
                 save_dir=Path(''),
                 save_txt=False,
                 save_hybrid=False,
                 save_conf=False,
                 plots=True,
                 half_precision=True,
                 trace=False,
                 is_coco=False,
                 v5_metric=False):
    """
    Test function specifically for fracture detection (single class)
    """
    # Initialize/load model and set device
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half precision
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    
    # For single-class fracture detection
    nc = 1  # number of classes (only fracture)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    # Initialize metrics
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {0: 'fracture'}  # Single class: fracture
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # Fracture class ID (should be 3 in your model, but we'll map it to 0 for evaluation)
    fracture_class_id = 3  # Original class ID in model
    eval_class_id = 0      # Class ID for evaluation (0 for single class)

    print(f"Evaluating fracture detection (class ID {fracture_class_id} -> {eval_class_id})")

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            # Filter predictions to only include fracture class
            filtered_out = []
            for pred in out:
                if len(pred):
                    # Keep only fracture predictions
                    mask = pred[:, 5] == fracture_class_id
                    filtered_pred = pred[mask].clone()
                    if len(filtered_pred):
                        # Remap class ID to 0 for evaluation
                        filtered_pred[:, 5] = eval_class_id
                    filtered_out.append(filtered_pred)
                else:
                    filtered_out.append(pred)
            out = filtered_out

            # Filter targets to only include fracture class
            if len(targets):
                mask = targets[:, 1] == fracture_class_id
                targets = targets[mask]
                if len(targets):
                    # Remap class ID to 0 for evaluation
                    targets[:, 1] = eval_class_id

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)

        # Calculate AP
        p_curve, r_curve, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric,
                                                         save_dir=save_dir, names=names)
        ap50 = ap[:, 0]
        ap = ap.mean(1)
        mp = p_curve.mean()
        mr = r_curve.mean()
        map50 = ap50.mean()
        map = ap.mean()

        # Create binary confusion matrix for fracture detection
        if plots:
            matrix_data = confusion_matrix.matrix
            if torch.is_tensor(matrix_data):
                matrix_data = matrix_data.cpu().numpy()
            
            # Binary confusion matrix: [TP, FN], [FP, TN]
            binary_matrix = np.zeros((2, 2))
            binary_matrix[0, 0] = matrix_data[0, 0]  # TP (fracture detected as fracture)
            binary_matrix[0, 1] = matrix_data[0, -1] if matrix_data.shape[1] > 1 else 0  # FN (fracture missed)
            binary_matrix[1, 0] = matrix_data[-1, 0] if matrix_data.shape[0] > 1 else 0  # FP (false fracture)
            binary_matrix[1, 1] = 0  # TN (not tracked in object detection)

            # Calculate metrics
            TP = float(binary_matrix[0, 0])
            FP = float(binary_matrix[1, 0])
            FN = float(binary_matrix[0, 1])

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Plot binary confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(binary_matrix, interpolation='nearest', cmap='Blues')
            plt.title('Fracture Detection Confusion Matrix')
            plt.ylabel('True')
            plt.xlabel('Predicted')
            
            # Add text annotations
            thresh = binary_matrix.max() / 2.
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, f'{binary_matrix[i, j]:.0f}',
                            horizontalalignment="center",
                            color="white" if binary_matrix[i, j] > thresh else "black")
            
            plt.xticks([0, 1], ['Fracture', 'No Fracture'])
            plt.yticks([0, 1], ['Fracture', 'No Fracture'])
            plt.savefig(str(save_dir / 'confusion_matrix.png'))
            plt.close()

            # Plot PR curve
            plt.figure(figsize=(8, 6))
            plt.plot(r_curve, p_curve, linewidth=2, color='blue')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve (Fracture Detection)')
            plt.grid(True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(str(save_dir / 'PR_curve.png'))
            plt.close()

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print('\nFracture Detection Results:')
    print(f"{'Class':20s}{'Images':>12s}{'Labels':>12s}{'P':>12s}{'R':>12s}{'mAP@.5':>12s}{'mAP@.5:.95':>12s}")
    print(pf % ('fracture', seen, nt.sum(), mp, mr, map50, map))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)
    if not training:
        print('\nSpeed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save results
    if not training:
        # Save summary to text file
        with open(save_dir / 'results.txt', 'w') as f:
            f.write('Fracture Detection Results:\n')
            f.write(f"{'Class':20s}{'Images':>12s}{'Labels':>12s}{'P':>12s}{'R':>12s}{'mAP@.5':>12s}{'mAP@.5:.95':>12s}\n")
            f.write(pf % ('fracture', seen, nt.sum(), mp, mr, map50, map) + '\n')
            
            if 'precision' in locals() and 'recall' in locals():
                f.write(f'\nBinary Classification Metrics:\n')
                f.write(f"TP: {TP:.0f}, FP: {FP:.0f}, FN: {FN:.0f}\n")
                f.write(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}\n")
        
        print(f"\nResults saved to {save_dir}")

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_fracture.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-p6-bonefracture.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)
    print(opt)

    if opt.task in ('train', 'val', 'test'):
        test_fracture(opt.data,
                     opt.weights,
                     opt.batch_size,
                     opt.img_size,
                     opt.conf_thres,
                     opt.iou_thres,
                     opt.save_json,
                     opt.augment,
                     opt.verbose,
                     save_txt=opt.save_txt | opt.save_hybrid,
                     save_hybrid=opt.save_hybrid,
                     save_conf=opt.save_conf,
                     trace=not opt.no_trace,
                     v5_metric=opt.v5_metric)
    elif opt.task == 'speed':
        for w in opt.weights:
            test_fracture(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric) 