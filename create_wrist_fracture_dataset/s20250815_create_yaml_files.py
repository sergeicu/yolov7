#!/usr/bin/env python3
"""
Python script to create YAML configuration files for YOLO training.
This script creates YAML files for the four wrist fracture classification datasets.
"""

import os
import yaml


def create_yaml_file(yaml_path, dataset_config):
    """Create a YAML configuration file for YOLO training."""
    yaml_content = {
        'train': dataset_config['train_path'],
        'val': dataset_config['val_path'], 
        'test': dataset_config['test_path'],
        'nc': dataset_config['num_classes'],
        'names': dataset_config['class_names']
    }
    
    # Add comment header and dump with flow style for lists
    yaml_content = f"# {dataset_config['name']} Dataset\n" + yaml.dump(yaml_content, default_flow_style=False, sort_keys=False, default_style=None)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created YAML file: {yaml_path}")


def main():
    # Define the base path for YAML files
    yaml_base_path = "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/data/"
    base_dataset_dir = "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs_aug2025/"
    
    # Dataset configurations
    dataset_configs = {
        'anatomical_regions_filter': {
            'name': 'Anatomical Regions Filter Dataset',
            'train_path': f"{base_dataset_dir}/anatomical_regions_01fracture/images/train",
            'val_path': f"{base_dataset_dir}/anatomical_regions_01fracture/images/val",
            'test_path': f"{base_dataset_dir}/anatomical_regions_01fracture/images/test",
            'num_classes': 5,
            'class_names': ["distal_radius_shaft", "distal_ulna_shaft", "ulnar_styloid", "scaphoid", "no_fracture"]
        },
        'classification_filter': {
            'name': 'Classification Filter Dataset',
            'train_path': f"{base_dataset_dir}/classification_01fracture/images/train",
            'val_path': f"{base_dataset_dir}/classification_01fracture/images/val",
            'test_path': f"{base_dataset_dir}/classification_01fracture/images/test",
            'num_classes': 6,
            'class_names': ["transverse", "salter_harris_II", "buckle", "comminuted", "avulsion", "no_fracture"]
        },
        'healing_filter': {
            'name': 'Healing Filter Dataset',
            'train_path': f"{base_dataset_dir}/healing_01fracture/images/train",
            'val_path': f"{base_dataset_dir}/healing_01fracture/images/val",
            'test_path': f"{base_dataset_dir}/healing_01fracture/images/test",
            'num_classes': 5,
            'class_names': ["healing", "acute", "healed", "nonunion", "no_fracture"]
        },
        'alignment_filter': {
            'name': 'Alignment Filter Dataset',
            'train_path': f"{base_dataset_dir}/alignment_01fracture/images/train",
            'val_path': f"{base_dataset_dir}/alignment_01fracture/images/val",
            'test_path': f"{base_dataset_dir}/alignment_01fracture/images/test",
            'num_classes': 4,
            'class_names': ["well_aligned", "acceptable_alignment", "poor_alignment", "no_fracture"]
        }
    }
    
    print("Creating YAML files for YOLO training...")
    print(f"Output directory: {yaml_base_path}")
    print()
    
    # Create YAML files for each dataset
    for config_name, config in dataset_configs.items():
        yaml_filename = f"s20250815_{config_name}.yaml"
        yaml_path = os.path.join(yaml_base_path, yaml_filename)
        create_yaml_file(yaml_path, config)
    
    print("\n" + "=" * 60)
    print("YAML files created successfully:")
    print("=" * 60)
    print("  - anatomical_regions_filter.yaml (5 classes: distal_radius_shaft, distal_ulna_shaft, ulnar_styloid, scaphoid, no_fracture)")
    print("  - classification_filter.yaml (6 classes: transverse, salter_harris_II, buckle, comminuted, avulsion, no_fracture)")
    print("  - healing_filter.yaml (5 classes: healing, acute, healed, nonunion, no_fracture)")
    print("  - alignment_filter.yaml (4 classes: well_aligned, acceptable_alignment, poor_alignment, no_fracture)")
    print()
    print("YAML files are ready for YOLO training!")
    print("You can now use these YAML files with your YOLO training script.")


if __name__ == "__main__":
    main() 