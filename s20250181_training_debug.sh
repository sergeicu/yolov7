#!/bin/bash

# ===== YOLOv7 Training vs Testing Debug Summary =====

# Initial Problem:
# - Different mAP results between direct testing and validation during training
# - Same weights file (yolov7-p6-bonefracture.pt) gave different results
# - Testing showed 169 correct predictions out of 2099
# - Training validation showed 0 correct predictions out of 9000

# Key Findings:

1. Architecture Mismatch
   - Initially tried to define new architecture via config file (yolov7-w6_ch9_bonefracture.yaml)
   - Config used IAuxDetect while pretrained model used IDetect
   - Only 152 out of 668 weights were being loaded due to architecture mismatch
   - Different number of detection layers (8 vs 3) caused index out of range errors

2. Weight Loading Strategy
   - Original approach: Define architecture in config + load weights
   - Better approach: Load complete model architecture from pretrained file
   - Used attempt_load() to preserve exact architecture and weights
   - Avoided config file mismatches entirely

3. Checkpoint Handling
   - Discovered checkpoint didn't include optimizer state
   - Created fresh checkpoint structure with necessary keys:
     * epoch
     * best_fitness
     * training_results
     * model
     * optimizer (set to None for fresh initialization)
     * wandb_id

4. Key Solutions:
   a. Abandoned config-based architecture definition
   b. Used attempt_load() to maintain exact pretrained architecture
   c. Enabled gradients for all parameters explicitly
   d. Created proper checkpoint structure for training

5. Important Learnings:
   - When finetuning, prefer loading complete model architecture over defining new one
   - Check checkpoint contents before assuming optimizer state availability
   - Verify architecture compatibility (IDetect vs IAuxDetect)
   - Monitor number of weights being transferred during loading
   - Ensure proper gradient computation setup for finetuning

# Final Solution:
# - Load complete model architecture using attempt_load()
# - Enable gradients for all parameters
# - Initialize fresh optimizer state
# - Maintain consistent architecture between training and testing

# This approach ensures:
# 1. Exact same architecture in training and testing
# 2. All pretrained weights are properly utilized
# 3. Proper finetuning setup with gradients enabled
# 4. Clean optimizer initialization 