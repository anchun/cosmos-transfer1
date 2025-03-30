#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
echo "CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg1 --controlnet_specs assets/synthetic_humanoid_seg1.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg2 --controlnet_specs assets/synthetic_humanoid_seg2.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg3 --controlnet_specs assets/synthetic_humanoid_seg3.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg4 --controlnet_specs assets/synthetic_humanoid_seg4.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg5 --controlnet_specs assets/synthetic_humanoid_seg5.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg6 --controlnet_specs assets/synthetic_humanoid_seg6.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg7 --controlnet_specs assets/synthetic_humanoid_seg7.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_humanoid_seg8 --controlnet_specs assets/synthetic_humanoid_seg8.json
