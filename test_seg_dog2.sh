#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
echo "CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg9 --controlnet_specs assets/synthetic_Dog_seg9.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg10 --controlnet_specs assets/synthetic_Dog_seg10.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg11 --controlnet_specs assets/synthetic_Dog_seg11.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg12 --controlnet_specs assets/synthetic_Dog_seg12.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg13 --controlnet_specs assets/synthetic_Dog_seg13.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg14 --controlnet_specs assets/synthetic_Dog_seg14.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg15 --controlnet_specs assets/synthetic_Dog_seg15.json

python cosmos_transfer1/diffusion/inference/transfer.py --checkpoint_dir $CHECKPOINT_DIR --fps 30 --offload_text_encoder_model --video_save_folder outputs/synthetic_Dog_seg16 --controlnet_specs assets/synthetic_Dog_seg16.json
