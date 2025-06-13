# #!/bin/bash

# export PROMPT="The video is captured from an autonomous vehicle's 360° LiDAR array as it glides along Xuhui Riverside, Shanghai. The camera faces forward, framing a futuristic urban drive at twilight, where the last embers of sunset blend with the neon pulse of the city. The sensors register every detail: the sleek curves of the AI-driven streetlights flickering to life, their cool white beams cutting through the violet-hued dusk. The road surface—embedded with smart pavement markers—emits a faint blue glow, syncing with the vehicle’s navigation system to guide its path. To the left, the Huangpu River mirrors the digital skyline, its surface rippling with reflections of LED-adorned skyscrapers that shift between gradients of electric blue and holographic silver. On the right, glass-paneled smart bus stops display augmented reality ads, their transparent screens overlaying real-time data onto the physical world. A delivery drone hums overhead, its blinking red navigation lights tracing a precise path toward a high-rise residential tower. The autonomous sedan moves in perfect sync with traffic, its sensors detecting self-driving taxis and electric scooters weaving seamlessly through the lanes. Ahead, a holographic traffic signal transitions from amber to green, casting an ephemeral shimmer over the carbon-fiber road dividers. The air carries a faint ozone scent—a byproduct of the city’s wireless charging infrastructure humming beneath the asphalt. As the vehicle rounds a bend, the undulating facade of the West Bund AI Tower dominates the view, its kinetic exterior panels adjusting in real-time to optimize solar absorption. The sky, now a gradient of deep indigo and neon pink, frames the scene in a cyberpunk glow. Every element—from the synchronized movement of smart traffic to the river’s data-responsive light show—converges into a symphony of urban efficiency. This isn’t just a drive; it’s a glimpse into the algorithmic heartbeat of tomorrow’s Shanghai."
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
# export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
# export NUM_GPU="${NUM_GPU:=1}"

# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
#     --checkpoint_dir $CHECKPOINT_DIR \
#     --video_save_name output_video_250604 \
#     --video_save_folder outputs/sample_av_multi_control_demo \
#     --prompt "$PROMPT" \
#     --sigma_max 80 \
#     --offload_text_encoder_model --is_av_sample \
#     --controlnet_specs assets/sample_av_multi_control_spec.json \
#     --num_gpus $NUM_GPU

#!/bin/bash
export PROMPT="The footage is captured by a surround-view camera, describing a vibrant outdoor parking lot in Russian. The scene is framed by a backdrop of modern architecture, where sleek buildings rise against a clear blue sky dotted with fluffy white clouds. The asphalt surface of the parking lot is marked with crisp white lines, guiding vehicles into neatly organized spaces. A mix of cars, from compact sedans to larger SUVs, are parked in this parking lot. To the left, a row of colorful brick buildings adds a touch of historical charm, their facades reflecting the warm sunlight."
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPUS=1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name simone_output_video_no_fisheye_hdmap \
--video_save_folder outputs/simone_sample_av_multiview_demo \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/simone_parking_demo/simone_sample_av_hdmap_multiview_spec_no_fisheye_hdmap.json --num_gpus ${NUM_GPUS} --num_steps 50 \
--view_condition_video assets/simone_parking_demo/no_fisheye/cam_front.mp4 \
--prompt "$PROMPT"

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_transfer1/diffusion/inference/transfer_multiview.py \
--checkpoint_dir $CHECKPOINT_DIR \
--video_save_name simone_output_video_no_fisheye_lidar \
--video_save_folder outputs/simone_sample_av_multiview_demo \
--offload_text_encoder_model \
--guidance 3 \
--controlnet_specs assets/simone_parking_demo/simone_sample_av_hdmap_multiview_spec_no_fisheye_lidar.json --num_gpus ${NUM_GPUS} --num_steps 50 \
--view_condition_video assets/simone_parking_demo/no_fisheye/cam_front.mp4 \
--prompt "$PROMPT"
