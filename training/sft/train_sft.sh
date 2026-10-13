CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 accelerate launch \
--config_file yaml/fsdp_config.yaml \
~/LLaMA-Factory/src/train.py yaml/llm_config.yaml