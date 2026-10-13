# Plot to Code Refine

A training project for generating and refining matplotlib code from chart images, including both SFT and RL training stages.

## Directory Structure

```
plot_to_code_refine_origin/
├── dataset/
│   └── generate_data.py                          # Generate training data using GPT-4o with three modification types: data, global_style, local_style
├── inference/
│   ├── chart_test.py                             # Chart testing and iterative correction process using vLLM for code generation and refinement
│   ├── insert_comment.py                         # Insert TODO comments into code based on code differences
│   └── code_diff.py                              # Code difference comparison to identify added, deleted, and modified lines
├── training/
│   ├── sft/
│   │   ├── train_sft.sh                          # SFT training launch script
│   │   └── yaml/
│   │       ├── llm_config.yaml                   # LLM training configuration (model path, hyperparameters, etc.)
│   │       └ fsdp_config.yaml                    # FSDP distributed training configuration
│   ├── rl/
│   │   ├── config/
│   │       ├── constants.py                      # Constant definitions (regular expressions)
│   │       └ image_diff.yaml                     # Complete RL training configuration file
│   │   ├── dataprocessing/
│   │       └ dataset_building.py                 # Convert raw data to RL training data format
│   │   ├── dataset/
│   │       └ image_diff_dataset.py               # Custom RL dataset loader for handling image inputs
│   │   ├── reward/
│   │       └ image_diff_reward.py                # Reward function computation for evaluating type prediction and comment accuracy
├── README.md                                     # Project documentation
├── LICENSE                                       # Open source license file
└── .gitignore                                    # Git ignore configuration
```

## Training Process

### 1. Data Preparation

#### Generate Training Data
Generate training data using GPT-4o API with three modification types:
- **data**: Modify data values and number of graphic shapes
- **global_style**: Modify global styles (axis ranges, labels, title, grid, etc.)
- **local_style**: Modify local styles (graphic element styles)

```bash
# Set OpenAI API configuration
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"

# Run data generation script
python dataset/generate_data.py
```

Generated data is saved in the `new_train_data/` directory, including:
- `code_{id}_{step}.py`: Code files for each step
- `image_{id}_{step}.png`: Generated images for each step
- `modification_orders.json`: Modification order records

#### Build SFT Dataset
Process data using `inference/code_diff.py` and `inference/insert_comment.py`:

```bash
# Generate code difference comparison results
python inference/code_diff.py

# Insert TODO comments into code
python inference/insert_comment.py
```

#### Build RL Dataset
Convert raw data to parquet format required for RL training:

```bash
python training/rl/dataprocessing/dataset_building.py \
    --dataset_path /path/to/raw_data.json \
    --verl_dataset_dir ./data/image_diff/
```

### 2. SFT Training (Supervised Fine-Tuning)

SFT training uses the LLaMA-Factory framework to fine-tune the Qwen3-VL-2B model.

#### Configuration
Modify `training/sft/yaml/llm_config.yaml`:
```yaml
model_name_or_path: ~/LLM/Qwen3-VL-2B-Instruct  # Model path
dataset: train_data                              # Dataset name
dataset_dir: train_data                          # Dataset directory
output_dir: Qwen3-VL-2B-Instruct-sft            # Output directory
num_train_epochs: 2                              # Number of training epochs
per_device_train_batch_size: 1                   # Batch size
gradient_accumulation_steps: 16                  # Gradient accumulation steps
learning_rate: 1e-5                              # Learning rate
```

Modify `training/sft/yaml/fsdp_config.yaml`:
```yaml
num_processes: 4                                 # Number of GPUs
num_machines: 1                                  # Number of nodes
```

#### Launch Training
```bash
cd training/sft
bash train_sft.sh
```

Or manually execute:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 accelerate launch \
    --config_file yaml/fsdp_config.yaml \
    ~/LLaMA-Factory/src/train.py yaml/llm_config.yaml
```

#### Training Output
After training completes, the model is saved in the `Qwen3-VL-2B-Instruct-sft/` directory.

### 3. RL Training (Reinforcement Learning)

RL training uses the veRL framework with GRPO algorithm for policy optimization.

#### Configuration
Modify `training/rl/config/image_diff.yaml`:

Key configuration items:
```yaml
data:
  train_files: ~/data/image_diff/train.parquet  # Training data path
  val_files: ~/data/image_diff/test.parquet     # Validation data path
  max_prompt_length: 4096                       # Maximum prompt length
  max_response_length: 3072                     # Maximum response length

actor_rollout_ref:
  model:
    path: ~/output_qwen3_vl_2b/checkpoint-64    # SFT model path
  actor:
    ppo_mini_batch_size: 16                     # PPO batch size
    ppo_micro_batch_size_per_gpu: 4            # GPU micro batch size
    lr: 1e-6                                    # Learning rate
  rollout:
    temperature: 1.0                            # Generation temperature
    n: 8                                        # Number of samples per prompt
    tensor_model_parallel_size: 2              # Model parallel size

trainer:
  total_epochs: 3                               # Total training epochs
  n_gpus_per_node: 4                            # Number of GPUs per node
  save_freq: 100                                # Save frequency
```

#### Launch Training
RL training requires veRL training scripts, specific commands depend on veRL deployment:

```bash
# Example command (adjust based on actual veRL deployment)
python -m verl.trainer.ppo_trainer \
    --config-path training/rl/config \
    --config-name image_diff.yaml
```

Or use hydra configuration:
```bash
cd training/rl
python train_rl.py --config-name image_diff
```

#### Reward Function
RL training uses a custom reward function `image_diff_reward.py` with two components:
- **type_reward**: Accuracy of predicting difference types (data/global_style/local_style)
- **comment_reward**: Accuracy of inserting TODO comments (position, type, line count)

Reward calculation formula:
```python
score = type_reward + comment_reward
```

### 4. Inference and Testing

Use the trained model for chart code generation and iterative refinement:

```bash
python inference/chart_test.py
```

Configuration parameters:
```python
GENERATION_MODEL_PATH = "qwen3_vl_4b_sft"        # Generation model path
CORRECTION_MODEL_PATH = "qwen3_vl_4b_sft"       # Correction model path
MAX_ITERATIONS = 6                               # Maximum number of iterations
BATCH_SIZE = 20                                  # Batch processing size
```

## Model Output

The trained model can be used for:
1. Generate matplotlib code from chart images
2. Identify chart difference types
3. Insert TODO comments in code to guide modifications

## Dependencies

Main dependencies:
- Python 3.8+
- PyTorch 2.0+
- Transformers
- LLaMA-Factory (SFT)
- veRL (RL)
- vLLM (inference)
- OpenAI API (data generation)

## Notes

1. SFT training requires training data containing TODO comments
2. RL training reward function is based on code differences and comment accuracy
3. Inference stage supports iterative refinement to gradually optimize generated code
4. GPU memory requirement: At least 32GB VRAM recommended for 4-GPU training