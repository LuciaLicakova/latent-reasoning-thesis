# Coconut Extensions: Learnable Latent Windows and LoRA

This repository reproduces and extends the Coconut framework from [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) by Hao et al. (2024). 

It includes a contextual latent embedding modification, where each latent token is computed as a learned aggregation over multiple recent hidden states. Additionally, we apply parameter-efficient fine-tuning via Low-Rank Adaptation to reduce the memory cost of training.

The implementation is optimised for limited hardware (2×11 GB GPUs) through gradient accumulation and reduced batch sizes, since multi-pass latent reasoning remains the dominant memory bottleneck.

![coconut](assets/coconut.png)

---

## Overview

The code reproduces the Coconut training pipeline:
1. **Stage 0, Chain-of-Thought (CoT)**: trains step-by-step reasoning.
2. **Stage 1+, Latent Reasoning (Coconut)**: learns to reason within a continuous latent space.
The contextual variant extends the latent reasoning stage by integrating short-range contextual information into the latent embeddings in an attempt to improve representation coherence. LoRA-based runs restrict parameter updates to low-rank adapters and can be evaluated for their impact on memory usage and reasoning performance.

---

## Installation

Clone the fork and set up the environment:

```
git clone https://github.com/LuciaLicakova/coconut-contextual.git
cd coconut-contextual
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```
For logging, this code uses [Weights & Biases (wandb)](https://wandb.ai/site/). Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

---

## Data

Datasets for training and evaluation are expected in JSON format:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

### Example datasets
- [GSM8K](https://arxiv.org/abs/2110.14168) – numerical reasoning
- [ProntoQA](https://arxiv.org/pdf/2210.01240) – symbolic reasoning
- **ProsQA** – procedural reasoning
- [MATH](https://arxiv.org/abs/2103.03874) – LaTeX-based mathematical reasoning


Preprocessing scripts for downloading and using these datasets are provided in the `preprocessing/` directory.

---
## Running Experiments

All commands below assume 2 × 11GB GPUs, reflecting the hardware constraints of this reproduction. Adjust `batch_size_training`, `gradient_accumulation_steps`, and `nproc_per_node` to fit your resources. The effective batch size can be preserved across datasets using gradient accumulation.
This repository enables two training branches:
1. **Full Fine-Tuning**: supports standard Coconut and the learnable latent-window variant.
2. **LoRA-Based Fine-Tuning**: for the learnable latent-window variant.

### Branch A: Full Fine-Tuning

Configuration files are located in: `run_learnable_full/`. 
All experiments in this branch use `run_full.py`.

In the YAML configuration, choose the latent update mechanism via:
- `latent_variant: coconut`: standard Coconut latent update.
- `latent_variant: learnable_weights`: learnable latent-window aggregation.

#### GSM8K

Preprocess data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/gsm_cot.yaml
```

Select a checkpoint as the initialisation of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [run_learnable_full/gsm_coconut.yaml](run_learnable_full/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [run_learnable_full/gsm_coconut_eval.yaml](run_learnable_full/gsm_coconut_eval.yaml). Then evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```
Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```

Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/prontoqa_coconut.yaml
```

Find the checkpoint with the best validation accuracy, put the path as `load_model_path` in [run_learnable_full/prosqa_coconut_eval.yaml](run_learnable_full/prosqa_coconut_eval.yaml), and evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/prontoqa_coconut_eval.yaml
```
### ProsQA

The ProsQA dataset is in [data/prosqa_*.json](data).

Run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/prosqa_coconut.yaml
```

Find the checkpoint with the best validation accuracy, put the path as `load_model_path` in [run_learnable_full/prosqa_coconut_eval.yaml](run_learnable_full/prosqa_coconut_eval.yaml), and evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 2 run_full.py run_learnable_full/prosqa_coconut_eval.yaml
```
### Branch B: LoRA-Based Fine-Tuning

Configuration files are located in: `run_learnable_weights/`. 
All experiments in this branch use `run.py`.

In the YAML configuration, LoRA-based latent reasoning via `latent_variant: learnable_weights_lora`: learnable latent-window aggregation.

Otherwise the execution is analogous to Branch A.

## Configuration

All experiment settings are defined in YAML files (for example [here](run_full.py run_learnable_full/gsm_coconut.yaml)).

- **General settings**

  - `project`: WandB project name
  - `save_path`: Your path to store the checkpoints
  - `only_eval`: If true, only load the model and test on the data from `val_path` (must be used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - `coconut`: Train a coconut model
  - `cot`: Train a chain-of-thought model
  - `no_thoughts`: Train a coconut (w/o thought) model
  - `no_cot`: Train a no-cot model
  - `latent_variant`: use `basic` for standard Coconut behaviour, `learnable_weights` for full fine-tuning with learnable weights, and `learnable_weights_lora` to apply low-rank adaptation. Note that `basic` and `learnable_weights` are used with `run_full.py`, while `learnable_weights` is used with `run.py`.


- **Training settings**

  - `c_thought`: Number of continuous latent thoughts for each reasoning step
  - `epochs_per_stage`: Number of epochs for every training stage
  - `max_latent_stage`: The maximum number of training stages (in addition to the initial stage)
  - `pad_latent_to_max`: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous latent thoughts.
  - `save_only_improve`: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might not get saved.
  - `uniform_prob`: The probability to mix data from other stages. It is set to 0 for a standard experiment.
  - `model_id`: Huggingface model ID to load as the initialisation, e.g., `openai-community/gpt2`
  - `load_model_path`: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize Coconut from a CoT-tuned model.
  - `seed`: Random seed.
  - `resume`: The epoch to resume from. Can be used when we want to skip the initial training stages.
  - `bf16`: Whether to use bf16 (mixed-precision) training.
  - `train_path`: Path to the training set.
  - `val_path`: Path to the validation or test set. If we only run evaluation (`only_eval = True`), this path should point to the test set. During training, it should point to the validation set.
  - `reset_optimizer`: Whether to reset the optimizer when switching training stages.
  - `batch_size_training`: Batch size to train the model per GPU.
  - `debug`: If true, there is no WandB logging and model saving and only a subset of data is used.
  - `gradient_accumulation_steps`: Gradient accumulation steps.
  - `num_epochs`: Maximum training epochs.
  - `lr`: Learning rate.
  - `weight_decay`: Weight decay.

## Citation
If you use this code base in your research, please cite the original paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).
