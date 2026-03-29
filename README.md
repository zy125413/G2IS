# G2IS: Gradient-based Graph Instruction Selection 

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue.svg)](https://aclanthology.org/2025.acl-long.1189/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository provides the core implementation of **G2IS**, proposed in the ACL 2025 Main Conference paper:

**[Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](https://aclanthology.org/2025.acl-long.1189/)**

G2IS is a gradient-based graph method for instruction tuning data selection. It uses validation gradients to extract core knowledge directions, then performs a constrained walk on a training gradient graph to select a compact and effective subset of training data.

## 🌟 Method

The implementation follows the main idea of G2IS:

1. **Validation core knowledge extraction**: PCA is applied to the validation gradients to extract principal components as core knowledge directions.
2. **Anchor selection**: For each validation component, the algorithm finds highly related training samples as candidate anchors.
3. **Gradient graph walk**: Starting from anchors, the algorithm selects training samples under three strict principles:
   - **Knowledge Coherence**
   - **No Conflict in Knowledge**
   - **Consistency with Core Knowledge**
4. **Fallback re-anchoring**: When local walking cannot continue without violating constraints, the algorithm falls back to a global search using the current validation knowledge direction.



## 🧠 Training and Gradient Setting

Following the paper, the training gradients are based on the **momentum-adjusted gradients** used in modern optimization, and the optimizer state is initialized with a **warmup method** before computing training gradients. 

In practice, the gradient extraction pipeline follows the setting introduced in [**LESS**](https://github.com/princeton-nlp/LESS). The validation gradients are computed separately and used to represent the target task knowledge direction. Gradient computation is performed on **LoRA layers**, and **Random Projection** is used for dimensionality reduction.

For instruction tuning and evaluation, experiments are conducted based on [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory).

## 📊 Evaluation Setting

The paper evaluates G2IS on multiple domain adaptation and instruction tuning benchmarks. Reported evaluation tasks include:
- **BBH**, **GPQA**, **GSM8K**, **Math**, **MMLU**

Experiments are conducted on several backbone models, including:
- **Llama-3.1-8B**, **Gemma-7B**, **Mistral-7B**

## ⚠️ Important Assumption (Algorithmic Fidelity)

The input `graph` parameter **MUST be a pre-sorted adjacency list**.

For each node `i`, `graph[i]` should contain its neighbor indices sorted in **descending order** of similarity to node `i`. This strict ordering is mathematically required for the local walk step to implement the intended constrained `argmax` behavior (Knowledge Coherence) efficiently via early-stopping. Passing an unsorted graph will lead to mathematically invalid selections.

## 🚀 Usage

The script expects the following precomputed inputs:
- Training gradients (`.pt`)
- Validation gradients (`.pt`)
- Precomputed graph over training samples (`.npy`)
- Original training data (`.jsonl`)

```bash
python gradient_walk.py \
  --train_gradients_file path/to/train_gradients.pt \
  --validation_gradients_file path/to/val_gradients.pt \
  --graph path/to/graph.npy \
  --train_data_dir path/to/train.jsonl \
  --save path/to/output_dir \
  --pca_mode variance \
  --val_k 0.5 \
  --train_k 0.01 \
  --ways 0.8
```

### Arguments

* `--train_gradients_file`: Path to the training gradients.
* `--validation_gradients_file`: Path to the validation gradients.
* `--graph`: Path to the precomputed sorted graph.
* `--train_data_dir`: Path to the original training JSONL file.
* `--save`: Directory to save the selected subset.
* `--transpose_train`: *(Optional)* Flag to transpose the training gradients after loading.
* `--transpose_val`: *(Optional)* Flag to transpose the validation gradients after loading.
* `--pca_mode`: PCA mode for extracting validation components (`variance` or `count_ratio`).
* `--val_k`: PCA threshold (e.g., `0.5` for 50% explained variance).
* `--train_k`: Ratio of total training data to select (e.g., `0.01` for 1%).
* `--ways`: Threshold ($\delta$) for maintaining consistency with validation core knowledge.

### Output

The script successfully saves:
* `target_sample.kpl`: Selected sample indices.
* `target.jsonl`: Selected data instances (ready for training).
* `target.kpl`: Selected data in pickle format.

## 📝 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zhao2025g2is,
  title={Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection},
  author={Zhao, Yang and Du, Li and Ding, Xiao and Ouyang, Yangou and Wang, Hepeng and Xiong, Kai and Gao, Jinglong and Sun, Zhouhao and Xu, Dongliang and Yang, Qing and Li, Dongchen and Qin, Bing and Liu, Ting},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2025},
  url={[https://aclanthology.org/2025.acl-long.1189/](https://aclanthology.org/2025.acl-long.1189/)}
}
```
