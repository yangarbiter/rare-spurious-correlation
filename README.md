# Understanding Rare Spurious Correlations in Neural Network


This repository contains the code of the experiments in the paper

[Understanding Rare Spurious Correlations in Neural Network](https://arxiv.org/abs/2202.05189)

Authors: [Yao-Yuan Yang](https://github.com/yangarbiter/), [Chi-Ning Chou](https://cnchou.github.io/), [Kamalika Chaudhuri](http://cseweb.ucsd.edu/~kamalika/)

## Abstract

Neural networks are known to use spurious correlations such as background information for classification. While prior work has looked at spurious correlations that are widespread in the training data, in this work, we investigate how sensitive neural networks are to *rare* spurious correlations, which may be harder to detect and correct, and may lead to privacy leaks. We introduce spurious patterns correlated with a fixed class to a few training examples and find that it takes only a handful of such examples for the network to learn the correlation. Furthermore, these rare spurious correlations also impact accuracy and privacy. We empirically and theoretically analyze different factors involved in rare spurious correlations and propose mitigation methods accordingly. Specifically, we observe that $\ell_2$ regularization and adding Gaussian noise to inputs can reduce the undesirable effects.

---

## Installation

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -r requirements.txt
```

## Scripts

- [notebooks/rare_spurious_correlation.ipynb](notebooks/rare_spurious_correlation.ipynb): compute the spurious scores
- [notebooks/visualize_weight.ipynb](notebooks/visualize_weight.ipynb): visualize MLP weights
- [notebooks/membership_inference.ipynb](notebooks/membership_inference.ipynb): generates membership inference attack results
- [notebooks/regularization.ipynb](notebooks/regularization.ipynb): generates results for using regularization methods to mitigate rare spurious correlations

## Usage

### Experiment options

- `train_classifier`: train a classifier ([implementation](experiments/train_classifier.py))
- `group_infulence`: remove spurious examples from a model using group influence ([implementation](experiments/group_influence.py))
- `incremental_retraining`: remove spurious examples from a model using incremental retraining ([implementation](experiments/incremental_retraining.py))
- `mem_inference`: train the models for membership inference attack ([implementation](experiments/mem_inference.py))

### Model options

#### Architectures

implementation: [spurious_ml/models/torch_utils/archs/](spurious_ml/models/torch_utils/archs/)

#### Examples

- 'ce-tor-LargeMLP': using [LargeMLP]((spurious_ml/models/torch_utils/archs/mlps.py) as the architecture
- 'aug01-ce-tor-altResNet20Norm02': using [altResNet20Norm02](spurious_ml/models/torch_utils/archs/alt_resnet.py#L141-L143) as the architecture with data augmentation [aug01](spurious_ml/models/torch_utils/data_augs.py#L4)

### Dataset options

Clean datasets: `mnist`, `fashion`, and `cifar10`

template: f'{clean_dataset}{spurious_pattern}-{n_spuious_examples}-{label}-{random_seed}'

#### Spurious pattern names

The name of each spurious pattern is different from the one used in the paper. Here, we provide a mapping.

- v1: small 1 (s1)
- v3: small 2 (s2)
- v8: small 3 (s3)
- v18: random 1 (r1)
- v19: random 2 (r2)
- v20: random 3 (r3)
- v30: core

#### Examples:

- cifar10v8-3-0-0: CIFAR10 with 3 spurious examples with pattern small 3 with target label 0. The spurious examples are chosen randomly with random seed 0.

### Commandline examples

```python
python ./main.py --experiment train_classifier \
    --dataset mnistv8-3-0-0 --epochs 70 --random_seed 0 \
    --batch_size 128 --model ce-tor-LargeMLP --optimizer sgd --learning_rate 0.01 --momentum 0.9
```

```python
python ./main.py --experiment group_influence \
    --dataset mnistv8-3-0-0 --epochs 70 --random_seed 0 \
    --batch_size 128 --model ce-tor-LargeMLP --optimizer sgd --learning_rate 0.01 --momentum 0.9 \
    --model_path {path_to_the_model_to_perform_data_deletion}
```

```python
python ./main.py --experiment incremental_retraining \
    --dataset mnistv8-3-0-0 --epochs 140 --random_seed 0 \
    --batch_size 128 --model ce-tor-LargeMLP --optimizer sgd --learning_rate 0.01 --momentum 0.9 \
    --model_path {path_to_the_model_to_continue_training}
```

Continue training until the 140-th epoch


## Citation

For more experimental and technical details, please check our [paper](https://arxiv.org/abs/2202.05189)

```bibtex
@article{yang2022understanding,
  title={Understanding Rare Spurious Correlations in Neural Network},
  author={Yao-Yuan Yang and Chi-Ning Chou and Kamalika Chaudhuri},
  journal={arXiv preprint arXiv:2202.05189},
  year={2022}
}
```
