# Understanding Rare Spurious Correlations in Neural Network


This repository contains the code of the experiments in the paper

[Understanding Rare Spurious Correlations in Neural Network](https://arxiv.org/abs/2202.05189)

Authors: [Yao-Yuan Yang](https://github.com/yangarbiter/), [Kamalika Chaudhuri](http://cseweb.ucsd.edu/~kamalika/)

## Abstract

Neural networks are known to use spurious correlations for classification; for
example, they commonly use background information to classify objects.
But how many examples does it take for a network to pick up these correlations?
This is the question that we empirically investigate in this work.
We introduce spurious patterns correlated with a specific class to a few
examples and find that it takes only a handful of such examples for the network
to pick up on the spurious correlation.
Through extensive experiments, we show that (1) spurious patterns with a larger
$\ell_2$ norm are learnt to correlate with the specified class more easily; (2)
network architectures that are more sensitive to the input are more susceptible
to learning these rare spurious correlations; (3) standard data deletion
methods, including incremental retraining and influence functions, are unable to
forget these rare spurious correlations through deleting the examples that cause
these spurious correlations to be learnt.

---

## Installation

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -r requirements.txt
```

## Scripts

- [notebooks/rare_spurious_correlation.ipynb](notebooks/rare_spurious_correlation.ipynb): compute the spurious scores
- [notebooks/visualize_weight.ipynb](notebooks/visualize_weight.ipynb): generates Figure 5

## Usage

### Experiment options

- `train_classifier`: train a classifier ([implementation](experiments/train_classifier.py))
- `group_infulence`: remove spurious examples from a model using group influence ([implementation](experiments/group_influence.py))
- `incremental_retraining`: remove spurious examples from a model using incremental retraining ([implementation](experiments/incremental_retraining.py)

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
  author={Yao-Yuan Yang and Kamalika Chaudhuri},
  journal={arXiv preprint arXiv:2202.05189},
  year={2022}
}
```
