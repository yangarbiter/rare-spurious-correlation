
from utils import ExpExperiments

random_seed = list(range(1))

__all__ = ['GroupInfluence', 'IncrementalRetraining', 'TrainClassifier',
           'MemInference', ]


class MemInference(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Models for membership inference"
        cls.experiment_fn = 'mem_inference'
        grid_params = []

        n_spurious = [1, 3, 5, 10, 20, 100, 1000]
        seeds = range(10)
        datasets = ["memmnist", "memfashion"]
        for spu in ['v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'memmnist{spu}-{i}-{digit}-{rs}')
                        datasets.append(f'memfashion{spu}-{i}-{digit}-{rs}')

        base_params = {
            'dataset': datasets,
            'model': [f'ce-tor-LargeMLP', ],
            'batch_size': [128],
            'epochs': [70],
            'weight_decay': [0.],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.0], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], **base_params,
        ))

        seeds = range(10)
        n_spurious = [1, 3, 5, 10, 20, 100, 500, 1000]
        datasets = ['memcifar10']
        for spu in ['v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'memcifar10{spu}-{i}-{digit}-{rs}')

        base_params = {
            'dataset': datasets,
            'batch_size': [128],
            'model': [f'aug01-ce-tor-altResNet20Norm02'],
            #'weight_decay': [0., 1e-4],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.1], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class GroupInfluence(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Group influence"
        cls.experiment_fn = 'group_influence'
        grid_params = []

        n_spurious = [3, 5, 10, 20, 100, 2000, 5000]
        n_spurious = [3, 5, 10, 20]

        seeds = range(5)
        datasets = []
        #for spu in ['v1', 'v3', 'v8', 'v18', 'v19', 'v20']:
        for spu in ['v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        #datasets.append(f'mnist{spu}-{i}-{digit}-{rs}')
                        datasets.append(f'fashion{spu}-{i}-{digit}-{rs}')

        for ds in datasets:
            for model in [f'ce-tor-LargeMLP',]:
                base_params = {
                    'dataset': [ds],
                    'model': [model],
                    'batch_size': [128],
                    'random_seed': list(range(1)),
                    'optimizer': ["sgd"],
                }
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0.pt',
                        #f'128-{ds}-70-0.01-{model}-0.9-sgd-0-0.0.pt',
                    ],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0.pt',
                    ],
                    #gi_scale=[200., 150., 50.],
                    gi_scale=[150.],
                    **base_params,
                ))

        n_spurious = [100]
        seeds = range(5)
        datasets = []
        for spu in ['v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'mnist{spu}-{i}-{digit}-{rs}')
                        datasets.append(f'fashion{spu}-{i}-{digit}-{rs}')

        for ds in datasets:
            for model in [f'ce-tor-LargeMLP',]:
                base_params = {
                    'dataset': [ds],
                    'model': [model],
                    'batch_size': [128],
                    'random_seed': list(range(1)),
                    'optimizer': ["sgd"],
                }
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0.pt',
                        #f'128-{ds}-70-0.01-{model}-0.9-sgd-0-0.0.pt',
                    ],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0.pt',
                    ],
                    #gi_scale=[200., 150., 50.],
                    gi_depth=[500],
                    gi_scale=[150.],
                    **base_params,
                ))

        seeds = range(5)
        datasets = []
        for spu in ['v20']:
            for i in n_spurious:
                for digit in [0]:
                    for rs in seeds:
                        datasets.append(f'cifar10{spu}-{i}-{digit}-{rs}')

        for ds in datasets:
            for model in [f'aug01-ce-tor-altResNet20Norm02',]:
                base_params = {
                    'dataset': [ds],
                    'model': [model],
                    'batch_size': [128],
                    'random_seed': list(range(1)),
                    'optimizer': ["sgd"],
                }
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0001.pt',
                        #f'128-{ds}-70-0.1-{model}-0.9-sgd-0-0.0.pt',
                    ],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0001.pt',
                        #f'128-{ds}-70-0.1-{model}-0.9-sgd-0-0.0.pt',
                    ],
                    gi_depth=[200],
                    gi_scale=[1000.],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[
                        f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0001.pt',
                        #f'128-{ds}-70-0.1-{model}-0.9-sgd-0-0.0.pt',
                    ],
                    gi_scale=[1000.],
                    **base_params,
                ))


        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class IncrementalRetraining(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "incremental retraining"
        cls.experiment_fn = 'incremental_retraining'
        grid_params = []

        n_spurious = [3, 5, 10, 20, 100, 2000, 5000]

        seeds = range(5)
        datasets = []
        for spu in ['v1', 'v3', 'v8', 'v18', 'v19', 'v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'mnist{spu}-{i}-{digit}-{rs}')
                        datasets.append(f'fashion{spu}-{i}-{digit}-{rs}')

        for ds in datasets:
            for model in [f'ce-tor-LargeMLP',]:
                base_params = {
                    'dataset': [ds],
                    'model': [model],
                    'batch_size': [128],
                    'epochs': [140],
                    'random_seed': list(range(1)),
                }
                grid_params.append(dict(
                    model_path=[f'128-{ds}-70-0.01-{model}-0.9-sgd-0-0.0.pt'],
                    weight_decay=[0.], optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0.pt'],
                    weight_decay=[0.], optimizer=['adam'], momentum=[0.0], learning_rate=[0.01],
                    **base_params,
                ))

        seeds = range(5)
        n_spurious = [3, 5, 10, 20, 100, 500]
        datasets = []
        for spu in ['v8', 'v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'cifar10{spu}-{i}-{digit}-{rs}')

        for ds in datasets:
            for model in [f'aug01-ce-tor-altResNet20Norm02',]:
                base_params = {
                    'dataset': [ds],
                    'model': [model],
                    'batch_size': [128],
                    'epochs': [140],
                    'random_seed': list(range(1)),
                }
                grid_params.append(dict(
                    model_path=[f'128-{ds}-70-0.1-{model}-0.9-sgd-0-0.0001.pt'],
                    weight_decay=[1e-4], optimizer=['sgd'], momentum=[0.9], learning_rate=[0.1],
                    **base_params,
                ))
                grid_params.append(dict(
                    model_path=[f'128-{ds}-70-0.01-{model}-0.0-adam-0-0.0001.pt'],
                    weight_decay=[1e-4], optimizer=['adam'], momentum=[0.0], learning_rate=[0.01],
                    **base_params,
                ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class TrainClassifier(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "train classifier"
        cls.experiment_fn = 'train_classifier'
        grid_params = []

        seeds = range(5)
        n_spurious = [3, 5, 10, 20, 100, 2000, 5000]

        datasets = ['mnist', 'fashion']
        for spu in ['v1', 'v3', 'v8', 'v18', 'v19', 'v20', 'v30']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'mnist{spu}-{i}-{digit}-{rs}')
                        datasets.append(f'fashion{spu}-{i}-{digit}-{rs}')

        base_params = {
            'dataset': datasets,
            'model': [f'ce-tor-LargeMLP', f'ce-tor-CNN002', ],
            'batch_size': [128],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01], **base_params,
        ))
        grid_params.append(dict(
            #grad_clip=[1.0, 0.1, 0.01],
            grad_clip=[0.1],
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))
        grid_params.append(dict(
            grad_clip=[1.0, 0.1, 0.01],
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))


        datasets = ['mnist', 'fashion']
        for spu in ['v1', 'v3', 'v8', 'v18', 'v19', 'v20', 'v30', ]:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'mnist{spu}-{i}-{digit}-{rs}')
                        if spu != 'v30':
                            datasets.append(f'fashion{spu}-{i}-{digit}-{rs}')

                for rs in seeds:
                    datasets.append(f'twoclassmnist{spu}-{i}-0-1-{rs}')

        base_params = {
            'dataset': datasets,
            'model': [f'ce-tor-MLP', f'ce-tor-MLPv2', f'ce-tor-LargeMLP', f'ce-tor-LargeMLPv2', f'ce-tor-CNN002', ],
            'batch_size': [128],
            'weight_decay': [0.],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))



        seeds = range(5)
        n_spurious = [3, 5, 10, 20, 100, 500]
        datasets = ['cifar10']
        for spu in ['v1', 'v3', 'v8', 'v18', 'v19', 'v20', 'v30']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'cifar10{spu}-{i}-{digit}-{rs}')

        base_params = {
            'dataset': datasets,
            'batch_size': [128],
            'model': [f'aug01-ce-tor-altResNet20Norm02'],
            'weight_decay': [1e-4],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.1], **base_params,
        ))
        grid_params.append(dict(
            grad_clip=[0.1],
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.1], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))
        grid_params.append(dict(
            grad_clip=[0.1],
            optimizer=['adam'], momentum=[0.], learning_rate=[0.01], **base_params,
        ))

        seeds = range(5)
        datasets = ['cifar10']
        for spu in ['v8', 'v20']:
            for i in n_spurious:
                for digit in [0, 1]:
                    for rs in seeds:
                        datasets.append(f'cifar10{spu}-{i}-{digit}-{rs}')
        base_params = {
            'dataset': datasets,
            'batch_size': [128],
            'model': [f'aug01-ce-tor-altResNet20Norm02',
                      f'aug01-ce-tor-altResNet32Norm02',
                      f'aug01-ce-tor-altResNet110Norm02',],
            'weight_decay': [1e-4],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.1], **base_params,
        ))
        base_params = {
            'dataset': datasets,
            'batch_size': [128],
            'model': [f'aug01-ce-tor-Vgg16Norm02'],
            'weight_decay': [1e-4],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01], **base_params,
        ))


        datasets = ['cifar10', 'cifar100', 'cifar100coarse']
        base_params = {
            'dataset': datasets,
            'batch_size': [64],
            'model': [f'ce-tor-ResNet18', f'ce-tor-ResNet50', 'ce-tor-Vgg16'],
            'weight_decay': [1e-4],
            'epochs': [70],
            'random_seed': list(range(1)),
        }
        grid_params.append(dict(
            optimizer=['adam'], momentum=[0.], learning_rate=[0.1, 0.01], **base_params,
        ))
        grid_params.append(dict(
            optimizer=['sgd'], momentum=[0.9], learning_rate=[0.01], **base_params,
        ))

        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)
