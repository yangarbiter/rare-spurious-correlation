import logging

import numpy as np

from autovar.base import RegisteringChoiceType, register_var, VariableClass

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def get_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    return x_train, y_train, x_test, y_test, np.empty(0)

def get_fashion():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, y_train, x_test, y_test = np.copy(x_train), np.copy(y_train), np.copy(x_test), np.copy(y_test)
    x_train.setflags(write=1)
    y_train.setflags(write=1)
    x_test.setflags(write=1)
    y_test.setflags(write=1)
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    return x_train, y_train, x_test, y_test, np.empty(0)

def get_cifar100(label_mode="fine"):
    from tensorflow.keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode)
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    return x_train, y_train, x_test, y_test, np.empty(0)

def get_cifar10():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    return x_train, y_train, x_test, y_test, np.empty(0)

def add_spurious_correlation(X, version, seed):
    if not version or version == "v1":
        X[:, 0, 0] = 1
    elif version == "v3":
        X[:, :3, :3] = 1
    elif version == "v8":
        X[:, :5, :5] = 1
    elif version == "v18":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + 0.25 * noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v19":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + 0.5 * noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v20":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v30":
        X[:, 3:25, 13:16, :] = 1
    else:
        raise ValueError(f"version: {version} not supported")
    return X

def add_colored_spurious_correlation(X, version, seed):
    if version == "v1":
        X[:, 0, 0, :] = 1
    elif version == "v3":
        X[:, :3, :3, :] = 1
    elif version == "v8":
        X[:, :5, :5, :] = 1
    elif version == "v18":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + 0.25 * noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v19":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + 0.5 * noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v20":
        random_state = np.random.RandomState(seed)
        noise = random_state.rand(*X.shape[1:])
        ret = X + noise.reshape(1, *X.shape[1:])
        X = np.clip(ret, 0, 1)
    elif version == "v30":
        X[:, 3:25, 13:16, :] = 1
    else:
        raise ValueError(f"version: {version} not supported")
    return X

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"mnist", shown_name="mnist")
    @staticmethod
    def mnist(auto_var):
        return get_mnist()

    @register_var(argument=r"mnist(?P<version>v[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no>[0-9])-(?P<seed>[0-9]+)", shown_name="mnist")
    @staticmethod
    def mnist_aug(auto_var, version, sp_counts, cls_no, seed):
        logger.info(f"[dataset] MNIST, {version}, {sp_counts}, {cls_no}, {seed}")
        sp_counts, cls_no, seed = int(sp_counts), int(cls_no), int(seed)
        x_train, y_train, x_test, y_test, _ = get_mnist()

        random_state = np.random.RandomState(seed)
        spurious_ind = random_state.choice(np.where(y_train == cls_no)[0], size=sp_counts, replace=False)
        x_train[spurious_ind] = add_spurious_correlation(x_train[spurious_ind], version, seed)

        return x_train, y_train, x_test, y_test, spurious_ind

    @register_var(argument=r"twoclassmnist(?P<version>v[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no1>[0-9])-(?P<cls_no2>[0-9])-(?P<seed>[0-9]+)", shown_name="mnist")
    @staticmethod
    def twoclasslmnist_aug(auto_var, version, sp_counts, cls_no1, cls_no2, seed):
        logger.info(f"[dataset] MNIST, {version}, {sp_counts}, {cls_no1}, {cls_no2}, {seed}")
        sp_counts, cls_no1, cls_no2, seed = int(sp_counts), int(cls_no1), int(cls_no2), int(seed)
        x_train, y_train, x_test, y_test, _ = get_mnist()

        random_state = np.random.RandomState(seed)
        spurious_ind1 = random_state.choice(np.where(y_train == cls_no1)[0], size=sp_counts, replace=False)
        x_train[spurious_ind1] = add_spurious_correlation(x_train[spurious_ind1], version, seed)
        spurious_ind2 = random_state.choice(np.where(y_train == cls_no2)[0], size=sp_counts, replace=False)
        x_train[spurious_ind2] = add_spurious_correlation(x_train[spurious_ind2], version, seed)

        return x_train, y_train, x_test, y_test, np.concatenate((spurious_ind1, spurious_ind2))

    @register_var(argument=r"fashion", shown_name="fashion mnist")
    @staticmethod
    def fashion(auto_var):
        return get_fashion()

    @register_var(argument=r"fashion(?P<version>v[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no>[0-9])-(?P<seed>[0-9]+)", shown_name="mnist")
    @staticmethod
    def fashion_aug(auto_var, version, sp_counts, cls_no, seed):
        logger.info(f"[dataset] fashion, {version}, {sp_counts}, {cls_no}, {seed}")
        sp_counts, cls_no, seed = int(sp_counts), int(cls_no), int(seed)
        x_train, y_train, x_test, y_test, _ = get_fashion()

        random_state = np.random.RandomState(seed)
        spurious_ind = random_state.choice(np.where(y_train == cls_no)[0], size=sp_counts, replace=False)
        x_train[spurious_ind] = add_spurious_correlation(x_train[spurious_ind], version, seed)

        return x_train, y_train, x_test, y_test, spurious_ind

    @register_var(argument=r"cifar100", shown_name="Cifar100")
    @staticmethod
    def cifar100(auto_var):
        return get_cifar100("fine")

    @register_var(argument=r"cifar100coarse", shown_name="Cifar100")
    @staticmethod
    def cifar100coarse(auto_var):
        return get_cifar100("coarse")

    @register_var(argument=r"cifar10", shown_name="Cifar10")
    @staticmethod
    def cifar10(auto_var):
        return get_cifar10()

    @register_var(argument=r"cifar10(?P<version>v[0-9a-z]+)?-(?P<sp_counts>[0-9]+)-(?P<cls_no>[0-9])-(?P<seed>[0-9]+)", shown_name="mnist")
    @staticmethod
    def cifar10_aug(auto_var, version, sp_counts, cls_no, seed):
        logger.info(f"[dataset] CIFAR10, {version}, {sp_counts}, {cls_no}, {seed}")
        sp_counts, cls_no, seed = int(sp_counts), int(cls_no), int(seed)
        x_train, y_train, x_test, y_test, _ = get_cifar10()

        random_state = np.random.RandomState(seed)
        spurious_ind = random_state.choice(np.where(y_train == cls_no)[0], size=sp_counts, replace=False)
        x_train[spurious_ind] = add_colored_spurious_correlation(x_train[spurious_ind], version, seed)

        return x_train, y_train, x_test, y_test, spurious_ind
