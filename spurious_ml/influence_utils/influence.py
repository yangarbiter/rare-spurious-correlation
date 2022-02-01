"""
implementation from https://github.com/nimarb/pytorch_influence_functions/
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from tqdm import tqdm


def s_test(z, yi, model, Xt, yt, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """
    get_inverse_hvp using LISSA

    LiSSA – Linear-time Second-order Stochastic Algorithm
    
    s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.
    Returns:
        h_estimate: list of torch tensors, s_test
    """
    device = next(model.parameters()).device
    v = grad_z(model, z.to(device), yi.to(device))
    h_estimate = [vv.detach() for vv in v.copy()]

    trn_loader = DataLoader(TensorDataset(Xt, yt), batch_size=1, shuffle=True)

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for _ in tqdm(range(recursion_depth)):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, y in trn_loader:
            x, y = x.to(device), y.to(device)
            loss = calc_loss(model(x), y)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = hvp(loss, params, h_estimate)
            hv = [hhvv.detach() for hhvv in hv]
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            #torch.cuda.reset_peak_memory_stats()
            break
    return h_estimate


def calc_loss(y, t):
    """Calculates the loss
    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    #y = torch.nn.functional.log_softmax(y)
    #loss = torch.nn.functional.nll_loss(y, t, weight=None, reduction='mean')
    loss = torch.nn.CrossEntropyLoss(reduction="mean")(y, t)
    return loss


def grad_z(model, z, y):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    # initialize
    outputs = model(z)
    loss = calc_loss(outputs, y)
    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = torch.autograd.grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = torch.autograd.grad(elemwise_products, w, create_graph=True)

    return return_grads


def calc_s_test_single(model, z_test, t_test, Xt, yt,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(z_test, t_test, model, Xt, yt,
                                      damp=damp, scale=scale, recursion_depth=recursion_depth))

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec

def first_order_group_influence(model, Z, y, Xt, yt, damp=0.01, scale=25, recursion_depth=5000, r=1):
    p = len(Z) / len(Xt)
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(Z, y), batch_size=1, shuffle=False)

    hessian_inv_hpv = None
    for z, yi in loader:
        z, yi = z.to(device), yi.to(device)
        temp = calc_s_test_single(model, z, yi, Xt, yt,
                                  damp=damp, scale=scale,
                                  recursion_depth=recursion_depth, r=r)
        if hessian_inv_hpv is not None:
            hessian_inv_hpv = [t + v for t, v in zip(hessian_inv_hpv, temp)]
        else:
            hessian_inv_hpv = temp

    return [-1 / (1-p) / len(Xt) * hihpv for hihpv in hessian_inv_hpv]


#def second_order_group_influence(model, Z, y, Xt, yt, ):
#    p = len(Z) / len(Xt)
#    p / (1-p)
#    grad_z(model, z, y):
#    pass


def calc_influence_single(model, z, yi, Xt, yt, recursion_depth, r):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
    """
    s_test_vec = calc_s_test_single(model, z, yi, Xt, yt, damp=0.01, scale=25,
                                    recursion_depth=recursion_depth, r=r)

    device = next(model.parameters()).device
    trn_loader = DataLoader(TensorDataset(Xt, yt), batch_size=1, shuffle=False)

    # Calculate the influence function
    influences = []
    for xti, yti in tqdm(trn_loader):
        xti, yti = xti.to(device), yti.to(device)
        grad_z_vec = grad_z(model, xti, yti)
        tmp_influence = -sum([torch.sum(k * j).cpu().detach() for k, j in zip(grad_z_vec, s_test_vec)]) / len(Xt)
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
        influences.append(tmp_influence.item())

    return influences
