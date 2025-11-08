import numpy as np
import wandb
import torch
import utils
import sys
import os
import time
import math
import random
import copy
from fickdich import FuckingContextClass
from contextlib import contextmanager
from torch import nn
from torch.nn import functional as F
from typing import Sequence
from itertools import islice
import matplotlib.pyplot as plt
import random
from torch.nn.utils import parameters_to_vector

#from itertools import islice
import itertools
from tqdm import tqdm
import sympy


sys.path.append(os.path.normpath(os.getcwd() + "/autora-theorist-darts/src"))
torch.set_default_dtype(torch.float64)

from autora.theorist.darts import DARTSExecutionMonitor, DARTSRegressor
from autora.theorist.darts.utils import NanError
from autora.theorist.darts.utils import Monitor
from autora.theorist.darts.dataset import darts_dataset_from_ndarray
from autora.theorist.darts.model_search import MixedOp
from autora.theorist.darts.operations import get_operation_as_sympy, operation_factory

import torch.nn.functional as F

def _clamp_weights(model, param_th=1000.0):
    for p in model.parameters():
        p.data.clamp_( -param_th, param_th)
        
class NRMSELoss(nn.Module):
    """
    Normalised RMSE (range normalisation).

        nRMSE = sqrt( mean( (ŷ − y)² ) ) / (y.max() − y.min())

    No NaN / inf handling, no safeguards.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        reduction: 'mean'  – default, same as torch.nn.MSELoss(reduction='mean')
                   'sum'   – divide sum by N before the sqrt
        """
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('reduction must be "mean" or "sum"')
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff2 = (y_pred - y_true) ** 2

        if self.reduction == 'mean':
            mse = diff2.mean()
        else:                           # 'sum'
            mse = diff2.sum() / diff2.numel()

        rmse = torch.sqrt(mse)

        # target range (will be zero if all elements are equal → inf/nan loss)
        rng = y_true.max() - y_true.min()

        return rmse / rng
    
class DebugSoftmax(torch.autograd.Function):
    """
    Behaves exactly like F.softmax in forward AND backward,
    but prints/records the forward input & output if the backward
    receives or produces non–finite numbers.
    """

    @staticmethod
    def forward(ctx, x, dim=-1):
        # plain forward
        y = F.softmax(x, dim=dim)
        # save *both* tensors so we can look at them later
        ctx.save_for_backward(x.detach(), y.detach())
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # unpack what we stored in the forward pass
        x, y = ctx.saved_tensors

        # check the incoming gradient
        if not torch.isfinite(grad_y).all():
            DebugSoftmax._report("incoming gradient", grad_y, x, y)

        # the canonical gradient for soft-max
        # (there is a C++ op for it, but the Python version is fine for debugging)
        s = y                       # alias: what soft-max returned
        dot = (grad_y * s).sum(dim=ctx.dim, keepdim=True)
        grad_x = (grad_y - dot) * s

        # check the gradient we are about to return
        if not torch.isfinite(grad_x).all():
            DebugSoftmax._report("outgoing gradient", grad_x, x, y)

        return grad_x, None         # second “None” for the dim arg

    # ------------------------------------------------------------------ #
    @staticmethod
    def _report(where, bad_tensor, x, y):
        mask_nan  = torch.isnan(bad_tensor)
        mask_posi = torch.isinf(bad_tensor) & (bad_tensor > 0)
        mask_negi = torch.isinf(bad_tensor) & (bad_tensor < 0)

        print(f"\n>>> NON-FINITE {where} in DebugSoftmax.backward")
        if mask_nan.any():  print("    # NaN  :", mask_nan)
        if mask_posi.any(): print("    # +Inf :", mask_posi)
        if mask_negi.any(): print("    # -Inf :", mask_negi)
        print("    first bad idx:",
              (mask_nan | mask_posi | mask_negi).nonzero(as_tuple=False)[:5])

        print("    Forward INPUT :", x)
        print("    Forward OUTPUT:", y)
        print("-" * 70)
        
def debug_softmax(x, dim=-1):
    return DebugSoftmax.apply(x, dim)

# sounds small, but for 1-3d data its pretty dense
n_samples = 200  # 200

'''def tqdm(x, **kwargs):
    return x'''

PRIMITIVES = (
    "none",
    "linear",
    # "linear",
    # "linear_logistic",
    "power_two",
    "power_three",
    # "safe_exp",
    # "safe_linear_exp",# linear_exp
    "exp",
    "ln",
    "reciprocal",
    "sin",
)


import torch
import contextlib


    
    
def nan_guard(grad):
    if not torch.isfinite(grad).all():
        print("grad shape      :", grad.shape) 
        print("non-finite rows :", (~torch.isfinite(grad)).any(1).nonzero())

def attach_nan_guards(model):
    """Attach nan-guard to every tensor *that requires grad*."""
    handles = []

    def fwd_hook(mod, inp, out):
        def add_hook(t):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                handles.append(t.register_hook(nan_guard))

        if isinstance(out, torch.Tensor):
            add_hook(out)
        elif isinstance(out, (tuple, list)):
            for t in out:
                add_hook(t)

    for m in model.modules():                       # recursive
        handles.append(m.register_forward_hook(fwd_hook))

    return handles

def attach_forward_nan_guards(model):
    """
    Recursively registers `fwd_nan_guard` on every sub-module of `model`.
    Returns the list of hook handles so they can be removed later.
    """
    handles = []

    for name, m in model.named_modules():            # recursive walk
        h = m.register_forward_hook(
                lambda mod, inp, out, n=name: fwd_nan_guard(mod, inp, out, mod_name=n))
        handles.append(h)

    return handles


def fwd_nan_guard(module, inputs, output, *, mod_name=''):
    """
    Abort if `output` (Tensor or tuple/list of Tensors) contains NaN / Inf.
    """
    def check(t):
        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).nonzero(as_tuple=True)
            print("inputs:", inputs)
            print("output:", output)
            print(module.__class__.__name__)
            print(list(module.parameters()))
            print(
                f"\n>>> NaN / Inf created in FORWARD of {mod_name} "
                f"(shape {tuple(t.shape)}) at indices {bad}"
            )

    if isinstance(output, torch.Tensor):
        check(output)
    elif isinstance(output, (tuple, list)):
        for o in output:
            check(o)
            
            
@contextmanager
def gumbel_softmax_context():
    # global gumbel_softmax_enabled
    original_state = FuckingContextClass.gumbel_softmax_enabled

    # Set the state at entry
    FuckingContextClass.gumbel_softmax_enabled = True
    try:
        yield
    finally:
        # Restore the original state on exit
        FuckingContextClass.gumbel_softmax_enabled = original_state


@contextmanager
def discrete_context():
    # global discrete_enabled
    original_state = FuckingContextClass.discrete_enabled

    # Set the state at entry
    FuckingContextClass.discrete_enabled = True
    try:
        yield
    finally:
        # Restore the original state on exit
        FuckingContextClass.discrete_enabled = original_state


def process_chunks(iterator, chunk_size):
    assert chunk_size==int(chunk_size)
    chunk_size=int(chunk_size)
    for _ in range(chunk_size):
        try:
            yield next(iterator)
        except StopIteration:
            return
        

y_th = torch.exp(torch.tensor(10, dtype=torch.float64))

class SafeMixedOp(nn.Module):
    """
    Mixture operation as applied in Differentiable Architecture Search (DARTS).
    A mixture operation amounts to a weighted mixture of a pre-defined set of operations
    that is applied to an input variable.
    """

    def __init__(self, primitives: Sequence[str] = PRIMITIVES):
        """
        Initializes a mixture operation based on a pre-specified set of primitive operations.

        Arguments:
            primitives: list of primitives to be used in the mixture operation
        """
        super(SafeMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.primitives=primitives
        # loop through all the 8 primitive operations
        for primitive in primitives:
            # OPS returns an nn module for a given primitive (defines as a string)
            op = operation_factory(primitive)

            # add the operation
            self._ops.append(op)
        # keep indices of ln, to block gradients for alphas from +eps safety measure. <0 safety does not need it, gradients will be multiplied by zero in backwards
        self.ln_idx=[i for i, primitive in enumerate(primitives) if 'ln' in primitive]
        
        # so far hard code to value in darts.operations
        self.epsilon = torch.tensor(1e-10)
            
    def __iter__(self):
        return iter(_ops)

    def __getitem__(self, index):
        return self._ops[index]  # Return the item at the specified index

    def __len__(self):
        return len(self._ops)
        
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> float:
        """
        Computes a mixture operation as a weighted sum of all primitive operations.

        Arguments:
            x: input to the mixture operations
            weights: weight vector containing the weights associated with each operation

        Returns:
            y: result of the weighted mixture operation
        """
        # there are 8 weights for all the eight primitives. then it returns the
        # weighted sum of all operations performed on a given input
        #print("weights", weights, file=autora.theorist.darts.utils.buffer)
        #print("primitives", self.primitives, file=autora.theorist.darts.utils.buffer)
        #print("ops", self._ops, file=autora.theorist.darts.utils.buffer)
        #print("OP:", *[f"{weights[i]}*{self.primitives[i]}({x}) = w* {self._ops[i](x)}\nwith weights:\n{[str(module.weight.data)+'\n' if isinstance(module, torch.nn.Linear) else '' for module in self._ops[i].modules()]}" for i in range(len(weights))], sep="\n", file=autora.theorist.darts.utils.buffer)
        
        if True:#not FuckingContextClass.discrete_enabled:
            # Soft selection: weighted sum
            #print(weights)
            #print(self._ops)
            # also add batch dimension
            w=weights.unsqueeze(0).expand(len(x), *weights.shape).unsqueeze(-2)
            w_masked = w.clone()
            op_results=torch.stack([op(x) for op in self._ops], dim=-1)#(op(x) for op in self._ops)
            utils.Dump.latent_states.append({"inputs":x, **{prim:res for prim, res in zip(self.primitives,op_results.swapaxes(0,-1))}})
            #print(x.shape)
            wtf = op_results.abs()>y_th
            
            '''if wtf.any():
                print(op_results)
                print(op_results.shape)
                for batch in range(len(wtf)):
                    print(wtf[batch])
                    for i, b in enumerate(wtf[batch][0]):
                        if b:
                            print(op_results[batch][0][i], "=", self.primitives[i], "(", x[batch], ")", [param.detach() for param in self._ops[i].parameters()])
                print("----batch end----")'''
                        
            
            unsafe_areas=op_results.abs()==y_th
            if unsafe_areas.any().item():
                #print("-----safety-overflow-----")
                #print(unsafe_areas)
                w_masked[unsafe_areas]=w.detach()[unsafe_areas]
            '''if (op_results>=torch.exp(torch.tensor(9.9))).any() or (op_results<=-torch.exp(torch.tensor(9.9))).any():
                print("overflow?")
                #print(op_results[(op_results.abs()>=torch.exp(torch.tensor(9.9)))])
                print(op_results[unsafe_areas])'''
            #for idx in self.ln_idx:
            zeros=op_results==torch.log(self.epsilon)
            
            if zeros.any().item():
                #print("-----safety-zero-----")
                #print(zeros)
                w_masked[zeros]=w.detach()[zeros]
            #print(w_masked * op_results)
            out = (w_masked * op_results).sum(dim=-1)#w * op(x) for w, op in zip(w_masked, self._ops))
            #print(out)
        else:
            # Hard selection: one-hot, use only the selected op
            idx = torch.argmax(weights).item()
            out = self._ops[idx](x)
    
        return out
    
class DARTS(nn.Module):

    def __init__(
        self,
        primitives: Sequence[str] = PRIMITIVES,
        #temp: float = 1.0,
        size=None,
        n_vars=1,
        n_outputs=1,
        train_output_layer=True,
        init_range: float = 1e-3,
    ):
        """
        Initializes a cell based on the number of hidden nodes (steps)
        and the number of input nodes (n_input_states).

        Arguments:
            steps: number of hidden nodes
            n_input_states: number of input nodes
        """
        # The first and second nodes of cell k are set equal to the outputs of
        # cell k − 2 and cell k − 1, respectively, and 1 × 1 convolutions
        # (ReLUConvBN) are inserted as necessary
        super(DARTS, self).__init__()
        print(primitives)
        self.n_nodes=size
        self.n_vars=n_vars
        self.train_output_layer=train_output_layer
        self.init_range=init_range
        
        # unlike the alphas, MixedOp holds its coefficients itself
        # mixed_op = MixedOp(primitives)

        # layers = [DARTSLayer(shape[i], shape[i+1], primitives) for i in range(len(shape)-1)]
        self.nodes = nn.Sequential()  # *layers)

        self.alphas = nn.ParameterList()

        self.primitives = primitives

        # [layer.alphas for layer in self.layers]
        self.coefficients = nn.ParameterList()
        n_in=n_vars
        for _ in range(self.n_nodes):
            layer = DARTSNode(n_in, primitives, init_range=init_range)
            self.nodes.append(layer)
            self.alphas.append(layer.alphas)
            # can be flat, we just need easy access to it
            for coeff in layer.coefficients():
                self.coefficients.append(coeff)
            n_in+=1
            
        self.linear=LinearDARTSLayer(n_inputs=self.n_nodes, n_outputs=n_outputs, randinit=train_output_layer)

        if train_output_layer:
            for coeff in self.linear.parameters():
                self.coefficients.append(coeff)
        else:
            #just for safety
            self.linear.weights.requires_grad_(False)
            self.linear.biases.requires_grad_(False)
        #print("original output transformation:", list(self.layers[-1].parameters()))
        
        # [nn.ParameterList(layer.coefficients) for layer in self.layers]
        #attach_forward_nan_guards(self)

    def forward(self, x):
        inputs=x.clone()
        # more precicely: latent variables, but h is more common than l
        hidden_states=torch.empty(len(x), self.n_nodes)
        
        #print("hidden init size:", hidden_states.shape)
        for i, layer in enumerate(self.nodes):
            #print("safety check inputs per node:", x.shape)
            h=layer(x)
            hidden_states[:,i]=h.clone()
            #print("hidden_states[:,:i+1]")
            #print(hidden_states[:,:i+1].shape)
            #print("inputs")
            #print(inputs.shape)
            x=torch.cat([hidden_states[:,:i+1], inputs], dim=-1)
            #print("x\n",x.shape)
            # gotta make sure we are not getting a view
        #print("hidden size:", hidden_states.shape)
        x = self.linear(hidden_states)
        return x

    def fit(
        self,
        inputs,
        labels,
        batch_size=128,
        ratio=(1, 1, 0.5),
        n_epochs=1,
        device="cpu",
        monitor=None,
        finetune_epochs=10,
        arch_discretization="softmax",
        coeff_discretization="gs",
        param_learning_rate_max: float = 2.5e-2,
        param_learning_rate_min: float = 0.01,
        param_momentum: float = 9e-1,
        param_weight_decay: float = 3e-4,
        arch_learning_rate_max: float = 3e-3,

        arch_weight_decay: float = 1e-4,
        disable_tqdm=False,
        linear_clamp=1e10,
        coeff_opti="sgd",
        loss="mse",
        reset_adam=True,

        #arch_momentum: float = 9e-1,
    ):
        # weights and biases of output layer will never have bigger magnitute, to prevent nans
        self.linear_clamp=linear_clamp
        
        if loss=="mse":
            loss_fn=nn.functional.mse_loss
        elif loss=="nrmse":
            loss_fn=NRMSELoss()
        
        #run=None
        if monitor is None:
            monitor = Monitor()

        linear_component=any(["linear" in primitive for primitive in self.primitives])

        coefficient_step=True
        if (not self.train_output_layer) and (not linear_component):
            coefficient_step=False

        train_size = math.ceil(len(inputs) * ratio[0] / sum(ratio))
        val_size = math.ceil(len(inputs) * ratio[1] / sum(ratio))
        #test_size = math.ceil(len(inputs) * ratio[2] / sum(ratio))

        train_set = darts_dataset_from_ndarray(inputs[:train_size], labels[:train_size])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        val_set = darts_dataset_from_ndarray(
            inputs[train_size : train_size + val_size],
            labels[train_size : train_size + val_size],
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=True
        )

        # If ratios lead to rounding issues, that's test set's problem
        test_set = darts_dataset_from_ndarray(
            inputs[train_size + val_size :], labels[train_size + val_size :]
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
        
        architecture_optimizer = torch.optim.Adam(
            self.alphas,
            lr=arch_learning_rate_max,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )
        
        full_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size = len(train_loader.dataset),
            shuffle    = False,
            num_workers= 0,      # fast because we load only once
            pin_memory = True)

        x_train, y_train = next(iter(full_loader))

        # weird place to put closure
        def closure():
            coefficients_optimizer.zero_grad()
            y_pred = self(x_train.to(device))

            # no idea what the namespace looks like right now, better take safe way
            loss_ = loss_fn(y_pred, y_train.to(device))
            loss_.backward()
            '''if any(not torch.isfinite(p.grad).all() for p in self.parameters()):
                return torch.tensor(float('inf'))  # forces line-search to back-track'''
            return loss_#.detach()

        finetune_opti=torch.optim.LBFGS(
            params=self.coefficients,
            lr=1.0,
            max_iter=200,
        )
                
        if coefficient_step:
            # here!!!
            if coeff_opti=="adam":
                coefficients_optimizer = torch.optim.Adam(
                    params=self.coefficients,
                    lr=param_learning_rate_max,
                    #momentum=param_momentum,
                    weight_decay=param_weight_decay,
                )
                
                
            elif coeff_opti=="lbfgs":
                coefficients_optimizer=finetune_opti
                # bit awkward
                """full_loader = torch.utils.data.DataLoader(
                    train_loader.dataset,
                    batch_size = len(train_loader.dataset),
                    shuffle    = False,
                    num_workers= 0,      # fast because we load only once
                    pin_memory = True)

                x_train, y_train = next(iter(full_loader))

                # weird place to put closure
                def closure():
                    coefficients_optimizer.zero_grad()
                    y_pred = self(x_train.to(device))

                    # no idea what the namespace looks like right now, better take safe way
                    loss_ = loss_fn(y_pred, y_train.to(device))
                    loss_.backward()
                    '''if any(not torch.isfinite(p.grad).all() for p in self.parameters()):
                        return torch.tensor(float('inf'))  # forces line-search to back-track'''
                    return loss_#.detach()
                
                coefficients_optimizer = torch.optim.LBFGS(
                    params=self.coefficients,
                    lr=1.0,
                    max_iter=200,
                )"""
                
                
            elif coeff_opti=="sgd":                
                coefficients_optimizer = torch.optim.SGD(
                    params=self.coefficients,
                    lr=param_learning_rate_max,
                    momentum=param_momentum,
                    weight_decay=param_weight_decay,
                )
            else:
                raise ValueError(f"Unknown coeff optimizer {coeff_opti}")
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=coefficients_optimizer,
                T_max=n_epochs+finetune_epochs,
                eta_min=param_learning_rate_min,
            )

        # utilizing that len(train_loader)/ratio[0] == len(val_loader)/ratio[1]
        num_iterations = math.ceil(len(train_loader) / ratio[0])
        
        # values at init
        with torch.no_grad():
            i=0
            test_preds=[]
            losses_=[]
            for x, y_true in test_loader:
                          
                test_preds.append(self(x.to(device)))
                losses_.append(loss_fn(test_preds[-1], y_true.to(device)).detach().numpy())
            monitor.test_loss.append(np.mean(losses_))
                               
                    
                    
            '''
            monitor.test_loss.append(
                np.mean(
                    [
                        loss_fn(self(x.to(device)), y_true.to(device))
                        .detach()
                        .clone()
                        .numpy()
                        for x, y_true in test_loader
                    ]
                )
            )'''
            monitor.train_loss.append(
                np.mean(
                    [
                        loss_fn(self(x.to(device)), y_true.to(device))
                        .detach()
                        .clone()
                        .numpy()
                        for x, y_true in train_loader
                    ]
                )
            )
            monitor.val_loss.append(
                np.mean(
                    [
                        loss_fn(self(x.to(device)), y_true.to(device))
                        .detach()
                        .clone()
                        .numpy()
                        for x, y_true in val_loader
                    ]
                )
            )
            monitor.alphas.append(
                [
                    [
                        {
                            op: value
                            for op, value in zip(
                                self.primitives, layer.alphas[i, :].detach().clone()
                            )
                        }
                        for i in range(layer.n_inputs)
                    ]
                    for layer in self.nodes
                ]
            )
            if coefficient_step:
                monitor.coefficients=parameters_to_vector(self.coefficients).unsqueeze(-1)
        test_loss = 0

        if isinstance(n_epochs, int):
            epochs = range(n_epochs)
            total = n_epochs
        else:
            epochs = itertools.count(start=0)
            total = 100

            
        #versions=[weight._version for weight in self.alphas]
        #print("initial versions", versions)
        print(f"starting main training with {coeff_opti}")
        tqdm_bar = tqdm(epochs, leave=True, total=total, disable = disable_tqdm)
        start_time = time.time()
        for epoch in tqdm_bar:
            
            tqdm_bar.set_postfix({"Test Loss": f"{test_loss:.5f}"})
            train_loss_batch_wise = []
            val_loss_batch_wise = []

            # Convert DataLoaders to iterators
            arch_iter = iter(val_loader)
            train_iter = iter(train_loader)

            # Iterate over the pre-determined number of chunks
            for _ in range(num_iterations):
                # Process chunk from loader1
                for x, y_true in process_chunks(arch_iter, ratio[1]):
                    
                    architecture_optimizer.zero_grad()

                    if arch_discretization == "gs":
                        with gumbel_softmax_context():
                            y_pred = self(x.to(device))
                    elif arch_discretization == "max":
                        raise RuntimeError("max not differentiable wrt alphas")
                    elif arch_discretization == "softmax":
                        y_pred = self(x.to(device))
                    else:
                        raise RuntimeError("no valid achr discretization")

                    val_loss = loss_fn(y_pred, y_true.to(device))
                    val_loss_batch_wise.append(val_loss.detach().clone().numpy())
                    val_loss.backward()
                    
                    #assert versions==[weight._version for weight in self.alphas]
                    
                    '''for name, p in self.named_parameters():
                        if not torch.isfinite(p).all():
                            print(name, f"became {p} before arch-optimizer.step")
                            break'''
                    architecture_optimizer.step()
                    utils.Dump.latent_states=[]
                    '''for name, p in self.named_parameters():
                        if not torch.isfinite(p).all():
                            print(name, f"became {p} during arch-optimizer.step")
                            break'''

                    
                    #versions=[weight._version for weight in self.alphas]
                    monitor.alphas.append(
                        [
                            [
                                {
                                    op: value
                                    for op, value in zip(
                                        self.primitives, layer.alphas[i, :].detach().clone()
                                    )
                                }
                                for i in range(layer.n_inputs)
                            ]
                            for layer in self.nodes
                        ]
                    )
    
                if coefficient_step:
                    if coeff_opti=="lbfgs":
                        try:
                            
                        
                            if coeff_discretization=="gs":
                                with gumbel_softmax_context():
                                    loss_ = coefficients_optimizer.step(closure).detach()
                            elif coeff_discretization=="max":
                                with discrete_context():
                                    loss_ = coefficients_optimizer.step(closure).detach()
                            elif coeff_discretization == "softmax":
                                loss_ = coefficients_optimizer.step(closure).detach()
                            else:
                                raise RuntimeError("no valid coeff discretization")
                            _clamp_weights(self)
                                
                        except RuntimeError as err:
                            # if lbfgs fails
                            print(err)
                            loss_=np.nan
                        train_loss_batch_wise=np.array([loss_])
                        
                        coefficients_optimizer.state.clear()
                        
                    else:
                        # Process chunk from loader2
                        for x, y_true in process_chunks(train_iter, ratio[0]):
                            coefficients_optimizer.zero_grad()
                            if coeff_discretization == "gs":
                                with gumbel_softmax_context():
                                    y_pred = self(x.to(device))
                            elif coeff_discretization == "max":
                                with discrete_context():
                                    y_pred = self(x.to(device))
                            elif coeff_discretization == "softmax":
                                y_pred = self(x.to(device))
                            else:
                                raise RuntimeError("no valid coeff discretization")

                            train_loss = loss_fn(y_pred, y_true.to(device))
                            train_loss_batch_wise.append(train_loss.detach().clone().numpy())
                            train_loss.backward()


                            '''for name, p in self.named_parameters():
                                if not torch.isfinite(p).all():
                                    print(name, f"became {p} before coeff-optimizer.step")
                                    break'''

                            coefficients_optimizer.step()

                            '''utils.Dump.latent_states=[]
                            for name, p in self.named_parameters():
                                if not torch.isfinite(p).all():
                                    print(name, f"became {p} during coeff-optimizer.step")
                                    break'''
                            '''monitor.alphas.append(
                                [
                                    [
                                        {
                                            op: value
                                            for op, value in zip(
                                                self.primitives, layer.alphas[i, :].detach().clone()
                                            )
                                        }
                                        for i in range(layer.n_inputs)
                                    ]
                                    for layer in self.nodes
                                ]
                            )'''
                            # dude, lets please not monitor coefficients all the timme...
                            if False:
                                monitor.coefficients=torch.cat((monitor.coefficients, parameters_to_vector(self.coefficients).detach().unsqueeze(-1)), dim=-1)

                            # nice idea, but if this is a problem I should rather use adam or clip grads
                            if False:
                                with torch.no_grad():
                                    self.linear.weights.clamp_(min=-self.linear_clamp, max= self.linear_clamp)
                                    self.linear.biases.clamp_(min=-self.linear_clamp, max= self.linear_clamp)

                        scheduler.step()
                    

            with torch.no_grad():
                monitor.test_loss.append(
                    np.mean(
                        [
                            loss_fn(self(x.to(device)), y_true.to(device))
                            .detach()
                            .clone()
                            .numpy()
                            for x, y_true in test_loader
                        ]
                    )
                )
            monitor.train_loss.append(np.mean(train_loss_batch_wise))
            monitor.val_loss.append(np.mean(val_loss_batch_wise))

            test_loss = monitor.test_loss[-1]
            if isinstance(n_epochs, str):
                # print(int(n_epochs)*60)
                # print(time.time()-start_time)
                if time.time() - start_time > float(n_epochs) * 60:
                    break
            if np.isnan(test_loss):
                return self, monitor
            
        
        # finetuning
        if coefficient_step:
            backup_state = copy.deepcopy(self.state_dict())
            # test loss not allowed here
            backup_loss=monitor.train_loss[-1]

            coefficients_optimizer=finetune_opti
            # do some coefficient fine-tuning                        
            print(f"starting finetune with {coefficients_optimizer}")

            # overall max steps for lbfgs = 200*(finetune_epochs//10)
            # avg steps: no idea
            for _ in tqdm(range(finetune_epochs//10+1), disable = disable_tqdm):
                try:
                    with discrete_context():
                        loss_ = coefficients_optimizer.step(closure).detach()
                        _clamp_weights(self)
                        #_clamp_weights(self)
                except RuntimeError as err:
                    # if lbfgs fails
                    print(err)
                    loss_=np.nan

                train_loss_batch_wise=np.array([loss_])
                
                test_losses=[]
                train_losses=[]
                # values during fine-tuning
                with torch.no_grad():
                    test_losses.append(
                        np.mean(
                            [
                                loss_fn(
                                    self(x.to(device)), y_true.to(device)
                                )
                                .detach()
                                .clone()
                                .numpy()
                                for x, y_true in test_loader
                            ]
                        )
                    )
                    train_losses.append(np.mean(train_loss_batch_wise))
                
            if not np.isnan(train_losses).any():
                monitor.test_loss+=test_losses
                monitor.train_loss+=train_losses
                     
            else:
                # if lbfgs failed, retry with adam (safer but tends to give worse results)
                
                print("Finetune failed. Rewinding.")
                self.load_state_dict(backup_state)
                
                coefficients_optimizer=torch.optim.Adam(
                    params=self.coefficients,
                    lr=param_learning_rate_max,
                    #momentum=param_momentum,
                    weight_decay=param_weight_decay,
                )
                print(f"retry finetune with {coefficients_optimizer}")

                test_losses=[]
                train_losses=[]
                
                for _ in tqdm(range(finetune_epochs), disable = disable_tqdm):
                    train_loss_batch_wise=[]
                    for x, y_true in iter(train_loader):
                        coefficients_optimizer.zero_grad()
                        with discrete_context():
                            y_pred = self(x.to(device))
                        train_loss = loss_fn(y_pred, y_true.to(device))
                        train_loss_batch_wise.append(train_loss.detach().clone().numpy())
                        train_loss.backward()
                        coefficients_optimizer.step()

                    # values during fine-tuning
                    with torch.no_grad():
                        test_losses.append(
                            np.mean(
                                [
                                    loss_fn(
                                        self(x.to(device)), y_true.to(device)
                                    )
                                    .detach()
                                    .clone()
                                    .numpy()
                                    for x, y_true in test_loader
                                ]
                            )
                        )
                        train_losses.append(np.mean(train_loss_batch_wise))
                
                print(f"Loss before finetuning ({backup_loss})\n      after finetuning ({train_losses[-1]})")
                if train_losses[-1]>backup_loss:
                    print("Finetune failed. Rewinding.")
                    self.load_state_dict(backup_state)

                else:
                    monitor.train_loss+=train_losses
                    monitor.test_loss+=test_losses
                

        #print("trained output transformation:", list(self.layers[-1].parameters()))
        return self, monitor

    def to_sympy(self, variable_prefix):
        variables = sympy.symbols([variable_prefix+str(i+1) for i in range(self.n_vars)])
        result = variables
        hidden_states=[]
        print("Convert to sympy")
        for layer in tqdm(self.nodes):
            h = layer.to_sympy(result)
            hidden_states.append(h)
            result = [*hidden_states,*variables]
        result=self.linear.to_sympy(hidden_states)
        return result[0]

class LinearDARTSLayer(nn.Module):
    # Not using torch.nn.Linear, so that to_sympy is easier (honestly, built in self.weights would have done the job as well...)
    def __init__(self, n_inputs: int=2, n_outputs: int=1, randinit=True):
        super(LinearDARTSLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        if randinit:
            some_linear=nn.Linear(n_inputs,n_outputs)
            self.weights=nn.Parameter(some_linear.weight.detach())
            self.biases=nn.Parameter(some_linear.bias.detach()[:,None])
        else:
            self.weights = nn.Parameter(torch.ones((n_outputs, n_inputs)))
            # since  we have no non_linearity, I guess this is incorrect by factor two, not sure in which direction
            #torch.nn.init.kaiming_uniform_(self.weights, nonlinearity='relu')
            self.biases=nn.Parameter(torch.zeros((n_outputs, 1)))
        
    def forward(self, x):
        #print("x in:", x.shape)
        #print(self.weights.shape)
        #print(self.biases.shape)
        x_out=x@self.weights.T+self.biases.T
        #print("x out:", x_out.shape)
        return x_out
        
    def to_sympy(self, inputs):
        layer_output=[sum([self.weights[j,i]*inputs[i] for i in range(self.n_inputs)])+self.biases[j] for j in range(self.n_outputs)]
        return layer_output

class DARTSNode(nn.Module):
    def __init__(
        self,
        n_inputs: int = 2,
        primitives: Sequence[str] = PRIMITIVES,
        temp: float = 1.0,  # 5#0.5
        init_range: float = 1e-3,
    ):
        super(DARTSNode, self).__init__()
        self.n_ops = len(primitives)
        self.n_inputs = n_inputs
        # obsolete
        self.temp = temp
        
        if any('safe' in w for w in primitives):
            print("extra safety (no)")
            mixed_ops = [SafeMixedOp(primitives) for _ in range(self.n_inputs)]
        else:
            # for now, lets keep it flat, lets see if we can vectorize later
            mixed_ops = [MixedOp(primitives) for _ in range(self.n_inputs)]
        self.mixed_ops = nn.ModuleList(mixed_ops)
        # same init as in autora.darts, no idea if that makes sense, but since they are softmaxed anyways, who cares as long as they are random
        rand=torch.randn(self.n_inputs, self.n_ops)
        self.alphas = nn.Parameter(
            init_range * rand,
            requires_grad=True,
        )
        
        #self.alphas.register_hook(nan_guard)
        # No, wee need empty lists for ops without params as placehoders for indexing, no again, to_sympy retrieves params by hand
        self.coefficients = self.mixed_ops.parameters
        #self.coefficients=[]
        #list(self.coefficients())[i * len(current_ops) + j][winner]
        #i=input, j=current op
        
        '''print("coeff:", list(self.coefficients))
        print("coeff:", list(self.coefficients))
        print("mixed ops params:", list(self.mixed_ops.parameters()))
        print("mixed ops params:", list(self.mixed_ops.parameters()))'''
        self.primitives = primitives

    def forward(self, x):
        #assert len(x[0])==self.n_inputs
        #x = x.unsqueeze(-1).expand(*x.shape, self.n_outputs)
        operation_outputs = torch.empty(*x.shape)
        # mixed ops is flattened...
        operation_iterator = iter(self.mixed_ops)

        # not so runtime efficient but at least good for memory
        for i in range(self.n_inputs):
            op = self.mixed_ops[i]#tuple(islice(operation_iterator, self.n_outputs))

            if FuckingContextClass.discrete_enabled:
                operation_outputs[:, i] = op(
                    x[:, i].unsqueeze(-1),
                    F.one_hot(
                        torch.argmax(self.alphas[i, :]), num_classes=self.n_ops
                    ),
                ).squeeze()
            elif FuckingContextClass.gumbel_softmax_enabled:
                soft_discrete = F.gumbel_softmax(
                    self.alphas[i, :] / self.temp, hard=True, tau=1
                )
                operation_outputs[:, i] = op(
                    x[:, i].unsqueeze(-1), soft_discrete
                ).squeeze()
            else:
                # print(self.alphas[i,j,:])
                # print(F.softmax(self.alphas[i,j,:]/self.temp))
                operation_outputs[:, i] = op(
                    x[:, i].unsqueeze(-1),
                    F.softmax(self.alphas[i, :] / self.temp, dim=-1).clone(),
                    #debug_softmax(self.alphas[i, :] / self.temp, dim=-1).clone(),
                ).squeeze()
                '''y=F.softmax(self.alphas[i, :] / self.temp, dim=-1).detach()
                if not torch.isfinite(y).all():
                    bad = ~torch.isfinite(y)
                    rc = bad.nonzero(as_tuple=True)
                    print("Soft-max NaN/Inf at", rc)
                    print("for logits", self.alphas[i,:])
                    print("temp sane?", self.temp)'''
        return torch.sum(operation_outputs, dim=-1)

    def to_sympy(self, inputs):
        assert len(inputs)==self.n_inputs
        #x = x.unsqueeze(-1).expand(*x.shape, self.n_outputs)
        #operation_outputs = torch.empty(*x.shape)
        operation_iterator = iter(self.mixed_ops)

        op_results = [None for _ in range(self.n_inputs)]

        # iterate over inputs
        for i in range(self.n_inputs):
            mixed_op=self.mixed_ops[i]# =  self.mixed_ops[i*self.n_outputs:(i+1)*self.n_outputs]#tuple(islice(operation_iterator, self.n_outputs))

            winner = torch.argmax(self.alphas[i, :])

            #print("coeffs:", list(self.coefficients()))
            #coefficients = list(self.coefficients())[i * len(current_ops) + j][winner]
            coefficients = list(mixed_op[winner].parameters())

            op_result = get_operation_as_sympy(
                self.primitives[winner], coefficients, inputs[i]
            )
            op_results[i] = op_result

        layer_output = sum(op_results)
        #print(layer_output)
        return layer_output