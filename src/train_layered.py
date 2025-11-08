# todo: last layer output no nonlinearity
# from utils import Monitor
import numpy as np
import wandb
import torch
import utils
import sys
import os
import time
import math
from fickdich import FuckingContextClass
from contextlib import contextmanager
from torch import nn
from torch.nn import functional as F
from typing import Sequence
from autora.theorist.darts.utils import Monitor
from autora.theorist.darts.dataset import darts_dataset_from_ndarray
from autora.theorist.darts.model_search import MixedOp
from autora.theorist.darts.operations import get_operation_as_sympy
from itertools import islice

#from itertools import islice
import itertools
from tqdm import tqdm
import sympy


sys.path.append(os.path.normpath(os.getcwd() + "/autora-theorist-darts/src"))
torch.set_default_dtype(torch.float64)

from autora.theorist.darts import DARTSExecutionMonitor, DARTSRegressor
from autora.theorist.darts.utils import NanError

# sounds small, but for 1-3d data its pretty dense
n_samples = 200  # 200

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

"""class DummyRun():
    def """


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


class LayeredDARTS(nn.Module):
    
    def __init__(
        self,
        shape=None,  #: list = [2,2,1],
        primitives: Sequence[str] = PRIMITIVES,
        temp: float = 1.0,
        size=None,
        n_in=None,
        n_out=None,
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
        super(LayeredDARTS, self).__init__()
        self.train_output_layer=train_output_layer

        # kinda arbitrary, lets see
        if shape is None:
            shape = [n_in]
            shape += [n_in * 2 + 1 + i for i in reversed(range(size+1))]
            shape += [n_out]
        self.shape=shape
        print("shape:",self.shape)
        # unlike the alphas, MixedOp holds its coefficients itself
        # mixed_op = MixedOp(primitives)

        # layers = [DARTSLayer(shape[i], shape[i+1], primitives) for i in range(len(shape)-1)]
        self.layers = nn.Sequential()  # *layers)

        self.alphas = nn.ParameterList()

        self.primitives = primitives

        # [layer.alphas for layer in self.layers]
        self.coefficients = nn.ParameterList()
        for i in range(len(shape) - 2):
            layer = DARTSLayer(shape[i], shape[i + 1], primitives, temp, init_range)
            self.layers.append(layer)
            self.alphas.append(layer.alphas)
            # can be flat, we just need easy access to it
            for coeff in layer.coefficients():
                self.coefficients.append(coeff)
        self.layers.append(LinearDARTSLayer(n_inputs=shape[-2], n_outputs=shape[-1], randinit=train_output_layer))
        
        if train_output_layer:
            for coeff in self.layers[-1].parameters():
                self.coefficients.append(coeff)
        else:
            #just for safety
            self.layers[-1].weights.requires_grad_(False)
            self.layers[-1].biases.requires_grad_(False)
        #print("original output transformation:", list(self.layers[-1].parameters()))
        
        # [nn.ParameterList(layer.coefficients) for layer in self.layers]

    def forward(self, x):
        x = self.layers(x)
        #print(x)
        return x

    def fit(
        self,
        inputs,
        labels,
        batch_size=128,
        ratio=(2, 2, 1),
        n_epochs=1,
        device="cpu",
        monitor=None,
        finetune_epochs=10,
        wandb_run=None,
        arch_discretization="softmax",
        coeff_discretization="gs",
        param_learning_rate_max: float = 2.5e-2,
        param_learning_rate_min: float = 0.01,
        param_momentum: float = 9e-1,
        param_weight_decay: float = 3e-4,
        arch_learning_rate_max: float = 3e-3,

        arch_weight_decay: float = 1e-4,

        arch_momentum: float = 9e-1,
        
        disable_tqdm=False,
        linear_clamp=1e10,

    ):
        # weights and biases of output layer will never have bigger magnitute, to prevent nans
        self.linear_clamp=linear_clamp
        
        run=None
        if monitor is None:
            monitor = Monitor()

        train_size = math.ceil(len(inputs) * ratio[0] / sum(ratio))
        val_size = math.ceil(len(inputs) * ratio[1] / sum(ratio))
        test_size = math.ceil(len(inputs) * ratio[2] / sum(ratio))

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

        
        coefficient_step=True
        if (not self.train_output_layer) and (not linear_component):
            coefficient_step=False
            
        architecture_optimizer = torch.optim.Adam(
            self.alphas,
            lr=arch_learning_rate_max,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

        if coefficient_step:
            coefficients_optimizer = torch.optim.SGD(
                params=self.coefficients,
                lr=param_learning_rate_max,
                momentum=param_momentum,
                weight_decay=param_weight_decay,
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=coefficients_optimizer,
                T_max=n_epochs,
                eta_min=param_learning_rate_min,
            )
        
        '''
        architecture_optimizer = torch.optim.Adam(self.alphas, lr=1e-3)
        coefficients_optimizer = torch.optim.Adam(self.coefficients, lr=1e-3)'''

        # utilizing that len(train_loader)/ratio[0] == len(val_loader)/ratio[1]
        num_iterations = math.ceil(len(train_loader) / ratio[0])

        # values at init
        with torch.no_grad():
            monitor.test_loss.append(
                np.mean(
                    [
                        nn.functional.mse_loss(self(x.to(device)), y_true.to(device))
                        .detach()
                        .clone()
                        .numpy()
                        for x, y_true in test_loader
                    ]
                )
            )
            monitor.train_loss.append(
                np.mean(
                    [
                        nn.functional.mse_loss(self(x.to(device)), y_true.to(device))
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
                        nn.functional.mse_loss(self(x.to(device)), y_true.to(device))
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
                                self.primitives, layer.alphas[i, j, :].detach().clone()
                            )
                        }
                        for j in range(layer.n_outputs)
                        for i in range(layer.n_inputs)
                    ]
                    for layer in self.layers[:-1]
                ]
            )

        test_loss = 0

        if isinstance(n_epochs, int):
            epochs = range(n_epochs)
            total = n_epochs
        else:
            epochs = itertools.count(start=0)
            total = 100

        tqdm_bar = tqdm(epochs, leave=True, total=total, disable=disable_tqdm)
        start_time = time.time()
        for epoch in tqdm_bar:
            """
            print("new epoch")
            print(self.alphas[0])
            print(self.coefficients[0])"""

            tqdm_bar.set_postfix({"Test Loss": f"{test_loss:.5f}"})
            train_loss_batch_wise = []
            val_loss_batch_wise = []

            # Convert DataLoaders to iterators
            arch_iter = iter(val_loader)
            train_iter = iter(train_loader)
            '''
            print(ratio)
            print(train_size, val_size, test_size)
            '''
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

                    val_loss = nn.functional.mse_loss(y_pred, y_true.to(device))
                    val_loss_batch_wise.append(val_loss.detach().clone().numpy())
                    val_loss.backward()
                    architecture_optimizer.step()

                if coefficient_step:
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
    
                        train_loss = nn.functional.mse_loss(y_pred, y_true.to(device))
                        train_loss_batch_wise.append(train_loss.detach().clone().numpy())
                        train_loss.backward()
                        coefficients_optimizer.step()
                        with torch.no_grad():
                            self.layers[-1].weights.clamp_(min=-self.linear_clamp, max= self.linear_clamp)
                            self.layers[-1].biases.clamp_(min=-self.linear_clamp, max= self.linear_clamp)

                # Its a bit a matter of taste when/how often to do scheduler step...
                scheduler.step()


            monitor.test_loss.append(
                np.mean(
                    [
                        nn.functional.mse_loss(self(x.to(device)), y_true.to(device))
                        .detach()
                        .clone()
                        .numpy()
                        for x, y_true in test_loader
                    ]
                )
            )
            monitor.train_loss.append(np.mean(train_loss_batch_wise))
            monitor.val_loss.append(np.mean(val_loss_batch_wise))
            if np.isnan(monitor.train_loss[-1]):
                return self, monitor
            monitor.alphas.append(
                [
                    [
                        {
                            op: value
                            for op, value in zip(
                                self.primitives, layer.alphas[i, j, :].detach().clone()
                            )
                        }
                        for j in range(layer.n_outputs)
                        for i in range(layer.n_inputs)
                    ]
                    for layer in self.layers[:-1]
                ]
            )
            if not (wandb_run is None):
                print("!!!!!!should not reach!!!!!!")
                alphas = monitor.alphas[-1]
                wandb_run.log(
                    {
                        "test_loss": monitor.test_loss[-1],
                        "train_loss": monitor.train_loss[-1],
                        "val_loss": monitor.val_loss[-1],  # "grid": grid,
                        **{
                            f"{key}_{layer}_{i}": alphas[layer][i][key]
                            for layer in range(len(alphas))
                            for i in range(len(alphas[layer]))
                            for key in alphas[layer][i].keys()
                        },
                    }
                )
            test_loss = monitor.test_loss[-1]
            if isinstance(n_epochs, str):
                # print(int(n_epochs)*60)
                # print(time.time()-start_time)
                if time.time() - start_time > float(n_epochs) * 60:
                    break
            if np.isnan(test_loss):
                return self, monitor
            
        if np.isnan(test_loss):
            return self, monitor

        if coefficient_step:
            # do some coefficient fine-tuning
            for _ in tqdm(range(finetune_epochs)):
                train_loss_batch_wise=[]
                for x, y_true in iter(train_loader):
                    coefficients_optimizer.zero_grad()
                    with discrete_context():
                        y_pred = self(x.to(device))
                    train_loss = nn.functional.mse_loss(y_pred, y_true.to(device))
                    train_loss_batch_wise.append(train_loss.detach().clone().numpy())
                    train_loss.backward()
                    coefficients_optimizer.step()
    
                # values after fine-tuning
                with torch.no_grad():
                    monitor.test_loss.append(
                        np.mean(
                            [
                                nn.functional.mse_loss(
                                    self(x.to(device)), y_true.to(device)
                                )
                                .detach()
                                .clone()
                                .numpy()
                                for x, y_true in test_loader
                            ]
                        )
                    )
                    monitor.train_loss.append(np.mean(train_loss_batch_wise))
                    monitor.val_loss.append(
                        np.mean(
                            [
                                nn.functional.mse_loss(
                                    self(x.to(device)), y_true.to(device)
                                )
                                .detach()
                                .clone()
                                .numpy()
                                for x, y_true in val_loader
                            ]
                        )
                    )

        #print("trained output transformation:", list(self.layers[-1].parameters()))
        return self, monitor

    def to_sympy(self, variable_prefix):
        variables = sympy.symbols([variable_prefix+str(i+1) for i in range(self.shape[0])])
        result = variables
        for layer in tqdm(self.layers):
            result = layer.to_sympy(result)
        return result

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

class DARTSLayer(nn.Module):
    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 2,
        primitives: Sequence[str] = PRIMITIVES,
        temp: float = 1.0,  # 5#0.5
        init_range=1e-3,
    ):
        super(DARTSLayer, self).__init__()
        self.n_ops = len(primitives)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.temp = temp

        # for now, lets keep it flat, lets see if we can vectorize later
        mixed_ops = [MixedOp(primitives) for _ in range(self.n_inputs * n_outputs)]
        self.mixed_ops = nn.ModuleList(mixed_ops)
        # same init as in autora.darts, no idea if that makes sense, but since they are softmaxed anyways, who cares as long as they are random
        self.alphas = nn.Parameter(
            init_range * torch.randn(self.n_inputs, self.n_outputs, self.n_ops),
            requires_grad=True,
        )
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
        x = x.unsqueeze(-1).expand(*x.shape, self.n_outputs)
        operation_outputs = torch.empty(*x.shape)
        operation_iterator = iter(self.mixed_ops)

        # not so runtime efficient but at least good for memory
        for i in range(self.n_inputs):
            current_ops = self.mixed_ops[i*self.n_outputs:(i+1)*self.n_outputs]#tuple(islice(operation_iterator, self.n_outputs))

            for j, op in enumerate(current_ops):
                # print(gumbel_softmax_enabled)
                # should alphas be expanded before gumbel_softmax?
                if FuckingContextClass.discrete_enabled:
                    operation_outputs[:, i, j] = op(
                        x[:, i, j].unsqueeze(-1),
                        F.one_hot(
                            torch.argmax(self.alphas[i, j, :]), num_classes=self.n_ops
                        ),
                    ).squeeze()
                elif FuckingContextClass.gumbel_softmax_enabled:
                    soft_discrete = F.gumbel_softmax(
                        self.alphas[i, j, :] / self.temp, hard=True, tau=1
                    )
                    operation_outputs[:, i, j] = op(
                        x[:, i, j].unsqueeze(-1), soft_discrete
                    ).squeeze()
                else:
                    # print(self.alphas[i,j,:])
                    # print(F.softmax(self.alphas[i,j,:]/self.temp))
                    operation_outputs[:, i, j] = op(
                        x[:, i, j].unsqueeze(-1),
                        F.softmax(self.alphas[i, j, :] / self.temp, dim=-1),
                    ).squeeze()
                    # print(operation_outputs[:,i,j])
        layer_output = torch.sum(operation_outputs, dim=-2)
        # print(layer_output)
        return layer_output

    def to_sympy(self, inputs):

        #x = x.unsqueeze(-1).expand(*x.shape, self.n_outputs)
        #operation_outputs = torch.empty(*x.shape)
        operation_iterator = iter(self.mixed_ops)

        op_results = [
            [None for _ in range(self.n_inputs)] for _ in range(self.n_outputs)
        ]

        # iterate over inputs
        for i in range(self.n_inputs):
            current_ops =  self.mixed_ops[i*self.n_outputs:(i+1)*self.n_outputs]#tuple(islice(operation_iterator, self.n_outputs))

            # iterate over mixture ops/layer nodes
            for j, mixed_op in enumerate(current_ops):
                winner = torch.argmax(self.alphas[i, j, :])

                #print("coeffs:", list(self.coefficients()))
                #coefficients = list(self.coefficients())[i * len(current_ops) + j][winner]
                coefficients = list(current_ops[j][winner].parameters())

                op_result = get_operation_as_sympy(
                    self.primitives[winner], coefficients, inputs[i]
                )
                op_results[j][i] = op_result

        layer_output = [sum(op_results[j]) for j in range(self.n_outputs)]
        #print(layer_output)
        return layer_output


def process_chunks(iterator, chunk_size):
    assert chunk_size==int(chunk_size)
    chunk_size=int(chunk_size)
    for _ in range(chunk_size):
        try:
            yield next(iterator)
        except StopIteration:
            return


class Bla:
    def __init__(self, trainloss, valloss, testloss):
        self.test_loss = testloss
        self.val_loss = valloss
        self.train_loss = trainloss


def train2(
    run,
    config,
    dataset,
    equation,
    training_seed,
    n_nodes=3,
    ratio=(1, 1, 0.5),
    id_="noname",
):
    # print(id_)

    seed = int(time.time() * 1e6) % 3000
    np.random.seed(seed)
    test = np.random.random(100)
    train = np.random.random(100)
    val = np.random.random(80)
    """print("logs:",test)
    for i, x in enumerate(test):
        wandb.log({f"test_{id_}": x}, step=i)"""
    return Bla(train, val, test), None, "bla"


def train(
    config, dataset, training_seed, ratio=(1, 1, 0.5), id_="noname", disable_tqdm=False, scale_steps=1,
):

    #size = utils.size_mapping(size, dataset["train_input"].shape[-1])
    # id_+=f"_{training_seed}"
    # monitor = Monitor()#DARTSExecutionMonitor()

    epochs = int(config["batch_size"]*42*scale_steps)

    # seed for training and model init
    np.random.seed(training_seed)
    torch.random.manual_seed(training_seed)

    x = dataset["train_input"]
    y = dataset["train_label"]
    train_size = math.ceil(len(x) * ratio[0] / sum(ratio))
    val_size = math.ceil(len(x) * ratio[1] / sum(ratio))
    test_size = math.ceil(len(x) * ratio[2] / sum(ratio))

    batch_size = config["batch_size"]
    param_updates_per_epoch = train_size // batch_size  # todo: should be floor
    coeff_updates_per_epoch = train_size // batch_size

    train_set = darts_dataset_from_ndarray(x[:train_size], y[:train_size])

    val_set = darts_dataset_from_ndarray(
        x[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )

    # If ratios lead to rounding issues, that's test set's problem
    test_set = darts_dataset_from_ndarray(
        x[train_size + val_size :], y[train_size + val_size :]
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    shape=[len(x[0]), *[i for i in range(config["size"]+1, 0, -1)], 1]
    print(shape)
    print(x[0], config["size"])
    
    # model = DARTSRegressor([2,2,1], primitives,temp=config["temp"],)
    try:
        model = LayeredDARTS(
            shape=shape,
            n_in=dataset["train_input"].shape[-1],
            n_out=dataset["train_label"].shape[-1],
            primitives=config["primitives"],
        )
        print(model.shape)

        model, monitor = model.fit(
            dataset["train_input"],
            dataset["train_label"],
            batch_size=config["batch_size"],
            ratio=ratio,
            n_epochs=epochs,
            finetune_epochs=config["finetune_epochs"],
            wandb_run=None,#wandb.run,
            arch_discretization=config["arch_discretization"],
            coeff_discretization=config["coeff_discretization"],
            arch_learning_rate_max=config["arch_learning_rate_max"],
            arch_momentum=config["arch_momentum"],
            param_learning_rate_max=config["arch_momentum"],
            param_momentum=config["arch_momentum"],
            disable_tqdm=disable_tqdm,
        )

        outcome = "normal"

    # for debugging, currently turned off
    except NanError as e:
        print("nan error:", monitor.train_loss, sep="\n")
        raise e
        # pass
    if any(np.isnan(loss) for loss in monitor.test_loss):
        print("nan catched:")
        print(monitor.test_loss[-10:])
        primitives = config["primitives"].copy()
        # if config["safety"]=="safe":#, "smooth", "ramped"]
        for i, primitive in enumerate(primitives):
            if primitive in ("power_two", "power_three", "exp", "reciprocal"):
                primitives[i] = "safe_" + primitives[i]
            if config["safety"] == "smooth":
                if primitive in ("power_two", "power_three", "exp"):
                    primitives[i] = primitives[i] + "_smooth"
            elif config["safety"] == "ramped":
                if primitive in ("power_two", "power_three"):
                    primitives[i] = primitives[i] + "_smooth"
                elif primitive in ("exp"):
                    primitives[i] = primitives[i] + "_ramped"

        # print(primitives)

        model = LayeredDARTS(
            shape=shape,
            n_in=dataset["train_input"].shape[-1],
            n_out=dataset["train_label"].shape[-1],
            primitives=primitives,
        )
        model, monitor = model.fit(
            dataset["train_input"],
            dataset["train_label"],
            batch_size=config["batch_size"],
            ratio=ratio,
            n_epochs=epochs,
            finetune_epochs=config["finetune_epochs"],
            wandb_run=None,#wandb.run,
            arch_discretization=config["arch_discretization"],
            coeff_discretization=config["coeff_discretization"],
            arch_learning_rate_max=config["arch_learning_rate_max"],
            arch_momentum=config["arch_momentum"],
            param_learning_rate_max=config["arch_momentum"],
            param_momentum=config["arch_momentum"],
        )


        if any(np.isnan(loss) for loss in monitor.test_loss):
            outcome = "failed"
        else:
            outcome = "safe"
        print(monitor.train_loss[-1])

    # print(model.execution_monitor.test_loss)
    # print(len(model.execution_monitor.test_loss))
    # monitor = model.execution_monitor
    train_log = {
        "train_loss": np.array([]),
        "test_loss": np.array([]),
        "alphas": np.array([]),
    }  # stupid
    train_log["train_loss"] = monitor.train_loss
    train_log["test_loss"] = monitor.test_loss
    train_log["val_loss"] = monitor.val_loss
    # train_log["alphas"] = monitor.arch_weight_history

    # config["train_loss"] = train_log["train_loss"]
    # config["test_loss"] = train_log["test_loss"]
    # config["alphas"] = train_log["alphas"]

    #variables = equation.free_symbols
    predicted_equation = model.to_sympy("x_")

    # rounded = utils.simplify(predicted_equation)

    # config["predicted_equation"] = predicted_equation

    # utils.save_experiment_config(config, folder="configs_layered_darts")
    # monitor.predicted_equation = str(predicted_equation)
    """
    if not (run is None):
        run.summary["predicted_equation"] = str(predicted_equation)
        alphas=monitor.alphas
        primitives=config["primitives"]
        # fuck logging alphas
        
        try:
            overall_log={f"test_loss_{id_}": monitor.test_loss, f"train_loss_{id_}": monitor.train_loss, f"val_loss_{id_}": monitor.val_loss,}
            #     **{f"{primitives[key]}_{edge}": [alphas[iter_][edge][key] for iter_ in range(len(alphas))] for edge in range(len(alphas[0])) for key in range(len(primitives))}}
            
            #print(overall_log)
            for i in range(len(overall_log[f"val_loss_{id_}"])):
                run.log({str(key): overall_log[key][i] for key in overall_log.keys()}, step=0)
                #print({str(key): overall_log[key][i] for key in overall_log.keys()})
            for i in range(len(overall_log[f"val_loss_{id_}"]), len(overall_log[f"train_loss_{id_}"]), ):
                run.log({str(key): overall_log[key][i] for key in (f"train_loss_{id_}", f"test_loss_{id_}")})
            
            
        except IndexError:
            print("-------------------------")
            print(alphas)
            print("-------------------------")
            print(primitives)
            raise"""

    # predicted_equation would be multi dimensional in case of multiple outputs
    return monitor, model, predicted_equation[0], outcome
