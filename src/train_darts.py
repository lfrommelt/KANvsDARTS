#from utils import Monitor
import numpy as np
import wandb
import torch
import utils
import sys
import os
import time
import contextlib

sys.path.append(os.path.normpath(os.getcwd() + "/autora-theorist-darts/src"))
torch.set_default_dtype(torch.float64)

from autora.theorist.darts import DARTSExecutionMonitor, DARTSRegressor
from autora.theorist.darts.utils import NanError

from darts import DARTS
#from layered_darts import LayeredDARTS
import sympy
import autora

# sounds small, but for 1-3d data its pretty dense
n_samples=200#200

import torch, contextlib

class check_non_finite(contextlib.AbstractContextManager):
    r"""
    with check_non_finite(model):
        loss = model(x).sum()
        loss.backward()
    """

    def __init__(self, model, *, forward=True, backward=True, param_grad=True):
        self.m      = model
        self.do_fwd = forward
        self.do_bwd = backward
        self.do_par = param_grad
        self.handles = []

    # ------------------------------------------------------------------ #
    #  Forward hook (per module)                                          #
    # ------------------------------------------------------------------ #
    def _make_fwd_hook(self, name):
        def hook(module, inputs, output):

            def _inspect(t):
                if not (isinstance(t, torch.Tensor) and not torch.isfinite(t).all()):
                    return
                mask_nan  = torch.isnan(t)
                mask_posi = torch.isinf(t) & (t > 0)
                mask_negi = torch.isinf(t) & (t < 0)

                print(f"\n>>> NON-FINITE in FORWARD of  {name}")
                print("    module class :", module.__class__.__name__)
                print("    shape        :", tuple(t.shape))
                if mask_nan.any():  print("    # NaN   :", mask_nan.sum().item())
                if mask_posi.any(): print("    # +Inf  :", mask_posi.sum().item())
                if mask_negi.any(): print("    # -Inf  :", mask_negi.sum().item())
                print("    first bad idx:",
                      (mask_nan | mask_posi | mask_negi).nonzero(as_tuple=False)[:5])
                if utils.Dump.latent_states:
                    print(*utils.Dump.latent_states, sep="\n")
                    utils.Dump.latent_states=[]
            # ------------------------------------------------------------
            if isinstance(output, torch.Tensor):
                _inspect(output)
            elif isinstance(output, (tuple, list)):
                for o in output: _inspect(o)
        return hook

    # ------------------------------------------------------------------ #
    #  Back-ward hook (per module)                                        #
    # ------------------------------------------------------------------ #
    def _make_bwd_hook(self, name):
        def hook(module, grad_in, grad_out):
            for grad in grad_out:
                if grad is None or torch.isfinite(grad).all():
                    continue
                mask_nan  = torch.isnan(grad)
                mask_posi = torch.isinf(grad) & (grad > 0)
                mask_negi = torch.isinf(grad) & (grad < 0)

                print(f"\n>>> NON-FINITE GRADIENT in BACKWARD of {name}")
                print("    module class :", module.__class__.__name__)
                print("    shape        :", tuple(grad.shape))
                if mask_nan.any():  print("    # NaN   :", mask_nan.sum().item())
                if mask_posi.any(): print("    # +Inf  :", mask_posi.sum().item())
                if mask_negi.any(): print("    # -Inf  :", mask_negi.sum().item())
                print("    first bad idx:",
                      (mask_nan | mask_posi | mask_negi).nonzero(as_tuple=False)[:5])
                if grad.grad_fn is not None:
                    print("    produced by  :", grad.grad_fn)
                if utils.Dump.latent_states:
                    print(*utils.Dump.latent_states, sep="\n")
                    utils.Dump.latent_states=[]
        return hook

    # ------------------------------------------------------------------ #
    #  Per-parameter gradient hook                                        #
    # ------------------------------------------------------------------ #
    def _make_param_hook(self, pname):
        def hook(grad):
            if torch.isfinite(grad).all():
                return
            mask_nan  = torch.isnan(grad)
            mask_posi = torch.isinf(grad) & (grad > 0)
            mask_negi = torch.isinf(grad) & (grad < 0)

            print(f"\n>>> NON-FINITE PARAMETER-GRAD '{pname}'")
            print("    shape        :", tuple(grad.shape))
            if mask_nan.any():  print("    # NaN   :", mask_nan.sum().item())
            if mask_posi.any(): print("    # +Inf  :", mask_posi.sum().item())
            if mask_negi.any(): print("    # -Inf  :", mask_negi.sum().item())
            print("    first bad idx:",
                  (mask_nan | mask_posi | mask_negi).nonzero(as_tuple=False)[:5])

            if grad.grad_fn is not None:           # seldom available here
                print("    produced by  :", grad.grad_fn)
            if utils.Dump.latent_states:
                    print(*utils.Dump.latent_states, sep="\n")
                    utils.Dump.latent_states=[]
        return hook

    # ------------------------------------------------------------------ #
    #  Context-manager interface                                          #
    # ------------------------------------------------------------------ #
    def __enter__(self):
        # module hooks ---------------------------------------------------
        for name, mod in self.m.named_modules():
            if self.do_fwd:
                h = mod.register_forward_hook(self._make_fwd_hook(name))
                self.handles.append(h)
            if self.do_bwd:
                try:
                    h = mod.register_full_backward_hook(
                            self._make_bwd_hook(name))
                except AttributeError:             # ≤ 1.7 fallback
                    h = mod.register_backward_hook(
                            self._make_bwd_hook(name))
                self.handles.append(h)

        # parameter hooks ------------------------------------------------
        if self.do_par:
            for pname, p in self.m.named_parameters():
                if p.requires_grad:
                    h = p.register_hook(self._make_param_hook(pname))
                    self.handles.append(h)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False
    
    
class Bla:
    def __init__(self, trainloss, valloss, testloss):
        self.test_loss=testloss
        self.val_loss=valloss
        self.train_loss=trainloss
        
"""def train2(run,
          config, 
          dataset,
          equation,
          training_seed,
          n_nodes=3, ratio=(1,1,0.5), id_="noname"):
    #print(id_)
    
    seed = int(time.time() * 1e6)%3000
    np.random.seed(seed)
    test= np.random.random(100)
    train= np.random.random(100)
    val= np.random.random(80)
    '''print("logs:",test)
    for i, x in enumerate(test):
        wandb.log({f"test_{id_}": x}, step=i)'''
    return Bla(train,val,test), None, "bla"""
    
    
def train(config, 
          dataset,
          training_seed, ratio=(1,1,0.5), scale_steps=1, training_rng=None):

    
    #id_+=f"_{training_seed}"
    #monitor = Monitor()#DARTSExecutionMonitor()

    epochs = int(config["steps"]*scale_steps)

    # seed for training and model init
    np.random.seed(training_seed)
    torch.random.manual_seed(training_seed)
    
    # just for initialization bc there is some well hidden stuff going on before
    rng = torch.Generator()
    rng.manual_seed(training_seed)
    
    print("seed", training_seed)
    print((torch.mean(torch.get_rng_state()[:10].to(torch.float)), torch.get_rng_state()[0]))
    print(ratio)
    #model = DARTSRegressor([2,2,1], primitives,temp=config["temp"],)
    try:
        model = DARTSRegressor(
            batch_size=config["batch_size"],
            num_graph_nodes=config["size"],
            fair_darts_loss_weight=0.0,
            max_epochs=epochs,
            primitives=config["primitives"],
            param_updates_for_sampled_model=config["finetune_epochs"],
            param_learning_rate_max=config["param_learning_rate_max"],
            param_learning_rate_min=config["param_learning_rate_min"],
            arch_learning_rate_max=config["arch_learning_rate_max"],
            darts_type = "original",
            param_momentum= config["param_momentum"],
            grad_clip = 5, #should sweep? Seems reasonable....
            
            train_classifier_coefficients = config["train_output_layer"],
            train_classifier_bias = config["train_output_layer"],
            
            param_weight_decay=0.0,

            arch_weight_decay=0.0,
            arch_weight_decay_df=0.0,
            arch_weight_decay_base=0.0,
        )
        print(model.primitives)
        #return model
        #print(config["coeff_discretization"])
        result = model.fit(dataset["train_input"], 
                          dataset["train_label"],
                          ratio=ratio,

                          #batch_size=config["batch_size"], 
                          #ratio=config["ratio"], 
                          #monitor=monitor, 
                          #n_epochs="2",
                          #finetune_epochs=config["finetune_epochs"],
                          arch_discretization=config["arch_discretization"],
                          coeff_discretization=config["coeff_discretization"],
                           rng=rng,
                         )

        outcome="normal"

    except NanError as e:
        print("nan error:",model.execution_monitor.train_loss,sep="\n")
        raise e
        #pass
    if any(np.isnan(loss) for loss in model.execution_monitor.test_loss):
        print("nan catched:")
        #print(model.execution_monitor.train_loss)
        #print(autora.theorist.darts.utils.buffer)
        primitives=list(config["primitives"])
        #if config["safety"]=="safe":#, "smooth", "ramped"]
        for i, primitive in enumerate(primitives):
            if primitive in ("power_two", "power_three", "exp", "reciprocal"):
                primitives[i]="safe_"+primitives[i]
            if config["safety"]=="smooth":
                if primitive in ("power_two", "power_three", "exp"):
                    primitives[i]=primitives[i]+"_smooth"
            elif config["safety"]=="ramped":
                if primitive in ("power_two", "power_three"):
                    primitives[i]=primitives[i]+"_ramped"
                elif primitive in ("exp"):
                    primitives[i]=primitives[i]+"_ramped"

        #print(primitives)

        model = DARTSRegressor(
            batch_size=config["batch_size"],
            num_graph_nodes=config["size"],
            fair_darts_loss_weight=0.0,
            max_epochs=epochs,
            primitives=primitives,
            param_updates_for_sampled_model=config["finetune_epochs"],
            param_learning_rate_max=config["param_learning_rate_max"],
            param_learning_rate_min=config["param_learning_rate_max"]*0.1,
            arch_learning_rate_max=config["arch_learning_rate_max"],
            darts_type = "original",
            param_momentum= config["param_momentum"],
            grad_clip = 5, #should sweep? Seems reasonable....
            
            train_classifier_coefficients = config["train_output_layer"],
            train_classifier_bias = config["train_output_layer"],
            #execution_monitor = Monitor(),
        )
    
        #return model
        #print(config["coeff_discretization"])
        result = model.fit(dataset["train_input"], 
                          dataset["train_label"],
                          ratio=ratio,
                          
                          #batch_size=config["batch_size"], 
                          #ratio=config["ratio"], 
                          #monitor=monitor, 
                          #n_epochs="2",
                          #finetune_epochs=config["finetune_epochs"],
                          arch_discretization=config["arch_discretization"],
                          coeff_discretization=config["coeff_discretization"],
                           rng=rng,
                         )

        
        if any(np.isnan(loss) for loss in model.execution_monitor.test_loss):
            outcome = "failed"
            return model.execution_monitor, model, model.to_sympy(), outcome
        else:
            outcome="safe"
        #print(model.execution_monitor.train_loss[-1])

    #print(model.execution_monitor.test_loss)
    #print(len(model.execution_monitor.test_loss))
    monitor = model.execution_monitor
    train_log = {
        "train_loss": np.array([]),
        "test_loss": np.array([]),
        "alphas": np.array([]),
    }  # stupid
    train_log["train_loss"] = monitor.train_loss
    train_log["test_loss"] = monitor.test_loss
    train_log["val_loss"] = monitor.val_loss
    #train_log["alphas"] = monitor.arch_weight_history

    #config["train_loss"] = train_log["train_loss"]
    #config["test_loss"] = train_log["test_loss"]
    #config["alphas"] = train_log["alphas"]


    predicted_equation = model.to_sympy()

    # rounded = utils.simplify(predicted_equation)

    #config["predicted_equation"] = predicted_equation

    #utils.save_experiment_config(config, folder="configs_layered_darts")
    #monitor.predicted_equation = str(predicted_equation)
    '''
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
            raise'''

                
    return monitor, model, predicted_equation, outcome




def train_new_darts(config, 
          dataset,
          training_seed, scale_steps=1, training_rng=None, disable_tqdm=False, debug=False, coeff_opti="sgd", failfast=False,
                   reset_adam=False):

    
    #id_+=f"_{training_seed}"
    #monitor = Monitor()#DARTSExecutionMonitor()

    epochs = int(config["batch_size"]*42*scale_steps)
    print(f'epochs = int({config["batch_size"]}*42*{scale_steps})')
    # seed for training and model init
    np.random.seed(training_seed)
    torch.random.manual_seed(training_seed)
    
    # just for initialization bc there is some well hidden stuff going on before
    rng = torch.Generator()
    rng.manual_seed(training_seed)
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)


    if not failfast:
        model=DARTS(
            primitives=config["primitives"],
            size=config["size"],
            n_vars=len(dataset["train_input"][0]),
            n_outputs=1,
            train_output_layer=config["train_output_layer"],
            init_range=config["init_range"],
        )

        with (check_non_finite(model) if debug==True else contextlib.nullcontext()):
            model, monitor=model.fit(
                dataset["train_input"],
                dataset["train_label"],
                batch_size=config["batch_size"],
                ratio=config["ratio"],
                n_epochs=epochs,
                param_learning_rate_max=config["param_learning_rate_max"],
                param_learning_rate_min=config["param_learning_rate_min"],
                arch_learning_rate_max=config["arch_learning_rate_max"],

                param_weight_decay=config["param_weight_decay"],
                arch_weight_decay=config["arch_weight_decay"],

                param_momentum = config["param_momentum"],

                coeff_discretization=config["coeff_discretization"],#"max",
                arch_discretization=config["arch_discretization"],#"softmax",
                disable_tqdm=disable_tqdm,

                finetune_epochs=config["finetune_epochs"],
                coeff_opti=coeff_opti,
                loss=config["loss_fn"],
                reset_adam=reset_adam,
                )

        outcome="normal"
    
    else:
        class FailMonitor:
            def __init__(self):
                self.test_loss=[np.nan]
        monitor=FailMonitor()


    if any(np.isnan(loss) for loss in monitor.test_loss):
        print("nan catched:")
        # just for initialization bc there is some well hidden stuff going on before
        rng = torch.Generator()
        rng.manual_seed(training_seed)
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        
        primitives=list(config["primitives"])
        #if config["safety"]=="safe":#, "smooth", "ramped"]
        for i, primitive in enumerate(primitives):
            if any([template in primitive for template in ("power_two", "power_three", "exp", "reciprocal")]):
                primitives[i]="safe_"+primitives[i]
            if config["safety"]=="smooth":
                if primitive in ("power_two", "power_three", "exp"):
                    primitives[i]=primitives[i]+"_smooth"
            elif config["safety"]=="ramped":
                if primitive in ("power_two", "power_three"):
                    primitives[i]=primitives[i]+"_ramped"
                elif primitive in ("exp"):
                    primitives[i]=primitives[i]+"_ramped"

        print(primitives)

        model=DARTS(
            primitives=primitives,
            size=config["size"],
            n_vars=len(dataset["train_input"][0]),
            n_outputs=1,
            train_output_layer=config["train_output_layer"],
            init_range=config["init_range"],
        )
    
        
        with (check_non_finite(model) if debug==True else contextlib.nullcontext()):
            model, monitor=model.fit(
                dataset["train_input"],
                dataset["train_label"],
                batch_size=config["batch_size"],
                ratio=config["ratio"],
                n_epochs=epochs,
                param_learning_rate_max=config["param_learning_rate_max"],
                param_learning_rate_min=config["param_learning_rate_min"],
                arch_learning_rate_max=config["arch_learning_rate_max"],

                param_weight_decay=config["param_weight_decay"],
                arch_weight_decay=config["param_weight_decay"],

                param_momentum = config["param_momentum"],

                coeff_discretization=config["coeff_discretization"],
                arch_discretization=config["arch_discretization"],#"softmax",
                disable_tqdm=disable_tqdm,
                finetune_epochs=config["finetune_epochs"],
                coeff_opti=coeff_opti,
                loss=config["loss_fn"],
                reset_adam=reset_adam,
            )

        
        if any(np.isnan(loss) for loss in monitor.test_loss):
            outcome = "failed"
            return monitor, model, model.to_sympy(variable_prefix="x_"), outcome
        else:
            outcome="safe"

    predicted_equation = model.to_sympy(variable_prefix="x_")

    return monitor, model, predicted_equation, outcome

def train_new_layered(config, 
          dataset,
          training_seed, scale_steps=1, training_rng=None, disable_tqdm=False):

    
    #id_+=f"_{training_seed}"
    #monitor = Monitor()#DARTSExecutionMonitor()

    epochs = int(config["batch_size"]*42*scale_steps)

    # seed for training and model init
    np.random.seed(training_seed)
    torch.random.manual_seed(training_seed)
    
    # just for initialization bc there is some well hidden stuff going on before
    rng = torch.Generator()
    rng.manual_seed(training_seed)
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)


    model=LayeredDARTS(
        primitives=config["primitives"],
        size=config["size"],
        n_vars=len(dataset["train_input"][0]),
        n_outputs=1,
        train_output_layer=config["train_output_layer"],
        init_range=config["init_range"],
    )

    model, monitor=model.fit(
        dataset["train_input"],
        dataset["train_label"],
        batch_size=config["batch_size"],
        ratio=config["ratio"],
        n_epochs=epochs,
        param_learning_rate_max=config["param_learning_rate_max"],
        param_learning_rate_min=config["param_learning_rate_min"],
        arch_learning_rate_max=config["arch_learning_rate_max"],

        param_weight_decay=config["param_weight_decay"],
        arch_weight_decay=config["arch_weight_decay"],

        param_momentum = config["param_momentum"],

        coeff_discretization="max",
        arch_discretization="softmax",
        disable_tqdm=disable_tqdm,
        
        finetune_epochs=config["finetune_epochs"],
        loss=config["loss_fn"],
        )

    outcome="normal"

    if any(np.isnan(loss) for loss in monitor.test_loss):
        print("nan catched:")

        primitives=list(config["primitives"])
        #if config["safety"]=="safe":#, "smooth", "ramped"]
        for i, primitive in enumerate(primitives):
            if primitive in ("power_two", "power_three", "exp", "reciprocal"):
                primitives[i]="safe_"+primitives[i]
            if config["safety"]=="smooth":
                if primitive in ("power_two", "power_three", "exp"):
                    primitives[i]=primitives[i]+"_smooth"
            elif config["safety"]=="ramped":
                if primitive in ("power_two", "power_three"):
                    primitives[i]=primitives[i]+"_ramped"
                elif primitive in ("exp"):
                    primitives[i]=primitives[i]+"_ramped"

        #print(primitives)

        model=DARTS(
            primitives=primitives,
            size=config["size"],
            n_vars=len(dataset["train_input"][0]),
            n_outputs=1,
            train_output_layer=config["train_output_layer"],
            init_range=config["init_range"],
        )
    
        model, monitor=model.fit(
            dataset["train_input"],
            dataset["train_label"],
            batch_size=config["batch_size"],
            ratio=config["ratio"],
            n_epochs=epochs,
            param_learning_rate_max=config["param_learning_rate_max"],
            param_learning_rate_min=config["param_learning_rate_min"],
            arch_learning_rate_max=config["arch_learning_rate_max"],

            param_weight_decay=0.0,
            arch_weight_decay=config["param_weight_decay"],

            param_momentum = config["param_momentum"],

            coeff_discretization="max",
            arch_discretization="softmax",
            disable_tqdm=disable_tqdm,
            finetune_epochs=config["finetune_epochs"],
            loss=config["loss_fn"],
        )

        
        if any(np.isnan(loss) for loss in monitor.test_loss):
            outcome = "failed"
            return monitor, model, model.to_sympy(variable_prefix="x_"), outcome
        else:
            outcome="safe"

    predicted_equation = model.to_sympy(variable_prefix="x_")

    return monitor, model, predicted_equation, outcome