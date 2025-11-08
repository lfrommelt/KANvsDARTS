
from itertools import chain
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pykan.kan.MultKAN import KAN
import numpy as np


def safe_fit(kan_model, dataset, **kwargs):
    outcome="normal"
    checkpoint=f"{kan_model.round}.{kan_model.state_id}"
    log=kan_model.fit(dataset, singularity_avoiding=False, **kwargs)
    if np.isnan(log['train_loss'][-1]):
        outcome="safe"
        print("nan detected, switchig to safe mode")
        kan_model=kan_model.rewind(checkpoint)
        print(f"{kan_model.round}.{kan_model.state_id}")
        log=kan_model.fit(dataset, singularity_avoiding=True, **kwargs)
    return log, outcome



def train_kan(config, dataset, tolerance=0.9):
    outcome="default"
    logs=[]
    model=KAN(width=config["width"], grid=config["grid"], k=config["k"], 
              mult_arity = 0, noise_scale=0.3, scale_base_mu=0.0, 
              scale_base_sigma=1.0, base_fun='zero', symbolic_enabled=True, 
              affine_trainable=False,
              grid_eps=0.02, grid_range=[0, 5.0], sp_trainable=False,
              sb_trainable=False, seed=config["seed"], save_act=True,
              sparse_init=False, auto_save=True, first_init=True,
              ckpt_path='./model', state_id=0, round=0,
              device='cpu', real_affine_trainable=0.
              )
    
    eps=1e-10
    ## Structure Phase
    lt=np.inf
    lt_1=np.inf
    steps=0
    while not (lt_1*tolerance>lt):
        steps+=1
        lt=lt_1
        singularity_avoiding=False

        checkpoint=f"{model.round}.{model.state_id}"
        print("current checkpoint:", checkpoint)
        logs.append(model.fit(dataset, opt="LBFGS", steps=config["structure_steps"], 
                              log=1, lamb=0., lamb_l1=config["lamb_l1"], lamb_entropy=config["lamb_entropy"], 
                              lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, 
                              loss_fn=config["loss_fn"], lr=config["lr"], start_grid_update_step=-1,
                              stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False, 
                              in_vars=None, out_vars=None, beta=3, save_fig_freq=1,
                              singularity_avoiding=singularity_avoiding, y_th=1000., 
                              reg_metric='edge_forward_spline_n', display_metrics=None, 
                              verbose=False, print_gradients=False,
                              )
                    )
        
        # forward pass for pruning values
        model(dataset["train_input"])
        check = model.prune_minimally(dataset["train_input"], v=False, local=False, per_neuron=False, semi_minimal=False)
        if check is None:
            break
        #model.plot()
        #plt.show()

        lt_1=logs[-1]["test_loss"][-1]
    outcome = [steps]

    model = model.rewind(checkpoint)

    steps=0
    ## Symbolification Phase
    outcome.append("normal")
    while sum([act_fun.mask.sum() for act_fun in model.act_fun]):
        steps+=1
        log, outcome_ = safe_fit(model, dataset, opt="LBFGS", steps=config["symbolification_steps"], 
                              log=1, lamb=0., lamb_l1=0.0, lamb_entropy=0.0, 
                              lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, 
                              loss_fn=config["loss_fn"], lr=config["lr"], start_grid_update_step=-1,
                              stop_grid_update_step=50, batch=-1, y_th=1000., 
                              reg_metric='edge_forward_spline_n', display_metrics=None, 
                              verbose=False, print_gradients=False,
                              )
        if outcome_=="safe":
            outcome[1]="safe"
        logs.append(log)
        model(dataset["train_input"])
        model.minimal_auto_symbolic(verbose=False, lib=config["lib"], n_1d_fits=config["n_1d_fits"])
        #print(*[act_fun.mask for act_fun in model.act_fun], sep="\n")


    outcome.append(steps)
    ## Finetune

    log, outcome = safe_fit(model, dataset, opt="LBFGS", steps=config["finetune_steps"], 
                            log=1, lamb=0., lamb_l1=0.0, lamb_entropy=0.0, 
                            lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, 
                            loss_fn=config["loss_fn"], lr=config["lr"], start_grid_update_step=-1,
                            stop_grid_update_step=50, batch=-1, y_th=1000., 
                            reg_metric='edge_forward_spline_n', display_metrics=None, 
                            verbose=False, print_gradients=False,
                            )
    logs.append(log)
    

    '''logs.append(model.fit(dataset, opt="LBFGS", steps=config["finetune_steps"], 
                            log=1, lamb=0., lamb_l1=0.0, lamb_entropy=0.0, 
                            lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, 
                            loss_fn=config["loss_fn"], lr=config["lr"], start_grid_update_step=-1,
                            stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False, 
                            in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
                            singularity_avoiding=singularity_avoiding, y_th=1000., 
                            reg_metric='edge_forward_spline_n', display_metrics=None, 
                            verbose=False, print_gradients=False,
                            )
                )'''
    
    
    #print("mask",*[layer.mask for layer in model.act_fun], sep="\n")
    #print("symbolic_mask",*[layer.mask for layer in model.symbolic_fun], sep="\n")

    # sympy tends to get stuck in some sort of endless recursion sometimes
    predicted_equation = model.symbolic_formula(simplify=False)[0][0]#_light()

    log = {
        key: list(chain.from_iterable(d[key] for d in logs))
        for key in logs[0]                        # assume at least one dict
    }

    monitor = SimpleNamespace(**log)
    return monitor, model, predicted_equation, outcome
