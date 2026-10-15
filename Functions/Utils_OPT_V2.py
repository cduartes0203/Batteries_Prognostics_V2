import os
import cmaes
import scipy
import numpy as np
import ipynbname
import optuna
import optunahub
from optuna.exceptions import TrialPruned
from collections import deque
from IPython.display import clear_output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EarlyStoppingCallback:

    def __init__(self, patience: int, count: int=1, n_last: int = 3):
        self.patience = patience

        self.count = count
        self.best_score = float('inf')
        self.trials_without_improvement = 0
        self.last_trials = deque(maxlen=n_last)
        self.best_trial_info = None


    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ):
        is_multi = study._is_multi_objective()

        # 1. Formata a linha do trial atual
        if trial.state == optuna.trial.TrialState.COMPLETE:
            score_str = (
                f"{[round(v, 4) for v in trial.values]}"
                if is_multi
                else f"{trial.value:.4f}"
            )
            msg = f"Trial #{trial.number:03d} [COMPLETE] | Score: {score_str} | Params: {trial.params}"
        else:
            msg = f"Trial #{trial.number:03d} [{trial.state.name}]"

        self.last_trials.append(msg)

        # 2. Avaliação de melhor score e Early Stopping
        if trial.state == optuna.trial.TrialState.COMPLETE:
            current_score = (
                sum(trial.values)
                if (is_multi and trial.values is not None)
                else trial.value
            )

            if current_score < self.best_score:
                self.best_score = current_score
                self.trials_without_improvement = 0

                # Armazena os dados do melhor trial até o momento
                if is_multi:
                    vals_str = [round(v, 4) for v in trial.values]
                    self.best_trial_info = f"Trial #{trial.number:03d} | Objetivos: {vals_str} (Soma: {current_score:.4f}) | Params: {trial.params}"
                else:
                    self.best_trial_info = f"Trial #{trial.number:03d} | Valor: {trial.value:.4f} | Params: {trial.params}"
            else:
                self.trials_without_improvement += 1

        # 3. Renderização Dinâmica no Notebook
        clear_output(wait=True)
        print("=" * 80)
        print(f"BEST TRIAL - Study {self.count}")
        print("=" * 80)
        if self.best_trial_info:
            print(self.best_trial_info)
            if is_multi and len(study.best_trials) > 0:
                print(
                    f"-> TOTAL NUMBER OF SOLUTIONS IN THE PARETO FRONT: {len(study.best_trials)}"
                )
        else:
            print("NO COMPLETED TRIALS FINISHED YET.")

        print("\n" + "=" * 80)
        print(f"LAST {self.last_trials.maxlen} TRIALS")
        print("=" * 80)
        for log in self.last_trials:
            print(log)

        print("-" * 80)
        print(
            f"STOPPING STATUS: {self.trials_without_improvement}/{self.patience} ITERATIONS WITHOUT IMPROVEMENT."
        )
        print("-" * 80)

        # 4. Condição de parada
        if self.trials_without_improvement >= self.patience:
            print(
                f"\n>>> STUDY INTERRUPTED! NO IMPROVEMENTS FOR {self.patience} ITERATIONS."
            )
            study.stop()

def SelSampler(mode=None,n_startup_trials=1000, seed=None):
    '''
    modes available: auto, GP, NSGAII, random, tpe

    None: sampler default, suport multivariate optimization
    GP: suport multivariate optimization
    Auto: suport multivariate optimization
    NSGAII: poorly suport multivariate optimization
    Random: poorly suport multivariate optimization
    TPE: suport multivariate optimization

    '''
    module = optunahub.load_module(package="samplers/auto_sampler")
    if mode is None or mode.lower() == 'none':
        sampler = None
    elif mode.lower() == 'gp':
        sampler = optuna.samplers.GPSampler(seed=seed)
    elif mode.lower() == 'auto':
        sampler = module.AutoSampler(seed=seed)    
    elif mode.lower() == 'nsgaii':
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    elif mode.lower() == 'tpe':
        sampler = optuna.samplers.TPESampler(multivariate=True,group=True,n_startup_trials=n_startup_trials,seed=seed)
    elif mode.lower() == 'random':
        sampler=optuna.samplers.RandomSampler(seed=seed)
    return sampler

def df_ParamsTable(names,FileName):
    if len(names) == 13 or len(names) == 11:
        study_dir = f'Optimization/{FileName}/multi/'
        out_path = f'Optimization/{FileName}/multi/Optimization.csv'
        study_dir = os.path.normpath(study_dir)
        out_path = os.path.normpath(out_path)
        os.makedirs(study_dir, exist_ok=True)
    elif len(names) == 12 or len(names) == 10:
        study_dir = f'Optimization/{FileName}/mono/'
        out_path = f'Optimization/{FileName}/mono/Optimization.csv'
        study_dir = os.path.normpath(study_dir)
        out_path = os.path.normpath(out_path)
        os.makedirs(study_dir, exist_ok=True)

    return study_dir, out_path