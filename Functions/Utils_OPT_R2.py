import os
import time
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


def FormatTime(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class EarlyStoppingCallback:
    '''
    context (opcional): informações da execução exibidas acima de BEST TRIAL.
        nL         : faixa de camadas [min, max] (ou int)
        session    : número da sessão atual;  n_study: total de sessões
        study_name : nome do estudo no banco
        config     : pasta/configuração dos resultados
        n_prev     : trials armazenados antes desta execução
        n_trials   : trials previstos nesta execução;  timeout: limite em segundos
        storage    : banco do Optuna (RDBStorage ou URL)  } se informados, BEST TRIAL mostra o melhor trial
        prefix     : prefixo dos estudos (ex.: 'obj1_')    } de cada nL (todas as sessões/fases armazenadas)
    '''

    def __init__(self, patience: int, n_last: int = 3, context: dict = None):
        self.patience = patience
        #self.ref = ref
        #self.count = count
        self.best_score = float('inf')
        self.trials_without_improvement = 0
        self.last_trials = deque(maxlen=n_last)
        self.best_trial_info = None
        self.context = context
        self.t0 = time.time()
        self.n_run = 0
        self.best_number = None
        # Melhor trial por nº de camadas: lido do banco uma vez no início da sessão (os demais nL não mudam
        # durante a sessão) e atualizado ao vivo com os trials da sessão atual
        self.nl_best = {}
        self.per_nl = bool(context) and context.get('storage') is not None and bool(context.get('prefix'))
        if self.per_nl:
            self.LoadBestPerNL()

    def UpdateBestPerNL(self, study_name, t):
        nL = t.params.get('n_layers')
        if nL is None or t.values is None: return
        score = sum(t.values) if len(t.values) > 1 else t.values[0]
        if nL not in self.nl_best or score < self.nl_best[nL][0]:
            self.nl_best[nL] = (score, study_name, t.number, t.params)

    def LoadBestPerNL(self):
        storage, prefix = self.context['storage'], self.context['prefix']
        for name in optuna.get_all_study_names(storage):
            if not name.startswith(prefix): continue
            for t in optuna.load_study(study_name=name, storage=storage).trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    self.UpdateBestPerNL(name, t)

    def PrintBestPerNL(self, is_multi):
        nLc = self.context.get('nL')
        running = set(range(nLc[0], nLc[1] + 1)) if isinstance(nLc, (list, tuple)) else {nLc}
        label = 'Score (sum)' if is_multi else 'Score'
        print("=" * 80)
        print("BEST TRIAL PER nL")
        print("=" * 80)
        for nL in sorted(set(self.nl_best) | running):
            mark = '   <-- running' if nL in running else ''
            if nL in self.nl_best:
                score, name, number, params = self.nl_best[nL]
                print(f"nL={nL} | {name} #{number:04d} | {label}: {score:.4f} | Params: {params}{mark}")
            else:
                print(f"nL={nL} | no completed trials yet{mark}")
        if self.best_score < float('inf'):
            number = f"#{self.best_number:04d}" if self.best_number is not None else "#----"
            print(f"Current session best: {number} | {label}: {self.best_score:.4f}")
        else:
            print("Current session best: no completed trials yet")

    def PrintContext(self, study):
        c = self.context
        nL = c.get('nL')
        if isinstance(nL, (list, tuple)):
            nL = f"{nL[0]}" if nL[0] == nL[1] else f"{nL[0]}-{nL[1]}"
        n_trials = c.get('n_trials')
        timeout = c.get('timeout')
        print("=" * 80)
        print("OPTIMIZATION PROGRESS")
        print("=" * 80)
        print(f"Layers (nL): {nL}  |  Session: {c.get('session')}/{c.get('n_study')} ({c.get('study_name')})  |  "
              f"Sampler: {type(study.sampler).__name__.replace('Sampler', '')}  |  Config: {c.get('config')}")
        print(f"Trials in this study: {c.get('n_prev', 0)} stored before this run  |  "
              f"This run: {self.n_run}/{int(n_trials) if n_trials is not None else '-'}  |  "
              f"Elapsed: {FormatTime(time.time() - self.t0)} / {FormatTime(timeout) if timeout is not None else '-'}")
        print()


    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ):
        is_multi = study._is_multi_objective()
        self.n_run += 1

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

            if self.per_nl:
                self.UpdateBestPerNL(study.study_name, trial)

            if current_score < self.best_score:
                self.best_score = current_score
                self.best_number = trial.number
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
        if self.context:
            self.PrintContext(study)
        if self.per_nl:
            self.PrintBestPerNL(is_multi)
            if is_multi and len(study.best_trials) > 0:
                print(f"-> TOTAL NUMBER OF SOLUTIONS IN THE PARETO FRONT: {len(study.best_trials)}")
        else:
            print("=" * 80)
            print("BEST TRIAL")
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
            f"STOPPING STATUS: {self.trials_without_improvement}/{int(self.patience)} ITERATIONS WITHOUT IMPROVEMENT."
        )
        print("-" * 80)

        # 4. Condição de parada
        if self.trials_without_improvement >= self.patience:
            print(
                f"\n>>> STUDY INTERRUPTED! NO IMPROVEMENTS FOR {int(self.patience)} ITERATIONS."
            )
            study.stop()

def SelSampler(mode=None,n_startup_trials=100, seed=None):
    '''
    modes available: auto, GP, NSGAII, random, tpe

    None: sampler default, suport multivariate optimization
    GP: suport multivariate optimization
    Auto: suport multivariate optimization
    NSGAII: poorly suport multivariate optimization
    Random: poorly suport multivariate optimization
    TPE: suport multivariate optimization

    n_startup_trials: random trials before TPE starts modelling (default 100, ~10x the number of parameters)
    '''
    if mode is None or mode.lower() == 'none':
        sampler = None
    elif mode.lower() == 'gp':
        # Objetivo determinístico (sementes fixas no eRNN): o GP não modela ruído
        sampler = optuna.samplers.GPSampler(seed=seed, deterministic_objective=True)
    elif mode.lower() == 'auto':
        # optunahub só é carregado (download) quando o AutoSampler é usado
        module = optunahub.load_module(package="samplers/auto_sampler")
        sampler = module.AutoSampler(seed=seed)
    elif mode.lower() == 'nsgaii':
        sampler=optuna.samplers.NSGAIISampler(seed=seed)
    elif mode.lower() == 'tpe':
        sampler = optuna.samplers.TPESampler(multivariate=True,group=True,n_startup_trials=n_startup_trials,seed=seed)
    elif mode.lower() == 'random':
        sampler=optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"OptSampler inválido: {mode}. Use None, 'auto', 'gp', 'nsgaii', 'tpe' ou 'random'.")
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