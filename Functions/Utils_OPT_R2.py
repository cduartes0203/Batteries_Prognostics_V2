import os
import io
import sys
import json
import time
import shutil
import contextlib
import cmaes
import scipy
import numpy as np
import pandas as pd
import ipynbname
import optuna
import optunahub
from optuna.exceptions import TrialPruned
from collections import deque
from IPython.display import clear_output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ tela do console
REFRESH_DT = 5.0           # intervalo mínimo entre atualizações da tela da otimização (s)
_VT = None                 # console aceita sequências ANSI (Windows 10+: ativadas em _EnableVT)


def _InNotebook():
    try:
        from IPython.core.getipython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def _EnableVT():
    '''Ativa as sequências ANSI no console do Windows (posicionar o cursor e apagar linhas). False se não houver
    console (saída redirecionada) ou se o console não aceitar.'''
    global _VT
    if _VT is not None: return _VT
    _VT = False
    try:
        if not sys.stdout.isatty(): return _VT
        if os.name == 'nt':
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)                                        # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not k32.GetConsoleMode(h, ctypes.byref(mode)): return _VT
            if not k32.SetConsoleMode(h, mode.value | 0x0004): return _VT   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        _VT = True
    except Exception:
        _VT = False
    return _VT


def DrawScreen(text, first=False):
    '''
    Exibe a tela inteira de uma vez, sobrescrevendo a anterior sem apagar a janela antes (sem piscar): cursor no
    topo, cada linha cortada na largura do console (sem quebra de linha) e apagada até o fim, e o resto da janela
    apagado. Notebook: clear_output. Sem console (saída redirecionada): apenas imprime.
    '''
    if _InNotebook():
        clear_output(wait=True)
        print(text, flush=True)
        return
    if not _EnableVT():
        print(text, flush=True)
        return
    width = max(40, shutil.get_terminal_size((120, 40)).columns - 1)
    lines = [ln[:width] + '\x1b[K' for ln in text.split('\n')]
    head = '\x1b[2J\x1b[3J\x1b[H' if first else '\x1b[H'      # 1ª tela: limpa também o histórico da janela
    sys.stdout.write(head + '\n'.join(lines) + '\n\x1b[J')
    sys.stdout.flush()


def FormatParams(params):
    '''Parâmetros de um trial em uma linha curta: n_layers e nR_layer_i viram nR=[...] (após nI); demais nome=valor.'''
    p = dict(params)
    nL = p.pop('n_layers', None)
    nR = [p.pop(f'nR_layer_{l}') for l in range(int(nL)) if f'nR_layer_{l}' in p] if nL is not None else []
    out = []
    for k, v in p.items():
        out.append(f'{k}={FormatNum(v)}')
        if k == 'nI' and nR: out.append(f'nR={FormatNum(nR)}')
    if nR and not any(s.startswith('nR=') for s in out): out.append(f'nR={FormatNum(nR)}')
    return ' '.join(out)


def MeanWapeRUL(trial):
    '''Média dos WAPE_RUL_<rolamento> finais gravados no trial (user_attrs de OptimizeSeq); None se não houver.'''
    v = [float(x) for k, x in trial.user_attrs.items() if k.startswith('WAPE_RUL_') and x is not None]
    return float(np.mean(v)) if v else None


def ParetoFront(study):
    '''
    Trials da frente de Pareto (o mesmo conjunto de study.best_trials, para estudos multiobjetivo sem restrições).
    study.best_trials (Optuna 5.0) faz deepcopy de TODOS os trials do estudo, inclusive os podados: com milhares de
    trials leva ~1 s por chamada e cresce com o estudo. Aqui só os trials concluídos são lidos, sem cópia (~ms).
    Os trials retornados não são cópias: não alterar.
    '''
    done = [t for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if t.values is not None]
    if not done: return []
    sgn = np.array([1.0 if d == optuna.study.StudyDirection.MINIMIZE else -1.0 for d in study.directions])
    V = np.array([t.values for t in done], dtype=float) * sgn          # todos os objetivos como minimização
    keep = []
    for i in range(len(V)):
        dom = np.all(V <= V[i], axis=1) & np.any(V < V[i], axis=1)    # algum trial domina o trial i?
        if not dom.any(): keep.append(done[i])
    return keep


def FormatTime(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


NDEC = 3   # casas decimais exibidas na tela (apenas exibição: os valores armazenados não mudam)


def FormatNum(x, nd=NDEC):
    '''
    Texto de x para exibição, com no máximo nd casas decimais nos números não inteiros.
        float/np.floating : nd casas (nan/inf inalterados);  bool/int/np.integer: inalterados
        list/tuple/ndarray: cada elemento formatado, entre colchetes
        dict              : cada valor formatado, como {'chave': valor} (usado nos Params dos trials)
    '''
    if isinstance(x, (bool, int, np.integer)): return str(x)
    if isinstance(x, (float, np.floating)):
        return str(x) if not np.isfinite(x) else f"{x:.{nd}f}"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "[" + ", ".join(FormatNum(v, nd) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k!r}: {FormatNum(v, nd)}" for k, v in x.items()) + "}"
    return str(x)


# ------------------------------------------------------------------ tempo acumulado por instancia
RUNTIME = 'runtime.json'   # ao lado do optuna.db: registro {inicio, fim} de cada processo de otimizacao da instancia
SAVE_DT = 10.0             # intervalo minimo entre gravacoes (s)
_PROC_T0 = None            # inicio do processo atual, quando o otimizador nao informa proc_t0


def _ProcessT0(default):
    global _PROC_T0
    if _PROC_T0 is None: _PROC_T0 = default
    return _PROC_T0


def _StorageDir(storage):
    """Pasta da instancia (a do optuna.db) a partir do storage do Optuna; None se nao for SQLite."""
    url = getattr(storage, 'url', storage if isinstance(storage, str) else None)
    pre = 'sqlite:///'
    if not url or not str(url).startswith(pre): return None
    return os.path.dirname(os.path.abspath(str(url)[len(pre):]))


def _RuntimeLoad(path):
    try:
        with open(path, encoding='utf-8') as fh: d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _RuntimeUnion(runs):
    """Uniao dos intervalos [inicio, fim] dos processos: janelas em paralelo nao sao contadas em dobro."""
    iv = sorted((r['start'], r['end']) for r in runs.values()
                if isinstance(r, dict) and isinstance(r.get('start'), (int, float))
                and isinstance(r.get('end'), (int, float)) and r['end'] >= r['start'])
    tot, s0, e0 = 0.0, None, None
    for a, b in iv:
        if e0 is None or a > e0:
            if e0 is not None: tot += e0 - s0
            s0, e0 = a, b
        else:
            e0 = max(e0, b)
    if e0 is not None: tot += e0 - s0
    return tot


def ClearRuntime(instance_dir):
    """Apaga o historico de tempos da instancia (usado por OptReset: a instancia recomeca do zero)."""
    try:
        p = os.path.join(instance_dir, RUNTIME)
        if os.path.exists(p): os.remove(p)
    except Exception:
        pass


def ProcessStart(instance_dir=None, reset=False):
    """
    Marca o inicio de um processo de otimizacao (uma chamada de Optimize/OptimizeSeq) e devolve o instante.
    reset=True (OptReset) zera o tempo acumulado da instancia. Passar o retorno em context['proc_t0'].
    """
    global _PROC_T0
    if instance_dir and reset: ClearRuntime(instance_dir)
    _PROC_T0 = time.time()
    return _PROC_T0


class EarlyStoppingCallback:
    '''
    Ordem da tela: OPTIMIZATION PROGRESS -> BEST TRIAL [PER nL] -> LAST n_last TRIALS -> melhor da sessão
    (Current session best) -> frente de Pareto (TOTAL NUMBER ...) -> STOPPING STATUS. Nada mais é exibido.
    A tela é redesenhada no lugar (DrawScreen, sem piscar) no máximo a cada refresh segundos (padrão REFRESH_DT = 5 s),
    e sempre no 1º trial, no último trial previsto e na parada antecipada; a contagem da parada é feita a cada trial.
    LAST n_last TRIALS: 'DONE' = trial concluído; Score = média dos WAPE_RUL finais dos rolamentos (MeanWapeRUL;
    sem esses atributos, o valor do objetivo).
    Números não inteiros são exibidos com NDEC casas decimais (FormatNum).
    Tempos exibidos: session (esta chamada de study.optimize) | process (esta chamada de OptimizeSeq, context['proc_t0'])
    | instance (soma de todos os processos da instância, em <instância>/runtime.json; zerada por OptReset).

    context (opcional): informações da execução exibidas no topo da tela.
        nL         : faixa de camadas [min, max] (ou int)
        session    : número da sessão atual;  n_study: total de sessões
        study_name : nome do estudo no banco
        config     : pasta/configuração dos resultados
        n_prev     : trials armazenados antes desta execução
        n_trials   : trials previstos nesta execução;  timeout: limite em segundos
        storage    : banco do Optuna (RDBStorage ou URL)  } se informados, BEST TRIAL mostra o melhor trial
        prefix     : prefixo dos estudos (ex.: 'obj1_')    } de cada nL (todas as sessões/fases armazenadas)
        proc_t0    : instante inicial do processo (ProcessStart); sem ele, o 1º callback do processo define o início
        instance_dir: pasta da instância para o runtime.json (padrão: a pasta do optuna.db, deduzida de storage)
    '''

    def __init__(self, patience: int, n_last: int = 3, context: dict = None, refresh: float = REFRESH_DT):
        self.patience = patience
        self.refresh = refresh                                   # s entre atualizações da tela
        self.t_draw = None                                       # última atualização (None: ainda não desenhada)
        #self.ref = ref
        #self.count = count
        self.best_score = float('inf')
        self.trials_without_improvement = 0
        self.last_trials = deque(maxlen=n_last)
        self.best_trial_info = None
        self.context = context
        self.t0 = time.time()                                    # inicio desta sessao (uma chamada de study.optimize)
        c = context or {}
        self.proc_t0 = c.get('proc_t0') or _ProcessT0(self.t0)   # inicio do processo (uma chamada de OptimizeSeq)
        self.inst_dir = c.get('instance_dir') or _StorageDir(c.get('storage'))   # pasta da instancia (optuna.db)
        self.run_id = f'pid{os.getpid()}_{int(self.proc_t0)}'    # um registro por processo em runtime.json
        self.inst_base, self.inst_base_t, self.inst_runs, self.t_save = 0.0, self.t0, 0, 0.0
        self.SaveRuntime(force=True)                             # registra este processo e le o total da instancia
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
                ses = name.rsplit('_', 1)[-1]                                  # sessão do estudo (s1, s2, ...)
                print(f"nL={nL} | {ses} #{number:04d} | {label}: {FormatNum(score)} | {FormatParams(params)}{mark}")
            else:
                print(f"nL={nL} | no completed trials yet{mark}")

    def PrintStatus(self, study, is_multi, stopped=False):
        '''
        Situação atual, depois de LAST N TRIALS: melhor trial da sessão, número de soluções da frente de Pareto
        (mode='multi') e contagem da parada antecipada.
        '''
        label = 'Score (sum)' if is_multi else 'Score'
        if self.per_nl:
            if self.best_score < float('inf'):
                number = f"#{self.best_number:04d}" if self.best_number is not None else "#----"
                print(f"Current session best: {number} | {label}: {FormatNum(self.best_score)}")
            else:
                print("Current session best: no completed trials yet")
        n_front = len(ParetoFront(study)) if is_multi else 0        # sem study.best_trials (lento; ver ParetoFront)
        if n_front > 0:
            print(f"-> TOTAL NUMBER OF SOLUTIONS IN THE PARETO FRONT: {n_front}")
        print("-" * 80)
        print(f"STOPPING STATUS: {self.trials_without_improvement}/{int(self.patience)} ITERATIONS WITHOUT IMPROVEMENT.")
        if stopped:
            print(f">>> STUDY INTERRUPTED! NO IMPROVEMENTS FOR {int(self.patience)} ITERATIONS.")
        print("-" * 80)

    def SaveRuntime(self, force=False):
        """
        Atualiza o registro deste processo em <instancia>/runtime.json e rele o tempo total da instancia.
        Grava no maximo a cada SAVE_DT s, de forma atomica (os.replace) e sempre protegida: a contagem de
        tempo e apenas informativa e nunca pode interromper a otimizacao.
        """
        now = time.time()
        if not self.inst_dir or (not force and now - self.t_save < SAVE_DT): return
        self.t_save = now
        path = os.path.join(self.inst_dir, RUNTIME)
        try:
            runs = _RuntimeLoad(path)
            runs[self.run_id] = {'start': self.proc_t0, 'end': now, 'pid': os.getpid(),
                                 'inicio': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.proc_t0))}
            tmp = f'{path}.{os.getpid()}.tmp'
            os.makedirs(self.inst_dir, exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as fh: json.dump(runs, fh, indent=1)
            os.replace(tmp, path)
            self.inst_base, self.inst_base_t, self.inst_runs = _RuntimeUnion(runs), now, len(runs)
        except Exception:
            pass

    def Elapsed(self):
        """Tempos em segundos: (sessao, processo, instancia); instancia = None quando nao ha runtime.json."""
        now = time.time()
        inst = self.inst_base + (now - self.inst_base_t) if self.inst_dir else None
        return now - self.t0, now - self.proc_t0, inst

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
              f"This run: {self.n_run}/{int(n_trials) if n_trials is not None else '-'}")
        # sessao (study.optimize) | processo (OptimizeSeq) | instancia (todos os processos ja executados)
        self.SaveRuntime()
        ses, proc, inst = self.Elapsed()
        line = (f"Elapsed: session {FormatTime(ses)} / {FormatTime(timeout) if timeout is not None else '-'}"
                f"  |  process {FormatTime(proc)}")
        if inst is not None:
            line += f"  |  instance {FormatTime(inst)} ({self.inst_runs} run{'s' if self.inst_runs != 1 else ''})"
        print(line)
        print()

    def Render(self, study, is_multi, stopped=False):
        '''Monta a tela inteira em memória e a exibe de uma vez (DrawScreen).'''
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if self.context:
                self.PrintContext(study)
            if self.per_nl:
                self.PrintBestPerNL(is_multi)
            else:
                print("=" * 80)
                print("BEST TRIAL")
                print("=" * 80)
                print(self.best_trial_info if self.best_trial_info else "NO COMPLETED TRIALS FINISHED YET.")
            print()
            print("=" * 80)
            print(f"LAST {self.last_trials.maxlen} TRIALS")
            print("=" * 80)
            for log in self.last_trials:
                print(log)
            print()
            self.PrintStatus(study, is_multi, stopped)
        DrawScreen(buf.getvalue().rstrip('\n'), first=self.t_draw is None)
        self.t_draw = time.time()

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ):
        is_multi = study._is_multi_objective()
        self.n_run += 1

        # 1. Formata a linha do trial atual (Score = média dos WAPE_RUL finais dos rolamentos)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            wr = MeanWapeRUL(trial)
            score_str = FormatNum(wr) if wr is not None else (FormatNum(trial.values) if is_multi else FormatNum(trial.value))
            msg = f"Trial #{trial.number:04d} [DONE] | Score: {score_str} | {FormatParams(trial.params)}"
        else:
            msg = f"Trial #{trial.number:04d} [{trial.state.name}]"

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
                    self.best_trial_info = (f"Trial #{trial.number:03d} | Objetivos: {FormatNum(trial.values)} "
                                            f"(Soma: {FormatNum(current_score)}) | Params: {FormatNum(trial.params)}")
                else:
                    self.best_trial_info = f"Trial #{trial.number:03d} | Valor: {FormatNum(trial.value)} | Params: {FormatNum(trial.params)}"
            else:
                self.trials_without_improvement += 1

        # 3. Tela (ordem: OPTIMIZATION PROGRESS -> BEST TRIAL PER nL -> LAST N TRIALS -> Current session best ->
        #    TOTAL NUMBER ... -> STOPPING STATUS), redesenhada no máximo a cada self.refresh s; sempre no 1º trial,
        #    no último trial previsto desta execução e na parada antecipada
        stop = self.trials_without_improvement >= self.patience
        n_tr = (self.context or {}).get('n_trials')
        last = n_tr is not None and self.n_run >= int(n_tr)
        if stop or last or self.t_draw is None or time.time() - self.t_draw >= self.refresh:
            self.Render(study, is_multi, stopped=stop)

        # 4. Condição de parada
        if stop:
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


def DbTrials(db_path, state='COMPLETE', study=None, keep_study=False):
    """
    Reúne em um único DataFrame os trials de TODOS os estudos de um optuna.db.

        db_path   : caminho do optuna.db (ou a URL 'sqlite:///...').
        state     : estado mantido; 'COMPLETE' (padrao) descarta PRUNED, FAIL e RUNNING. None mantém todos.
        study     : None (todos os estudos) ou nome/prefixo (str) ou lista de nomes/prefixos.
        keep_study: True acrescenta a coluna 'study' com o nome do estudo de cada linha.

    Colunas: params_<hiperparâmetro> e user_attrs_<métrica>, em ordem alfabética. Estudos com números de
    camadas diferentes geram colunas params_nR_layer_<l> distintas, com NaN onde não se aplicam.
    """
    url = str(db_path)
    if not url.startswith('sqlite:///'):
        url = 'sqlite:///' + os.path.abspath(url).replace(os.sep, '/')
    names = optuna.get_all_study_names(url)
    if study is not None:
        sel = [study] if isinstance(study, str) else list(study)
        names = [n for n in names if any(n == x or n.startswith(x) for x in sel)]
    if not names: raise ValueError(f'nenhum estudo encontrado em {url}')

    dfs = []
    for n in names:
        # attrs limita as colunas a params_* e user_attrs_*; 'state' é lido apenas para filtrar (optuna 5
        # não aceita states= em trials_dataframe) e removido em seguida
        d = optuna.load_study(study_name=n, storage=url).trials_dataframe(attrs=('params', 'user_attrs', 'state'))
        if state is not None: d = d[d['state'] == state]
        d = d.drop(columns='state')
        if keep_study: d.insert(0, 'study', n)
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    cols = ([c for c in df.columns if c == 'study']
            + sorted(c for c in df.columns if c.startswith('params_'))
            + sorted(c for c in df.columns if c.startswith('user_attrs_')))
    return df[cols]


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