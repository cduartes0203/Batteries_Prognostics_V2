'''
Otimização (Optuna) do AutoCloud_V5 (rho estático ou dinâmico) em uma sequência de um ou mais rolamentos.

    OptimizeSeq : rolamentos assimilados em sequência pelo mesmo modelo (AutoCloud.restart entre rolamentos); com um
                  único rolamento em data, otimização de um rolamento (substitui a antiga função Optimize).
                  mode='single': minimiza a média (ou soma) de WAPE_RUL*WAPE_HI final de cada rolamento;
                  mode='multi' : minimiza separadamente F_RUL = média (ou soma) de WAPE_RUL e F_HI = média (ou soma)
                                 de WAPE_HI (otimização multiobjetivo, frente de Pareto). WAPE sem normalização.
    SeqTable    : Optimization.csv reconstruído do banco (n_best melhores por nº de camadas).
    ParetoTable : mode='multi': todos os trials, com os objetivos e a frente de Pareto (gráfico da frente).
    BestParams  : melhor linha de um Optimization.csv no formato de ParseParams.
    BestParamsW : idem, com a janela w da mesma linha: (params, w).
    SimSeq      : simulação final de uma sequência de rolamentos com parâmetros fixos (mesmo procedimento de OptimizeSeq),
                  com rho estático ou dinâmico (rho_kw: rho_mode e parâmetros da atualização, ver AutoCloud_V5).
    OptimizeRho : otimização do rho (estático ou dinâmico) com o modelo FIXO encontrado por OptimizeSeq: maximiza
                  V = média(cov)*média(sp) do intervalo do HI (ou da RUL, oráculo); resultados em <instância>/rho/.
    RhoTable    : rho/Optimization.csv reconstruído do banco.   BestRho: melhor (rho, rho_kw) para SimSeq.
    RhoReplay   : atualização do rho repetida sobre as séries de uma simulação (sem simular o modelo).

nLS = [min, max] executa n_study sessões de n_trials trials para CADA nº de camadas.
Uso no notebook (na pasta principal): from Functions.TedaOptimize_V5 import *

Mudanças em relação à V4_8:
    - usa AutoCloud_V5 (rho_mode 'static', 'grid', 'grad' ou 'conformal'). As predições pontuais não dependem do rho:
      OptimizeSeq (RUL pontual, WAPE_RUL e WAPE_HI) dá os mesmos resultados da V4_8 e não recebe parâmetros do rho;
      o rho (estático ou dinâmico) entra na simulação final (SimSeq, rho e rho_kw) e nos intervalos do HI e da RUL;
    - função Optimize removida: OptimizeSeq com data = [(nome, RS, HI)] (um rolamento) otimiza um único rolamento
      (pasta <FileName>_<B1_1>_restart_...; com um rolamento o restart não ocorre), com o objetivo de OptimizeSeq
      (mode='single': WAPE_RUL*WAPE_HI; mode='multi': WAPE_RUL e WAPE_HI), e não WAPE_RUL + WAPE_HI como Optimize;
    - otimização em duas etapas: OptimizeSeq (parâmetros do modelo: WAPE_RUL e WAPE_HI) e depois OptimizeRho
      (parâmetros do rho, com o modelo fixo; mesmo optuna.db da instância, tabela em <instância>/rho/);
    - SimSeq(rho_kw=None): dicionário com rho_mode e os parâmetros da atualização online do rho, repassado ao
      AutoCloud_V5; métricas por rolamento incluem rho_ini e rho_fim (rho usado no 1º e no último passo).

Mudanças da V4_8 em relação à V4_7:
    - a janela do DSI em X_k (w, PrepareData) também pode ser otimizada: wS = [min, max] (inteiros >= 1).
      wS = None (padrão) mantém w fixo (w = 1 por padrão), como na V4_7. Em ambos os casos w é registrado no banco
      como parâmetro do trial ('w', fixo quando wS = None, como n_layers), de modo que as combinações repetidas
      (_Key) e as tabelas consideram w;
    - Optimization.csv ganha a coluna 'w' logo após 'activation' (as 10 primeiras colunas de parâmetros não mudam:
      BestParams continua compatível); BestParamsW devolve também o w da melhor linha;
    - OptimizeSeq(worker, n_workers): vários processos no MESMO estudo (mesmo optuna.db), com semente por worker,
      TPE com constant_liar, combinações repetidas verificadas também nos trials dos outros workers e patience
      zerada por melhora de qualquer worker. Com worker = 0 e n_workers = 1 (padrão), igual à versão anterior.

Mudanças da V4_7 em relação à V4_6:
    - a semente do sampler de cada sessão (_Seed) depende também do nº de trials já armazenados no estudo (n_prev).
      Retomar um estudo com o mesmo OptSeed gera uma sequência nova (sem repetir as sugestões aleatórias da execução
      anterior) e continua reprodutível para o mesmo estado do banco;
    - combinação de parâmetros já testada no estudo (nesta execução ou em anteriores) não é simulada de novo:
      o trial reutiliza o resultado armazenado (_Seen, _Track, _Repeat).
'''
import os
import time
import numpy as np
import pandas as pd
import optuna
from optuna.exceptions import TrialPruned
from Functions.AutoCloud_V5 import AutoCloud, RefEoL
from Functions.Utils import PrepareData, ParseParams
from Functions.Utils_OPT_R2 import EarlyStoppingCallback, SelSampler, ProcessStart

__all__ = ['OptimizeSeq', 'SeqTable', 'ParetoTable', 'BestParams', 'BestParamsW', 'SimSeq',
           'OptimizeRho', 'RhoTable', 'BestRho', 'RhoReplay']


def _WRange(w, wS):
    # Faixa da janela w do DSI: wS = [min, max]; None -> w fixo ([w, w], registrado no banco como n_layers)
    if wS is None: wS = [int(w), int(w)]
    wS = [int(wS[0]), int(wS[1])]
    if not 1 <= wS[0] <= wS[1]: raise ValueError(f'wS inválido: requer 1 <= min <= max (recebido {wS}).')
    return wS


def _Storage(db_path, timeout=300, tries=10):
    # Banco SQLite do Optuna. timeout (s): espera pelo bloqueio do arquivo quando vários processos (janelas em paralelo)
    # gravam no mesmo banco, em vez de falhar com 'database is locked'. Banco novo aberto ao mesmo tempo por vários
    # processos: a criação das tabelas pode colidir ('table ... already exists'); nova tentativa encontra o banco pronto
    url = 'sqlite:///' + db_path.replace(os.sep, '/')
    for i in range(tries):
        try:
            return optuna.storages.RDBStorage(url, engine_kwargs={'connect_args': {'timeout': timeout}})
        except Exception:
            if i == tries - 1: raise
            time.sleep(1 + i)


def _Complete(study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def _Seed(OptSeed, nL, i, study_name, storage, worker=0):
    '''
    Semente do sampler da sessão i do nL. None se OptSeed é None (sequência diferente a cada execução).
    Combina (OptSeed, nL, i, n_prev[, worker]) com np.random.SeedSequence, n_prev = nº de trials já armazenados no
    estudo (0 se o estudo ainda não existe). Assim:
        - sessões e nº de camadas diferentes recebem sementes independentes (sem colisões como OptSeed + 100);
        - retomar o estudo com o mesmo OptSeed não repete as sugestões aleatórias da execução anterior
          (n_prev mudou), o que ocorria na V4_6 com RandomSampler e na fase inicial do TPE/GP;
        - workers do mesmo estudo (processos em paralelo, worker = 0, 1, ...) que começam juntos veem o mesmo n_prev;
          o nº do worker os separa (worker 0 mantém a semente de uma execução com um único processo);
        - o resultado é reprodutível para o mesmo OptSeed e o mesmo estado do banco.
    Retorna (seed, n_prev). Chamar depois de OptReset (estudo apagado -> n_prev = 0).
    '''
    try:
        n_prev = len(storage.get_all_trials(storage.get_study_id_from_name(study_name), deepcopy=False))
    except KeyError:
        n_prev = 0
    if OptSeed is None: return None, n_prev
    key = [int(OptSeed), int(nL), int(i), n_prev] + ([int(worker)] if worker else [])
    return int(np.random.SeedSequence(key).generate_state(1)[0]), n_prev


def _Sampler(mode, seed, n_workers=1):
    # SelSampler; com vários workers no mesmo estudo, o TPE usa constant_liar: os trials em execução nos outros
    # processos entram no modelo com um valor pessimista, o que evita que todos sugiram a mesma região ao mesmo tempo
    if n_workers > 1 and mode is not None and str(mode).lower() == 'tpe':
        return optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=100, seed=seed,
                                          constant_liar=True)
    return SelSampler(mode=mode, seed=seed)


# Combinações já testadas. A nova semente (_Seed) evita repetir a SEQUÊNCIA de sugestões da execução anterior, mas o
# sampler ainda pode sugerir, por acaso, uma combinação já avaliada (espaço inteiro, TPE concentrado na região boa).
# Como o objetivo é determinístico (sementes fixas nas RNNs), a combinação repetida não é simulada de novo:
#   COMPLETE anterior -> o trial novo recebe o mesmo valor e os mesmos user_attrs (WAPE_RUL_<b>, WAPE_HI_<b>);
#   PRUNED anterior   -> o trial novo é podado (mesmos limiares; a decisão do Hyperband poderia variar, mas é aceita).
# O trial repetido fica no banco com user_attr 'repetido_de' = nº do trial original; SeqTable já ignora conjuntos
# de parâmetros repetidos.
_DONE_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def _Key(params):
    return tuple(sorted(params.items()))


def _Seen(study):
    seen = {}
    for t in study.get_trials(deepcopy=False, states=_DONE_STATES):
        if 'repetido_de' not in t.user_attrs: seen.setdefault(_Key(t.params), t)
    return seen


def _Track(seen):
    # Callback: acrescenta cada trial concluído/podado desta execução ao dicionário de combinações testadas
    def cb(study, t):
        if t.state in _DONE_STATES and 'repetido_de' not in t.user_attrs: seen.setdefault(_Key(t.params), t)
    return cb


def _Repeat(trial, prev):
    trial.set_user_attr('repetido_de', prev.number)
    if prev.state == optuna.trial.TrialState.PRUNED: raise TrialPruned()
    for k, v in prev.user_attrs.items(): trial.set_user_attr(k, v)
    return tuple(prev.values) if len(prev.values) > 1 else prev.value   # mode='multi': os dois objetivos


def _Score(t):
    # Valor escalar de um trial concluído: o objetivo (mode='single') ou a soma dos dois objetivos normalizados
    # (mode='multi'; o trial de menor soma pertence à frente de Pareto). Mesmo critério do EarlyStoppingCallback.
    return float(sum(t.values))


def _ParetoMask(R, H):
    # Pontos não dominados (minimização de R e H): varredura em ordem crescente de R (empates por H);
    # pontos idênticos a um ponto da frente também ficam na frente
    R, H = np.asarray(R, dtype=float), np.asarray(H, dtype=float)
    front = np.zeros(len(R), dtype=bool)
    best, last = np.inf, None
    for i in np.lexsort((H, R)):
        if H[i] < best: front[i], best, last = True, H[i], (R[i], H[i])
        elif (R[i], H[i]) == last: front[i] = True
    return front


def _Lookup(seen, trial, n_workers):
    # Combinação já testada? Com vários workers no mesmo estudo, os trials concluídos pelos outros processos só estão
    # no banco: na falta no dicionário local, relê o estudo (uma consulta por trial novo) antes de simular
    key = _Key(trial.params)
    prev = seen.get(key)
    if prev is None and n_workers > 1:
        seen.update({k: t for k, t in _Seen(trial.study).items() if k not in seen})
        prev = seen.get(key)
    return prev


def _SharedBest(callback, n_workers):
    # Callback (antes do EarlyStoppingCallback): com vários workers, melhora encontrada por outro processo também zera
    # a contagem de patience deste processo (patience = trials deste worker sem melhora do ESTUDO)
    def cb(study, t):
        if n_workers <= 1: return
        if len(study.directions) > 1:                # mode='multi': menor soma dos objetivos na frente de Pareto
            front = study.best_trials
            if not front: return
            best = min(front, key=_Score)
        else:
            try: best = study.best_trial
            except ValueError: return                # nenhum trial concluído ainda
        if best.number != t.number and _Score(best) < callback.best_score:
            callback.best_score, callback.best_number = _Score(best), best.number
            callback.best_trial_info = f'Trial #{best.number:03d} (outro worker) | Score: {_Score(best):.4f} | Params: {best.params}'
            callback.trials_without_improvement = 0
    return cb

def OptimizeSeq(
        data=None,
        FileName=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,actS=None,
        feedback='symmetric',wtaG=False,wtaP=False,
        eol=0.3,w=1,wS=None,agg='mean',state='keep',
        OptReset=False,OptWarm=False,n_best=10,
        folder='Optimization',write_table=True,worker=0,n_workers=1,mode='single'):

    '''
        mode: 'single' (padrão) -> um objetivo: agg, sobre os rolamentos, de WAPE_RUL*WAPE_HI de cada rolamento;
              'multi'           -> dois objetivos minimizados separadamente (frente de Pareto):
                                   F_RUL = agg_b(WAPE_RUL_b) e F_HI = agg_b(WAPE_HI_b) (sem normalização).
              Em 'multi': estudos com prefixo mseq<agg>_ (não se misturam com os seq<agg>_ do modo single no mesmo
              banco); sem trial.report/should_prune (o Optuna não os permite com vários objetivos), as podas por
              limiares continuam; patience e melhor trial usam a soma F_RUL + F_HI (o menor valor está na frente).
        Otimização em uma SEQUÊNCIA de um ou mais rolamentos assimilada pelo mesmo modelo (AutoCloud_V5.restart):
            o 1º rolamento é simulado até o EoL; antes de cada rolamento seguinte, teda.restart(ref) zera RUL/HI/WAPE e
            mantém grânulos, RNNs e modelo do erro, e a simulação prossegue (mesmo procedimento da última célula).

        data: lista ordenada de (nome, RS, HI), ex.: [('B1_1', RS11, HI11['PC1']), ('B1_2', RS12, HI12['PC1'])];
              um único rolamento ([('B1_1', RS11, HI11['PC1'])]) = otimização de um rolamento (sem restart).
        agg : 'mean' -> minimiza a média, sobre os rolamentos, de (WAPE_RUL + WAPE_HI) final de cada rolamento;
              'sum'  -> minimiza a soma.
        state: estado recorrente das RNNs no início de cada rolamento seguinte (teda.restart):
              'keep' (estado do fim do rolamento anterior), 'reset' (estado inicial) ou 'fixed' (equilíbrio para a
              primeira janela Y[0] do novo rolamento). Pastas: ..._restart_... ('keep') ou ..._restart-<state>_...
        Cada rolamento usa o próprio HI em RefEoL (ref, J, L locais). WAPE_RUL e WAPE_HI de cada rolamento são os valores
        finais (ponderados por (k - k_seg)^2, locais ao rolamento), lidos antes do restart seguinte.
        Faixas de busca [min, max]: mS (m), nIS (nI), nLS (nº de camadas), nRS (neurônios por camada), nOS (nO), mOS (mO),
        N1S, N2S, N3S (expoentes das taxas de aprendizado), τS (τ), actS (ativação), wS (janela w do DSI; None = w fixo).
        Sessões: n_study sessões de n_trials trials (timeout em s, patience em trials sem melhora) para cada nº de camadas;
        OptSampler/OptSeed: sampler e semente base; OptReset apaga os estudos; OptWarm enfileira o melhor das outras
        sessões; n_best: linhas por nº de camadas no Optimization.csv; folder: pasta dos resultados.
        O mesmo w é usado em todos os rolamentos do trial. O rho não entra na otimização (predições pontuais).

        Resultados em <folder>/<FileName>_<B1_1-B1_2>_restart[-<state>]_<feedback>_G<wtaG>P<wtaP>/ (optuna.db e mono/Optimization.csv).
        Optimization.csv: coluna do objetivo (WAPE_mean ou WAPE_sum), os 10 parâmetros (formato de ParseParams),
        w e WAPE_RUL_<nome>, WAPE_HI_<nome> de cada rolamento.
        OptPrune: poda ao fim de cada rolamento (trial.report do objetivo parcial) e por limiares dentro do rolamento.
        write_table: False = não grava o Optimization.csv ao fim (execuções em paralelo no mesmo banco: a tabela é
              gerada uma única vez com SeqTable depois que todas terminam). Retorna None nesse caso.
        worker, n_workers: vários processos (workers) otimizando os MESMOS estudos ao mesmo tempo, no mesmo optuna.db
              (worker = 0, ..., n_workers-1; cada processo chama OptimizeSeq com o seu nº e write_table=False).
              O Optuna distribui os trials entre os processos pelo banco. Com n_workers > 1:
                - a semente do sampler inclui o nº do worker (_Seed);
                - o TPE usa constant_liar (_Sampler);
                - combinações concluídas pelos outros workers são relidas do banco antes de simular (_Lookup);
                - melhora encontrada por outro worker zera a contagem de patience deste (_SharedBest).
              n_trials, timeout e patience valem POR worker.
    '''
    if not data: raise ValueError('Informe data com pelo menos um rolamento: [(nome, RS, HI), ...].')
    if agg not in ('mean', 'sum'): raise ValueError("agg deve ser 'mean' ou 'sum'.")
    if mode not in ('single', 'multi'): raise ValueError("mode deve ser 'single' ou 'multi'.")
    multi = mode == 'multi'
    if state not in ('keep', 'reset', 'fixed'): raise ValueError("state deve ser 'keep', 'reset' ou 'fixed'.")
    names_b = [str(d[0]) for d in data]
    if len(set(names_b)) != len(names_b): raise ValueError('Nomes de rolamentos repetidos em data.')
    sigs = []
    for nm, RS_b, HI_b in data:
        s_ = np.asarray(HI_b, dtype=float)
        if s_.ndim > 1 and s_.shape[1] != 1: raise ValueError(f'{nm}: HI deve ter uma única coluna.')
        sigs.append(s_.ravel())
    tag_rs = 'restart' if state == 'keep' else f'restart-{state}'   # 'keep': mesma pasta das versões anteriores
    FileName = f'{FileName}_{"-".join(names_b)}_{tag_rs}_{feedback}_G{int(wtaG)}P{int(wtaP)}'

    if n_study is None: n_study = 1
    if timeout is None: timeout = 60
    if n_trials is None and patience is None: n_trials = 25
    if patience is None:
        if n_trials is None or n_trials < 0.5e4: patience = 500
        else: patience = int(0.02*n_trials)

    if mS  is None: mS  = [1.75,4.25]
    if nIS is None: nIS = [2,40]
    if nLS is None: nLS = [1,5]
    if not 1 <= nLS[0] <= nLS[1]: raise ValueError(f'nLS inválido: requer 1 <= min <= max (recebido {nLS}).')
    if nRS is None: nRS = [1,60]
    if nOS is None: nOS = [1,40]
    if mOS is None: mOS = [0,40]
    if N1S is None: N1S = [1,9]
    if N2S is None: N2S = [1,9]
    if N3S is None: N3S = [1,9]
    if τS  is None: τS  = [1,40]
    if actS is None: actS = [0,1,2]
    wS = _WRange(w, wS)

    # nO limitado para que o fluxo de TODOS os rolamentos alcance o EoL
    for nm, s_ in zip(names_b, sigs):
        if not np.any(s_ <= eol): raise ValueError(f'{nm}: o HI nunca atinge o EoL.')
    nO_max = min(len(s_) - int(np.argmax(s_ <= eol)) for s_ in sigs)
    if nOS[1] > nO_max:
        nOS = [nOS[0], nO_max]
        print(f'nO limitado a {nOS} para que o fluxo de todos os rolamentos alcance o EoL.')
    if nOS[0] > nOS[1]: raise ValueError(f'nOS inválido: o limite para alcançar o EoL é nO <= {nO_max}.')

    obj_name = f'SCORE_{agg}' if multi else f'WAPE_{agg}'
    study_dir = os.path.join(folder, FileName, 'mono')
    os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')

    db_path = os.path.abspath(os.path.join(folder, FileName, 'optuna.db'))
    storage = _Storage(db_path)
    # Relógio do processo e tempo acumulado da instância (runtime.json, ao lado do optuna.db).
    # Em paralelo, só o processo principal (worker 0) zera o histórico, para não apagar o tempo dos outros.
    proc_t0 = ProcessStart(os.path.dirname(db_path), reset=OptReset and worker == 0)
    prefix = _SeqPrefix(agg, mode)                 # objetivos 'mean'/'sum' e modos single/multi não se misturam no banco
    summary = []
    nL_cur = None                                   # nº de camadas da rodada atual (laço abaixo)
    seen_cur = {}                                   # combinações já testadas no estudo atual (_Seen; atualizado por _Track)
    Agg = (lambda v: float(np.mean(v))) if agg == 'mean' else (lambda v: float(np.sum(v)))

    def objective(trial):
        m = trial.suggest_float('m', mS[0], mS[1],step=0.0125)
        nI = trial.suggest_int('nI', nIS[0], nIS[1])
        n_layers = trial.suggest_int('n_layers', nL_cur, nL_cur)   # fixo na rodada (registrado no banco)
        nR = [trial.suggest_int(f'nR_layer_{l}', nRS[0], nRS[1]) for l in range(n_layers)]
        nO = trial.suggest_int('nO', nOS[0], nOS[1])
        N1 = trial.suggest_int('N1', N1S[0], N1S[1])
        N2 = trial.suggest_int('N2', N2S[0], N2S[1])
        N3 = trial.suggest_int('N3', N3S[0], N3S[1])
        τ = trial.suggest_int('τ', τS[0], τS[1])
        act = trial.suggest_int('act', actS[0], actS[1])
        mO = trial.suggest_int('mO', max(1, mOS[0]), min(mOS[1], nI)) 
        w_ = trial.suggest_int('w', wS[0], wS[1])          # janela do DSI (fixa quando wS = None)
        if mO > nI: raise TrialPruned()
        prev = _Lookup(seen_cur, trial, n_workers)
        if prev is not None: return _Repeat(trial, prev)   # combinação já testada: reutiliza o resultado, sem simular

        # Janelas, EoL e RUL real de cada rolamento (com o próprio HI)
        ds = []
        for (nm, RS_b, HI_b), s_ in zip(data, sigs):
            X, Y, Z, kX = PrepareData(RS_b, HI_b, w_, nI, nO, mO)
            try: ref, J, L = RefEoL(s_, eol, kX, strict=True)
            except ValueError: raise TrialPruned()
            ds.append((nm, X, Y, Z, ref, J, L))

        teda = AutoCloud(m=m,nI=nI,nR=nR,nO=nO,mO=mO,ηS=[N1,N2,N3],act=act,
                         tau=τ,eol=eol,ref=ds[0][4],wtaG=wtaG,wtaP=wtaP,feedback=feedback)
        errs, rs, hs = [], [], []
        for b, (nm, X, Y, Z, ref, J, L) in enumerate(ds):
            if b > 0: teda.restart(ref, state=state, x0=Y[0])   # mantém grânulos, pesos e modelo do erro; zera RUL/HI/WAPE
            for j in range(J):
                teda.run(X[j])
                teda.RUL_Prediction(Y[j],mode='single',lim=L)   # WAPE_RUL do passo
                teda.Adapt(Y[j],Z[j])                            # WAPE_HI do passo e treino das RNNs
                if not np.isfinite(teda.hiP[-1]): raise TrialPruned()
                if OptPrune and j >= 40 and j%8 == 0:
                    if teda.wape_RUL > 0.6 or teda.wape_HI > 0.3: raise TrialPruned()
            #if teda.wape_RUL > 0.25 or teda.wape_HI > 0.125: raise TrialPruned()
            wR, wH = float(teda.wape_RUL), float(teda.wape_HI)
            trial.set_user_attr(f'WAPE_RUL_{nm}', wR)
            trial.set_user_attr(f'WAPE_HI_{nm}', wH)
            
            #wR = (wR - 0)/(0.3 - 0)
            #wH = (wH - 0)/(0.2 - 0) 
            errs.append(wR * wH)
            #errs.append(wR + wH)
            rs.append(wR); hs.append(wH)             # mode='multi': objetivos separados

            if OptPrune and not multi:
                trial.report(Agg(errs), b)           # objetivo parcial ao fim de cada rolamento
                if trial.should_prune(): raise TrialPruned()
        if multi: return Agg(rs), Agg(hs)
        else: return Agg(errs)

    for nL in range(nLS[0], nLS[1] + 1):
        nL_cur = nL
        base_name = f'{prefix}L{nL}-{nL}'
        print(f'\n===== nº de camadas: {nL} ({n_study} sessão(ões) x {n_trials} trials) | {" -> ".join(names_b)} | {obj_name} =====')
        for i in range(n_study):
            study_name = f'{base_name}_s{i+1}'
            print(f'iteration: {i+1} | study: {study_name}')
            if OptReset and study_name in optuna.get_all_study_names(storage):
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f'  {study_name}: estudo anterior apagado (OptReset=True).')

            seed, _ = _Seed(OptSeed, nL, i, study_name, storage, worker)   # depende de n_prev (e do worker)
            if multi:
                study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                            directions=['minimize', 'minimize'],pruner=optuna.pruners.NopPruner(),
                                            sampler=_Sampler(OptSampler,seed,n_workers))
            else:
                study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,direction='minimize',
                                            pruner=optuna.pruners.HyperbandPruner(),sampler=_Sampler(OptSampler,seed,n_workers))
            n_prev, done = len(study.trials), _Complete(study)
            seen_cur.clear(); seen_cur.update(_Seen(study))   # inclui as execuções anteriores (retomada)

            if OptWarm and worker == 0:
                others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
                cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
                if cand:
                    best = min(cand, key=_Score)
                    study.enqueue_trial(best.params, skip_if_exists=True)
                    print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {_Score(best):.4f}).')

            callback = EarlyStoppingCallback(patience=patience, context={
                'nL': [nL, nL], 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
                'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout, 'proc_t0': proc_t0,
                'storage': storage, 'prefix': prefix})
            if done:
                best_prev = min(done, key=_Score)
                callback.best_score = _Score(best_prev)
                callback.best_number = best_prev.number
                callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {_Score(best_prev):.4f} | Params: {best_prev.params}'
            print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else '')
                  + f' | semente do sampler: {seed}' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
            study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout,
                           callbacks=[_Track(seen_cur), _SharedBest(callback, n_workers), callback])

            done = _Complete(study)
            summary.append((study_name, n_prev, len(study.trials), min(_Score(t) for t in done) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials (todos os workers) | melhor {best:.4f}')

    if not write_table:
        print('write_table=False: Optimization.csv não foi atualizado (use SeqTable ao fim de todas as janelas).')
        return None
    return SeqTable(db_path, names_b, agg=agg, n_best=n_best, out_path=out_path, w=w, mode=mode)


def _SeqPrefix(agg, mode='single'):
    # Prefixo dos estudos de OptimizeSeq no banco: seq<agg>_ (single, como antes) ou mseq<agg>_ (multi)
    return f'seq{agg}_' if mode == 'single' else f'mseq{agg}_'


def _SeqRows(db_path, names_b, agg, mode, w):
    # Todos os trials concluídos dos estudos <prefixo de mode/agg> (uma linha por combinação de parâmetros, menor escore
    # primeiro): escore, parâmetros, w, [OBJ_RUL_<agg>, OBJ_HI_<agg> (multi)], WAPE_RUL_<b>, WAPE_HI_<b>, _nL e
    # [pareto (multi)]. None se não houver trial concluído.
    multi = mode == 'multi'
    storage = _Storage(db_path)
    prefix = _SeqPrefix(agg, mode)
    obj_name = f'SCORE_{agg}' if multi else f'WAPE_{agg}'
    obj_cols = [f'OBJ_RUL_{agg}', f'OBJ_HI_{agg}'] if multi else []
    seg_cols = [c for nm in names_b for c in (f'WAPE_RUL_{nm}', f'WAPE_HI_{nm}')]
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['act'],p.get('w', w)]
            rows.append([_Score(t)] + prm + (list(t.values) if multi else [])
                        + [t.user_attrs.get(c, np.nan) for c in seg_cols] + [p['n_layers']])
    if not rows: return None
    df = pd.DataFrame(rows, columns=[obj_name] + _PRM_NAMES + obj_cols + seg_cols + ['_nL'])
    df = df.sort_values(obj_name).drop_duplicates(subset=_PRM_NAMES).reset_index(drop=True)
    if multi: df['pareto'] = _ParetoMask(df[obj_cols[0]], df[obj_cols[1]]).astype(int)
    return df


_PRM_NAMES = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation', 'w']


def SeqTable(db_path, names_b, agg='mean', n_best=10, out_path=None, w=1, mode='single'):
    '''
    Reconstrói o Optimization.csv de OptimizeSeq a partir do banco do Optuna (todos os estudos seq<agg>_*), sem executar
    trials: n_best melhores por nº de camadas, com w e o WAPE de cada rolamento. Usado ao fim de OptimizeSeq e, nas
    execuções em paralelo (uma janela por faixa de nº de camadas, mesmo banco), uma única vez depois que todas terminam.
        db_path : caminho do optuna.db.
        names_b : nomes dos rolamentos, na ordem (ex.: ['B1_1', 'B1_2']).
        out_path: caminho do .csv; None = <pasta do banco>/mono/Optimization.csv.
        w       : valor da coluna w para trials sem o parâmetro 'w' (bancos da V4_7 ou anteriores).
        mode    : 'single' (estudos seq<agg>_) ou 'multi' (estudos mseq<agg>_ de OptimizeSeq(mode='multi')).
    Colunas gravadas: objetivo (WAPE_<agg>; multi: SCORE_<agg> = F_RUL + F_HI, o menor valor está na frente de Pareto),
    os parâmetros, w e, para cada rolamento na ordem da sequência, WAPE_RUL_<b> e WAPE_HI_<b> sem normalização, finais
    (último passo do rolamento enviado ao modelo, no EoL). A 1ª coluna é usada por BestParams/BestParamsW.
    Linhas: n_best menores valores da 1ª coluna por nº de camadas; em 'multi', também todos os pontos da frente de
    Pareto (os objetivos e a marcação da frente ficam em ParetoTable, não no .csv).
    '''
    if agg not in ('mean', 'sum'): raise ValueError("agg deve ser 'mean' ou 'sum'.")
    if mode not in ('single', 'multi'): raise ValueError("mode deve ser 'single' ou 'multi'.")
    multi = mode == 'multi'
    if not os.path.exists(db_path): raise FileNotFoundError(f'{db_path} não encontrado.')
    if out_path is None:
        os.makedirs(os.path.join(os.path.dirname(db_path), 'mono'), exist_ok=True)
        out_path = os.path.join(os.path.dirname(db_path), 'mono', 'Optimization.csv')
    obj_name = f'SCORE_{agg}' if multi else f'WAPE_{agg}'
    obj_cols = [f'OBJ_RUL_{agg}', f'OBJ_HI_{agg}']
    cols = [obj_name] + _PRM_NAMES + [c for nm in names_b for c in (f'WAPE_RUL_{nm}', f'WAPE_HI_{nm}')]

    df_all = _SeqRows(db_path, names_b, agg, mode, w)
    if df_all is None:
        print('Nenhum trial concluído: Optimization.csv não foi gerado.')
        return pd.DataFrame(columns=cols)
    if multi:
        keep = df_all.groupby('_nL', group_keys=False).head(n_best).index.union(df_all.index[df_all['pareto'] == 1])
        df_stdy = df_all.loc[keep]
    else:
        df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best)
    df_stdy = df_stdy.sort_values(['_nL', obj_name], ascending=[True, False])

    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas' + (' + frente de Pareto' if multi else '')
          + f' ({out_path})')
    if multi:
        fr = df_all[df_all['pareto'] == 1].sort_values(obj_cols[0])
        print(f'  frente de Pareto: {len(fr)} pontos não dominados entre {len(df_all)} combinações '
              f'(F_RUL de {fr[obj_cols[0]].min():.4f} a {fr[obj_cols[0]].max():.4f}, '
              f'F_HI de {fr[obj_cols[1]].min():.4f} a {fr[obj_cols[1]].max():.4f})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp[obj_name].min():.4f} | pior mantido {grp[obj_name].max():.4f}')

    df_stdy = df_stdy[cols].reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


def ParetoTable(db_path, names_b, agg='mean', w=1):
    '''
    mode='multi': todos os trials concluídos dos estudos mseq<agg>_ (uma linha por combinação de parâmetros), com
    SCORE_<agg>, parâmetros, w, OBJ_RUL_<agg> (F_RUL), OBJ_HI_<agg> (F_HI), WAPE de cada rolamento, nL e pareto
    (1 = não dominado). Ordenada por SCORE (1ª linha = modelo escolhido). Não grava arquivo (gráfico da frente).
    '''
    if not os.path.exists(db_path): raise FileNotFoundError(f'{db_path} não encontrado.')
    df = _SeqRows(db_path, names_b, agg, 'multi', w)
    if df is None: return pd.DataFrame()
    return df.rename(columns={'_nL': 'nL'})


def BestParams(csv_path):
    '''Melhor linha (menor erro, 1ª coluna) de um Optimization.csv de OptimizeSeq, no formato de ParseParams:
    [m, nI, nR, nO, mO, N1, N2, N3, τ, act].'''
    dfP = pd.read_csv(csv_path).iloc[:, :11]
    return _Parse10(dfP.sort_values(dfP.columns[0], ascending=True).iloc[0, 1:].values)


def _Parse10(vals10, w=1):
    # Utils.ParseParams passou a exigir 11 itens [..., act, w]: completa com w e devolve os 10 primeiros
    # (formato [m, nI, nR, nO, mO, N1, N2, N3, τ, act] usado por SimSeq)
    return ParseParams(list(vals10) + [w])[:10]


def BestParamsW(csv_path, w=1):
    '''Melhor linha (menor erro, 1ª coluna) de um Optimization.csv de OptimizeSeq: (params, w), com params no
    formato de ParseParams [m, nI, nR, nO, mO, N1, N2, N3, τ, act] e w da coluna 'w' da mesma linha
    (tabelas sem a coluna 'w', da V4_7 ou anteriores: devolve o w informado).'''
    df = pd.read_csv(csv_path)
    row = df.sort_values(df.columns[0], ascending=True).iloc[0]
    params = _Parse10(row.iloc[1:11].values)
    return params, int(row['w']) if 'w' in df.columns else int(w)


def SimSeq(params, data, state='keep', mode_rul='interval', rho=0.1, intr_state='carry',
           eol=0.3, w=1, feedback='symmetric', wtaG=False, wtaP=False, store=True, on_end=None, rho_kw=None):
    '''
    Simula uma sequência de rolamentos com parâmetros fixos, no mesmo procedimento de OptimizeSeq:
        1º rolamento até o EoL (SimGranularRUL); antes de cada rolamento seguinte, teda.restart(ref, state, Y[0]).
    params  : [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (ParseParams/BestParams).
    data    : lista ordenada de (nome, RS, HI).
    mode_rul: 'interval' (RUL e HI intervalares, com rho e intr_state) ou 'single' (apenas pontuais).
    rho     : rho estático (rho_mode='static') ou valor inicial da atualização online.
    rho_kw  : None (rho estático) ou dicionário com rho_mode ('static', 'grid', 'grad', 'conformal') e os parâmetros da
              atualização (rho_min, rho_max, sp_range, rho_n, rho_lam, rho_eta, rho_T, rho_beta, rho_kappa, rho_alpha),
              repassado ao AutoCloud_V5. O rho e o estado da atualização continuam de um rolamento para o seguinte.
    on_end  : função opcional on_end(teda, nome, i) chamada ao fim de cada rolamento (gráficos, registros).
    Retorna (teda, linhas): o modelo ao fim da sequência e uma lista com as métricas finais de cada rolamento.
    '''
    m, nI, nR, nO, mO, N1, N2, N3, τ, act = params
    teda, rows = None, []
    for i, (nm, RS_b, HI_b) in enumerate(data):
        sig = np.asarray(HI_b, dtype=float).ravel()
        X, Y, Z, kX = PrepareData(RS_b, HI_b, w, nI, nO, mO)
        ref, J, L = RefEoL(sig, eol, kX)
        if i == 0:
            teda = AutoCloud(m=m, nI=nI, nR=nR, nO=nO, mO=mO, ηS=[N1, N2, N3], tau=τ, eol=eol, ref=ref, act=act,
                             wtaG=wtaG, wtaP=wtaP, store=store, feedback=feedback, rho=rho, intr_state=intr_state,
                             **(rho_kw or {}))
        else:
            teda.restart(ref, state=state, x0=Y[0])
        teda.SimGranularRUL(J, L, X, Y, Z, mode_rul)
        rows.append(dict(ordem=i + 1, rolamento=nm, passos=len(teda.rulP), granulos=len(teda.c),
                         WAPE_RUL=float(teda.wape_RUL), WAPE_HI=float(teda.wape_HI),
                         V_HI=float(teda.V), cov_HI=float(teda.cov), sp_HI=float(teda.sp),
                         rho_ini=float(teda.rho_hist[0]) if len(teda.rho_hist) else float('nan'),
                         rho_fim=float(teda.rho_hist[-1]) if len(teda.rho_hist) else float('nan')))
        if on_end is not None: on_end(teda, nm, i + 1)
    return teda, rows


# =====================================================================================================================
# Otimização do rho (separada da otimização do modelo)
# =====================================================================================================================
_RHO_MODES = ('static', 'grid', 'grad', 'conformal')
_RHO_PRM = {'static': [], 'grid': ['rho_lam'], 'grad': ['rho_eta', 'rho_T', 'rho_beta'],
            'conformal': ['rho_kappa', 'rho_alpha']}
_RHO_COLS = ['rho', 'rho_lam', 'rho_eta', 'rho_T', 'rho_beta', 'rho_kappa', 'rho_alpha']


def _RhoParams(csv_path, row, params, w):
    # Parâmetros do modelo: diretamente (params, w) ou de uma linha do Optimization.csv de OptimizeSeq
    if params is not None:
        if w is None: raise ValueError('Informe w junto com params.')
        return list(params), int(w)
    if csv_path is None: raise ValueError('Informe csv_path (Optimization.csv de OptimizeSeq) ou params e w.')
    if row is None: return BestParamsW(csv_path)
    df = pd.read_csv(csv_path)
    r = df.loc[row] if row in df.index else df.iloc[int(row)]
    return _Parse10(r.iloc[1:11].values), int(r['w']) if 'w' in df.columns else 1


def _RhoPid(params, w, names_b, state, eol, feedback, wtaG, wtaP):
    # Identificador curto do modelo (parâmetros e configuração): os estudos do rho de modelos diferentes não se misturam
    import hashlib
    key = repr(([float(params[0])] + [p if isinstance(p, list) else int(p) for p in params[1:]], int(w), list(names_b),
                state, float(eol), feedback, bool(wtaG), bool(wtaP)))
    return hashlib.sha1(key.encode('utf8')).hexdigest()[:8]


def _RhoBase(params, data, w, state, eol, feedback, wtaG, wtaP):
    # Simulação única do modelo (predições pontuais): por rolamento, h previsto, h real, erro real |e_k| e erro estimado
    # e_k = |RLS| que forma o intervalo do passo. Nada disso depende do rho (AutoCloud_V5).
    base = []
    def on_end(t, nm, i):
        base.append(dict(nome=nm, hiP=t.hiP.copy(), hiR=t.hiR.copy(), e=t.eR.copy(), ehat=np.abs(t.eP).copy()))
    SimSeq(params, data, state=state, mode_rul='single', eol=eol, w=w, feedback=feedback, wtaG=wtaG, wtaP=wtaP,
           store=False, on_end=on_end)
    return base


def RhoReplay(base, rho, rho_kw):
    '''
    Repete a atualização online do rho do AutoCloud_V5 (UpdateRho, mesma ordem de UpdateRLS) sobre as séries gravadas
    por _RhoBase, sem simular o modelo: rho de cada passo, intervalo do HI [hiP -/+ rho_k*e_k] e qualidade por rolamento.
    O rho e o estado da atualização continuam de um rolamento para o seguinte (como em AutoCloud.restart).
    Retorna uma lista (um dict por rolamento): rho_hist, hiL, hiU, cov, sp, V (média(cov)*média(sp)), V_step (média(cov*sp)).
    '''
    ac = AutoCloud(m=1.0, rho=rho, **rho_kw)                 # só o estado do rho é usado
    out = []
    for b in base:
        n = len(b['e'])
        rh = np.empty(n)
        for k in range(n):
            rh[k] = ac.rho
            ac.UpdateRho(b['e'][k], b['ehat'][k])
        half = rh * b['ehat']
        hiL, hiU = b['hiP'] - half, b['hiP'] + half
        cov = ((b['hiR'] >= hiL) & (b['hiR'] <= hiU)).astype(float)
        sp = np.maximum(0.0, 1.0 - (hiU - hiL) / ac.sp_range)
        out.append(dict(nome=b['nome'], rho_hist=rh, hiL=hiL, hiU=hiU, cov=float(cov.mean()), sp=float(sp.mean()),
                        V=float(cov.mean() * sp.mean()), V_step=float((cov * sp).mean())))
    return out


def _RulQuality(t):
    # Qualidade do intervalo da RUL de um rolamento (oráculo: usa a RUL real): cobertura, especificidade com
    # Range = amplitude da RUL real do rolamento, V = média(cov)*média(sp)
    rR, rL, rU = np.asarray(t.rulR, float), np.asarray(t.rulL, float), np.asarray(t.rulU, float)
    Range = max(float(rR.max() - rR.min()), 1.0)
    cov = ((rR >= rL) & (rR <= rU)).astype(float)
    sp = np.maximum(0.0, 1.0 - (rU - rL) / Range)
    return dict(cov=float(cov.mean()), sp=float(sp.mean()), V=float(cov.mean() * sp.mean()),
                V_step=float((cov * sp).mean()), rho_hist=np.asarray(t.rho_hist, float).copy())


def OptimizeRho(
        data=None, csv_path=None, row=None, params=None, w=None, folder=None,
        Vtarget='HI', rho_modes=('grid',), agg='mean',
        state='keep', eol=0.3, feedback='symmetric', wtaG=False, wtaP=False, intr_state='carry',
        rhoS=None, lamS=None, etaS=None, TS=None, betaS=None, kappaS=None, alphaS=None,
        rho_min=1e-3, rho_max=10.0, rho_n=50, sp_range=1.0,
        OptSampler='tpe', OptSeed=1, n_study=1, n_trials=500, patience=200, timeout=3600,
        OptReset=False, n_best=10):
    '''
    Otimiza os parâmetros do rho (AutoCloud_V5) com o modelo FIXO, encontrado antes por OptimizeSeq.
    Maximiza V = média(cov)*média(sp) do intervalo (Princípio da Granularidade Justificável), agregado sobre os
    rolamentos (agg: 'mean' ou 'sum'); o Optuna minimiza 1 - V (mean) ou -V (sum).

    Modelo:
        csv_path: Optimization.csv de OptimizeSeq (<instância>/mono/Optimization.csv); row: índice (rótulo do DataFrame)
                  ou posição da linha; None = melhor linha (BestParamsW). Ou params = [m, nI, nR, nO, mO, N1, N2, N3, τ, act]
                  e w diretamente, com folder = pasta da instância.
        data, state, eol, feedback, wtaG, wtaP: os mesmos de OptimizeSeq (sequência de rolamentos e restart).
    Vtarget:
        'HI' : intervalo do HI um passo à frente (causal: o valor real é conhecido no passo seguinte). O modelo é simulado
               UMA vez (predições pontuais; o rho não altera predições, treino das RNNs nem o modelo RLS do erro) e cada
               trial apenas repete a atualização do rho sobre as séries gravadas (RhoReplay): milissegundos por trial.
               Especificidade com Range = sp_range (1 = faixa do HI).
        'RUL': intervalo da RUL (RULInterv, com intr_state): ORÁCULO, usa a RUL real (EoL conhecido só ao fim da vida),
               apenas para análise; cada trial simula toda a sequência com RUL intervalar (lento).
    rho_modes: modos de AutoCloud_V5 otimizados, um estudo por modo ('static', 'grid', 'grad', 'conformal').
        Faixas de busca [min, max] (log: escala logarítmica):
            rhoS   : rho estático ou inicial (log; padrão [rho_min, rho_max])
            lamS   : 'grid', fator de esquecimento rho_lam (padrão [0.8, 1.0])
            etaS, TS, betaS: 'grad', rho_eta (log, [1e-3, 1]), rho_T (log, [0.05, 2]), rho_beta ([0, 0.95])
            kappaS, alphaS : 'conformal', rho_kappa (log, [1e-3, 0.5]), rho_alpha ([0.01, 0.5])
        Fixos: rho_min, rho_max (limites do rho nos modos dinâmicos), rho_n (grade de 'grid'), sp_range.
    Sessões: n_study sessões de n_trials trials por modo; patience (trials sem melhora), timeout (s), OptSampler, OptSeed,
        OptReset (apaga os estudos do rho deste modelo e alvo).
    Banco: <instância>/optuna.db (mesmo de OptimizeSeq), estudos rho_<Vtarget>_<agg>_<modo>_<pid>_s<i>, com pid = id do
        modelo (parâmetros, w, rolamentos e configuração): modelos diferentes não se misturam.
    Resultado: <instância>/rho/Optimization.csv (Vtarget='HI') ou rho/Optimization_RUL.csv (Vtarget='RUL'), RhoTable:
        n_best maiores V de cada modo já otimizado para este modelo (todos os estudos do banco, não só os desta chamada),
        com os parâmetros do rho, V, cov, sp e rho final de cada rolamento, e os parâmetros do modelo. Retorna o DataFrame.
        BestRho lê a melhor linha.
    '''
    if not data: raise ValueError('Informe data com pelo menos um rolamento: [(nome, RS, HI), ...].')
    if Vtarget not in ('HI', 'RUL'): raise ValueError("Vtarget deve ser 'HI' ou 'RUL'.")
    if agg not in ('mean', 'sum'): raise ValueError("agg deve ser 'mean' ou 'sum'.")
    rho_modes = [rho_modes] if isinstance(rho_modes, str) else list(rho_modes)
    bad = [m_ for m_ in rho_modes if m_ not in _RHO_MODES]
    if bad: raise ValueError(f'rho_modes inválidos: {bad} (use {_RHO_MODES}).')
    rhoS = rhoS or [rho_min, rho_max]
    lamS, etaS, TS, betaS = lamS or [0.8, 1.0], etaS or [1e-3, 1.0], TS or [0.05, 2.0], betaS or [0.0, 0.95]
    kappaS, alphaS = kappaS or [1e-3, 0.5], alphaS or [0.01, 0.5]
    if not 0 < rhoS[0] <= rhoS[1]: raise ValueError(f'rhoS inválido: {rhoS} (requer 0 < min <= max).')

    params, w = _RhoParams(csv_path, row, params, w)
    names_b = [str(d[0]) for d in data]
    if folder is None:
        if csv_path is None: raise ValueError('Informe folder (pasta da instância) quando params é passado diretamente.')
        folder = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))     # <instância>/mono/Optimization.csv
    os.makedirs(os.path.join(folder, 'rho'), exist_ok=True)
    db_path = os.path.abspath(os.path.join(folder, 'optuna.db'))
    storage = _Storage(db_path)
    pid = _RhoPid(params, w, names_b, state, eol, feedback, wtaG, wtaP)
    fixed = dict(rho_min=float(rho_min), rho_max=float(rho_max), rho_n=int(rho_n), sp_range=float(sp_range))
    Agg = (lambda v: float(np.mean(v))) if agg == 'mean' else (lambda v: float(np.sum(v)))
    print(f'OptimizeRho | Vtarget={Vtarget} | modos={rho_modes} | modelo {pid}: params={params} w={w} | '
          f'{" -> ".join(names_b)} | banco {db_path}')
    if Vtarget == 'RUL':
        print('  Aviso: Vtarget=RUL usa a RUL real (oráculo, apenas análise) e simula a sequência inteira a cada trial.')
        base = None
    else:
        t0 = time.time()
        base = _RhoBase(params, data, w, state, eol, feedback, wtaG, wtaP)
        print(f'  simulação do modelo (uma vez): {time.time() - t0:.1f} s | passos por rolamento: '
              f'{[len(b["e"]) for b in base]}')

    def suggest(trial, mode):
        rho = trial.suggest_float('rho', rhoS[0], rhoS[1], log=True)
        kw = dict(rho_mode=mode, **fixed)
        if mode == 'grid':
            kw['rho_lam'] = trial.suggest_float('rho_lam', lamS[0], lamS[1])
        elif mode == 'grad':
            kw['rho_eta'] = trial.suggest_float('rho_eta', etaS[0], etaS[1], log=True)
            kw['rho_T'] = trial.suggest_float('rho_T', TS[0], TS[1], log=True)
            kw['rho_beta'] = trial.suggest_float('rho_beta', betaS[0], betaS[1])
        elif mode == 'conformal':
            kw['rho_kappa'] = trial.suggest_float('rho_kappa', kappaS[0], kappaS[1], log=True)
            kw['rho_alpha'] = trial.suggest_float('rho_alpha', alphaS[0], alphaS[1])
        return rho, kw

    def evaluate(rho, kw):
        if Vtarget == 'HI': return RhoReplay(base, rho, kw)
        res = []
        def on_end(t, nm, i): res.append(dict(nome=nm, **_RulQuality(t)))
        SimSeq(params, data, state=state, mode_rul='interval', rho=rho, intr_state=intr_state, eol=eol, w=w,
               feedback=feedback, wtaG=wtaG, wtaP=wtaP, store=False, on_end=on_end, rho_kw=kw)
        return res

    summary = []
    for mi, mode in enumerate(rho_modes):
        base_name = f'rho_{Vtarget}_{agg}_{mode}_{pid}'
        def objective(trial, mode=mode):
            rho, kw = suggest(trial, mode)
            res = evaluate(rho, kw)
            for r in res:
                for k in ('V', 'cov', 'sp', 'V_step'): trial.set_user_attr(f'{k}_{r["nome"]}', r[k])
                trial.set_user_attr(f'rho_fim_{r["nome"]}', float(r['rho_hist'][-1]) if len(r['rho_hist']) else float('nan'))
            V = Agg([r['V'] for r in res])
            trial.set_user_attr('V', V)
            return 1.0 - V if agg == 'mean' else -V
        print(f'\n===== rho_mode: {mode} ({n_study} sessão(ões) x {n_trials} trials) | Vtarget={Vtarget} =====')
        for i in range(n_study):
            study_name = f'{base_name}_s{i + 1}'
            if OptReset and study_name in optuna.get_all_study_names(storage):
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f'  {study_name}: estudo anterior apagado (OptReset=True).')
            seed, _ = _Seed(OptSeed, mi, i, study_name, storage)
            study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='minimize',
                                        sampler=_Sampler(OptSampler, seed))
            n_prev, done = len(study.trials), _Complete(study)
            callback = EarlyStoppingCallback(patience=patience, context={
                'nL': mode, 'session': i + 1, 'n_study': n_study, 'study_name': study_name, 'config': f'rho {pid}',
                'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout})
            if done:
                best_prev = min(done, key=lambda t: t.value)
                callback.best_score, callback.best_number = best_prev.value, best_prev.number
                callback.best_trial_info = (f'Trial #{best_prev.number:03d} (sessões anteriores) | V: '
                                            f'{best_prev.user_attrs.get("V", float("nan")):.4f} | Params: {best_prev.params}')
            study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout,
                           callbacks=[callback])
            done = _Complete(study)
            summary.append((study_name, n_prev, len(study.trials),
                            max(t.user_attrs.get('V', float('nan')) for t in done) if done else float('nan')))

    print('\nResumo das sessões do rho (banco: ' + db_path + ')')
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor V {best:.4f}')
    return RhoTable(db_path, names_b, params, w, Vtarget=Vtarget, agg=agg, rho_modes=_RHO_MODES, n_best=n_best,
                    state=state, eol=eol, feedback=feedback, wtaG=wtaG, wtaP=wtaP, fixed=fixed)


def RhoTable(db_path, names_b, params, w, Vtarget='HI', agg='mean', rho_modes=_RHO_MODES, n_best=10, out_path=None,
             state='keep', eol=0.3, feedback='symmetric', wtaG=False, wtaP=False, fixed=None):
    '''
    Reconstrói <instância>/rho/Optimization.csv (Vtarget='HI'; 'RUL': rho/Optimization_RUL.csv) a partir do banco
    (estudos do rho do modelo params/w e do alvo Vtarget):
    n_best maiores V por modo. Colunas: V_<agg>, rho_mode, rho, rho_lam, rho_eta, rho_T, rho_beta, rho_kappa, rho_alpha
    (vazias quando o modo não as usa), rho_min, rho_max, rho_n, sp_range, por rolamento V_<b>, cov_<b>, sp_<b>,
    rho_fim_<b> e, ao fim, Vtarget, os parâmetros do modelo (params) e w.
    '''
    storage = _Storage(db_path)
    pid = _RhoPid(params, w, names_b, state, eol, feedback, wtaG, wtaP)
    fixed = fixed or dict(rho_min=1e-3, rho_max=10.0, rho_n=50, sp_range=1.0)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(db_path), 'rho', 'Optimization.csv' if Vtarget == 'HI' else f'Optimization_{Vtarget}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seg = [c for nm in names_b for c in (f'V_{nm}', f'cov_{nm}', f'sp_{nm}', f'rho_fim_{nm}')]
    cols = [f'V_{agg}', 'rho_mode'] + _RHO_COLS + list(fixed) + seg + ['Vtarget', 'params', 'w']
    rows = []
    for mode in rho_modes:
        prefix = f'rho_{Vtarget}_{agg}_{mode}_{pid}_s'
        for name in optuna.get_all_study_names(storage):
            if not name.startswith(prefix): continue
            for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
                rows.append([t.user_attrs.get('V', np.nan), mode] + [t.params.get(c, np.nan) for c in _RHO_COLS]
                            + list(fixed.values()) + [t.user_attrs.get(c, np.nan) for c in seg]
                            + [Vtarget, str(list(params)), int(w)])
    if not rows:
        print('Nenhum trial do rho concluído: rho/Optimization.csv não foi gerado.')
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols).sort_values(f'V_{agg}', ascending=False)
    df = df.drop_duplicates(subset=['rho_mode'] + _RHO_COLS).groupby('rho_mode', group_keys=False).head(n_best)
    df = df.sort_values(f'V_{agg}', ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f'\nrho/Optimization.csv: {n_best} maiores V por modo ({out_path})')
    for mode, g in df.groupby('rho_mode', sort=False):
        print(f'  {mode}: melhor V {g[f"V_{agg}"].max():.4f} | ' + ' | '.join(
            f'{c}={g.iloc[0][c]:.4g}' for c in ['rho'] + _RHO_PRM[mode]))
    return df


def BestRho(csv_path, mode=None):
    '''
    Melhor linha (maior V, 1ª coluna) do rho/Optimization.csv de OptimizeRho; mode = None (todos os modos) ou um modo.
    Retorna (rho, rho_kw), prontos para SimSeq(..., rho=rho, rho_kw=rho_kw) ou AutoCloud(..., rho=rho, **rho_kw).
    '''
    df = pd.read_csv(csv_path)
    if mode is not None:
        df = df[df['rho_mode'] == mode]
        if df.empty: raise ValueError(f'Nenhuma linha com rho_mode={mode} em {csv_path}.')
    r = df.sort_values(df.columns[0], ascending=False).iloc[0]
    kw = dict(rho_mode=r['rho_mode'], rho_min=float(r['rho_min']), rho_max=float(r['rho_max']),
              rho_n=int(r['rho_n']), sp_range=float(r['sp_range']))
    kw.update({c: float(r[c]) for c in _RHO_PRM[r['rho_mode']]})
    return float(r['rho']), kw
