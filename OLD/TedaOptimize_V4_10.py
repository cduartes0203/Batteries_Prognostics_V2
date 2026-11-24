'''
Otimização (Optuna) do AutoCloud_V4_6 com rho estático.

    Optimize    : um rolamento; minimiza WAPE_RUL + WAPE_HI (predições pontuais).
    OptimizeSeq : sequência de rolamentos assimilada pelo mesmo modelo (AutoCloud.restart entre rolamentos);
                  minimiza a agregação (agg) sobre os rolamentos do erro combinado (norm, obj) de cada rolamento.
    BestParams  : melhor linha de um Optimization.csv no formato de ParseParams.
    BestParamsW : idem, com a janela w da mesma linha: (params, w).
    SimSeq      : simulação final de uma sequência de rolamentos com parâmetros fixos (mesmo procedimento de OptimizeSeq).

    RefWAPE     : WAPE_RUL e WAPE_HI de preditores de referência (normalização norm='ref').
    ObjPrefix   : prefixo dos estudos no banco para uma configuração do objetivo (norm, obj, agg, ...).

Nas duas otimizações, nLS = [min, max] executa n_study sessões de n_trials trials para CADA nº de camadas.
Uso no notebook (na pasta principal): from Functions.TedaOptimize_V4_10 import *

Mudanças em relação à V4_9:
    - objetivo de OptimizeSeq em três escolhas independentes (antes: agg = 'mean' | 'sum' | 'norm'):
        norm : normalização de cada rolamento, r = WAPE_RUL / a_R e h = WAPE_HI / a_H
               'none' (a = 1, WAPE brutos), 'lim' (a = lims = (0.3, 0.2): fração do erro admissível) ou
               'ref' (a = WAPE dos preditores de referência, RefWAPE, como agg='norm' da V4_9);
        obj  : combinação de r e h em um rolamento
               'sum'  -> alpha*r + (1 - alpha)*h (compensação linear);
               'prod' -> r*h (V4_8; h ~ 0 anula r: uma métrica esconde a outra, mantido para reprodução);
               'cheb' -> max(r, h) + eps*(r + h) (Chebyshev aumentado, padrão): a pior métrica define o erro, nenhuma
                         esconde a outra; o termo eps desempata e garante mínimo Pareto-ótimo;
        agg  : agregação sobre os rolamentos: 'mean', 'sum' ou 'max' (pior rolamento).
      lim_prune: poda o trial se WAPE_RUL > lims[0] ou WAPE_HI > lims[1] ao fim de um rolamento (limiares da V4_8);
      None = ativo somente com norm='lim' (r, h em [0, 1]).
      Equivalências: V4_9 agg='norm' = (norm='ref', obj='sum', alpha=0.5, agg='mean'); V4_9 agg='mean'/'sum' =
      (norm='none', obj='sum', alpha=0.5, agg='mean'/'sum') com objetivo multiplicado por 1/2 (mesma ordenação);
      V4_8 = (norm='lim', obj='prod', agg='mean', lim_prune=True).
    - prefixo dos estudos (ObjPrefix) codifica toda a configuração, ex.: seq_ref10_cheb0.05_mean_,
      seq_lim0.3-0.2_prod_mean_cut0.3-0.2_: configurações diferentes não se misturam no mesmo optuna.db;
    - Optimization.csv: coluna do objetivo OBJ_<obj>_<agg>; por rolamento WAPE_RUL_<b>, WAPE_HI_<b> (brutos), R_<b>, H_<b>
      (normalizados), E_<b> (erro combinado do rolamento) e, com norm='ref', REF_RUL_<b> e REF_HI_<b>.

Mudanças da V4_9 em relação à V4_8:
    - OptimizeSeq(agg='norm'): objetivo normalizado por preditores de referência (skill score), para que WAPE_RUL e
      WAPE_HI tenham escalas comparáveis (na V4_8, WAPE_HI ~ 20-30x menor que WAPE_RUL: o objetivo era quase só RUL).
      Por rolamento: e = (WAPE_RUL / WAPE_RUL_ref + WAPE_HI / WAPE_HI_ref) / 2; objetivo = média sobre os rolamentos.
      e = 1: desempenho igual ao das referências; e < 1: melhor. Referências (RefWAPE), causais e calculadas nos
      MESMOS passos e pesos (k - k_seg)^2 do AutoCloud em cada trial:
        HI : persistência, h_{k+1} = h_k;
        RUL: tendência linear (mínimos quadrados) das ref_win últimas amostras do HI até k (ref_win = 0: todo o
             histórico do rolamento), extrapolada até o EoL; RUL de referência limitada a [1, L], como a do modelo.
      Os estudos usam o prefixo seqnormW<ref_win>_ (não se misturam com seqmean_ / seqsum_ nem entre ref_win
      diferentes). Optimization.csv mantém WAPE_RUL_<b> e WAPE_HI_<b> brutos e acrescenta REF_RUL_<b> e REF_HI_<b>.
      Limiares de poda (OptPrune) continuam sobre os WAPE brutos.

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
from Functions.AutoCloud_V4_6 import AutoCloud, RefEoL
from Functions.Utils import PrepareData, ParseParams
from Functions.Utils_OPT_R2 import EarlyStoppingCallback, SelSampler

__all__ = ['Optimize', 'OptimizeSeq', 'SeqTable', 'BestParams', 'BestParamsW', 'SimSeq', 'RefWAPE', 'ObjPrefix']

_AGGS = ('mean', 'sum', 'max')
_NORMS = ('none', 'lim', 'ref')
_OBJS = ('sum', 'prod', 'cheb')
_AGGF = {'mean': np.mean, 'sum': np.sum, 'max': np.max}


def _Spec(agg='mean', norm='ref', obj='cheb', lims=(0.3, 0.2), alpha=0.5, eps=0.05, ref_win=10, lim_prune=None):
    '''Valida e completa a configuração do objetivo de OptimizeSeq / SeqTable (ver OptimizeSeq). Retorna um dict.'''
    if agg not in _AGGS: raise ValueError(f'agg deve ser um de {_AGGS}.')
    if norm not in _NORMS: raise ValueError(f'norm deve ser um de {_NORMS}.')
    if obj not in _OBJS: raise ValueError(f'obj deve ser um de {_OBJS}.')
    lims = tuple(float(v) for v in lims)
    if len(lims) != 2 or min(lims) <= 0: raise ValueError('lims = (lim_RUL, lim_HI), ambos > 0.')
    if not 0 <= alpha <= 1: raise ValueError('alpha deve estar em [0, 1].')
    if eps < 0: raise ValueError('eps deve ser >= 0.')
    if int(ref_win) < 0: raise ValueError('ref_win deve ser >= 0.')
    if lim_prune is None: lim_prune = (norm == 'lim')
    return dict(agg=agg, norm=norm, obj=obj, lims=lims, alpha=float(alpha), eps=float(eps), ref_win=int(ref_win),
                lim_prune=bool(lim_prune))


def ObjPrefix(**kw):
    '''Prefixo dos estudos no banco para uma configuração do objetivo (argumentos de _Spec): configurações diferentes
    não se misturam. Ex.: seq_ref10_cheb0.05_mean_ ; seq_lim0.3-0.2_prod_mean_cut0.3-0.2_'''
    s = _Spec(**kw)
    lim = f"{s['lims'][0]:g}-{s['lims'][1]:g}"
    nt = {'none': 'none', 'lim': f'lim{lim}', 'ref': f"ref{s['ref_win']}"}[s['norm']]
    ot = {'sum': f"sum{s['alpha']:g}", 'prod': 'prod', 'cheb': f"cheb{s['eps']:g}"}[s['obj']]
    return f"seq_{nt}_{ot}_{s['agg']}" + (f'_cut{lim}' if s['lim_prune'] else '') + '_'


def _ObjName(s):
    return f"OBJ_{s['obj']}_{s['agg']}"


def _Combine(wR, wH, ref, s):
    '''Erro de um rolamento: normalização (r, h) e combinação (s = _Spec). ref = (WAPE_RUL_ref, WAPE_HI_ref) com
    norm='ref'. Retorna (e, r, h).'''
    if s['norm'] == 'none': r, h = wR, wH
    elif s['norm'] == 'lim': r, h = wR / s['lims'][0], wH / s['lims'][1]
    else: r, h = wR / max(ref[0], 1e-12), wH / max(ref[1], 1e-12)
    if s['obj'] == 'sum': e = s['alpha'] * r + (1 - s['alpha']) * h
    elif s['obj'] == 'prod': e = r * h
    else: e = max(r, h) + s['eps'] * (r + h)
    return float(e), float(r), float(h)


def RefWAPE(sig, eol, kX, J, L, ref_win=10):
    '''
    WAPE de preditores de referência nos mesmos passos e pesos do AutoCloud (normalização norm='ref' de OptimizeSeq).
    sig: HI do rolamento; kX, J, L: saídas de PrepareData / RefEoL do trial (passos avaliados j = 1, ..., J-1, instante
    k = kX[j], peso (j + 2)^2 = (k - k_seg)^2, RUL real iEoL - k).
        HI : persistência: predição de h_{k+1} = h_k.
        RUL: reta de mínimos quadrados sobre h_{k-ref_win+1..k} (ref_win = 0: h_0..h_k), extrapolada a partir do valor
             ajustado em k: RUL = 1º n >= 1 com reta(k + n) <= eol; inclinação >= 0 -> L; limitada a [1, L].
    Ambas usam apenas o HI até k (causais). Retorna (WAPE_RUL_ref, WAPE_HI_ref).
    '''
    sig = np.asarray(sig, dtype=float).ravel()
    kX = np.asarray(kX, dtype=int)
    iEoL = int(np.argmax(sig <= eol))
    j = np.arange(1, J)
    if len(j) == 0: raise ValueError('RefWAPE: nenhum passo avaliado (J <= 1).')
    k, t = kX[j], (j + 2.0) ** 2
    wH = np.sum(np.abs(sig[k + 1] - sig[k]) * t) / np.sum(np.abs(sig[k + 1]) * t)
    rulR, rulP = iEoL - k, np.empty(len(k))
    for i, kk in enumerate(k):
        lo = 0 if ref_win <= 0 else max(0, kk - int(ref_win) + 1)
        if kk - lo < 1: rulP[i] = L; continue        # um único ponto: sem tendência
        b, a = np.polyfit(np.arange(lo, kk + 1), sig[lo:kk + 1], 1)
        hk = a + b * kk
        if hk <= eol: n = 1
        elif b >= 0: n = L
        else: n = int(np.ceil((hk - eol) / (-b)))
        rulP[i] = min(max(n, 1), L)
    wR = np.sum(np.abs(rulR - rulP) * t) / np.sum(np.abs(rulR) * t)
    return float(wR), float(wH)


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
    return prev.value


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
        try: best = study.best_trial
        except ValueError: return                    # nenhum trial concluído ainda
        if best.number != t.number and best.value < callback.best_score:
            callback.best_score, callback.best_number = best.value, best.number
            callback.best_trial_info = f'Trial #{best.number:03d} (outro worker) | Score: {best.value:.4f} | Params: {best.params}'
            callback.trials_without_improvement = 0
    return cb

def Optimize(
        RS=None, HI=None,
        FileName=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,actS=None,
        feedback='symmetric',wtaG=False,wtaP=False,
        eol=0.3,w=1,wS=None,
        OptReset=False,OptWarm=False,n_best=10,
        folder='Optimization'):

    '''
        Otimização mono-objetivo dos hiperparâmetros do AutoCloud_V4_6 [m, nI, nR, nO, mO, N1, N2, N3, τ, act] e da
            janela w do DSI: minimiza WAPE_RUL + WAPE_HI de um rolamento.

        RS, HI : DSI (uma amostra por linha) e HI (uma coluna, ex.: PC1) do rolamento.
        eol    : limiar de EoL do HI.
        w, wS  : janela do DSI em X_k (PrepareData). wS = [min, max] -> w otimizado (inteiro);
                 wS = None -> w fixo (padrão w = 1).
        nO / mO: valores futuros / passados (incluindo h_k) de Z_k; h_{k+1} = Z_k[mO].
        Faixas de busca (xxS = [min, max]): mS, nIS, nRS (neurônios por camada), nOS, mOS,
            N1S, N2S, N3S (taxas de aprendizado, expoentes), τS, actS.
        nLS = [min, max]: nº de camadas. Cada valor nL = min, ..., max é otimizado separadamente, com n_study sessões
            de n_trials trials cada (ex.: nLS = [1, 3] -> n_trials para 1 camada, n_trials para 2 e n_trials para 3).
        OptSampler: None, Auto, GP, NSGAII, Random, TPE (SelSampler).

        feedback, wtaG, wtaP: configuração do modelo. Resultados em <folder>/<FileName>_<feedback>_G<wtaG>P<wtaP>/.
        Armazenamento persistente (SQLite, <folder>/<...>/optuna.db):
            para cada nL, cada uma das n_study sessões é um estudo independente chamado obj1_L<nL>-<nL>_s<i>,
            retomado a cada execução (n_trials = nº de trials NOVOS por execução, por sessão e por nL).
        OptReset: apaga e recria os estudos destas sessões (use após mudar o objetivo/código).
        OptWarm : enfileira o melhor trial das outras sessões da mesma configuração antes de otimizar.
        OptSeed : semente base; a sessão i do nL usa _Seed(OptSeed, nL, i, n_prev), com n_prev = trials já
                  armazenados no estudo (sessões independentes; retomar com o mesmo OptSeed não repete sugestões).
        n_best  : linhas mantidas por nº de camadas (nL) em mono/Optimization.csv, reconstruída com TODOS os trials
                  do banco (n_best menores erros de conjuntos de parâmetros distintos por nL; pior -> melhor em cada nL).

        Objetivo: mesma sequência de SimGranularRUL(mode_rul='single'), por passo k:
            run -> RUL_Prediction(mode='single') -> Adapt.
            Erro = teda.wape_RUL + teda.wape_HI (WAPE ponderados por k^2 do próprio AutoCloud; HI do passo k contra o real
            do passo k). Apenas predições pontuais: rho, intr_state e o modelo RLS do erro não afetam o objetivo
            (WAPE_HI2 e UpdateRLS são omitidos por isso).
    '''
    if RS is None or HI is None: raise ValueError('Informe RS e HI.')
    sig = np.asarray(HI, dtype=float)
    if sig.ndim > 1 and sig.shape[1] != 1: raise ValueError('HI deve ter uma única coluna.')
    sig = sig.ravel()
    FileName = f'{FileName}_{feedback}_G{int(wtaG)}P{int(wtaP)}'

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

    # Limita nO para que o fluxo sempre alcance o EoL: a RUL é avaliada até a última amostra antes
    # do EoL em todos os trials (evita favorecer nO grande, que encurta o trecho avaliado)
    nO_max = len(sig) - int(np.argmax(sig <= eol))   # último instante k = K-1-nO >= iEoL-1
    if nOS[1] > nO_max:
        nOS = [nOS[0], nO_max]
        print(f'nO limitado a {nOS} para que o fluxo alcance o EoL.')
    if nOS[0] > nOS[1]: raise ValueError(f'nOS inválido: o limite para alcançar o EoL é nO <= {nO_max}.')

    prm_names = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation', 'w']
    names = ['WAPE_RUL+WAPE_HI'] + prm_names
    study_dir = os.path.join(folder, FileName, 'mono')
    os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')

    # Armazenamento persistente: um banco SQLite por configuração; cada sessão é um estudo independente e retomável
    db_path = os.path.abspath(os.path.join(folder, FileName, 'optuna.db'))
    storage = _Storage(db_path)
    prefix = 'obj1_'
    summary = []
    nL_cur = None                                   # nº de camadas da rodada atual (laço abaixo)
    seen_cur = {}                                   # combinações já testadas no estudo atual (_Seen; atualizado por _Track)

    def objective(trial):
        m = trial.suggest_float('m', mS[0], mS[1],step=0.25)
        nI = trial.suggest_int('nI', nIS[0], nIS[1])
        n_layers = trial.suggest_int('n_layers', nL_cur, nL_cur)   # fixo na rodada (registrado no banco)
        nR = [trial.suggest_int(f'nR_layer_{l}', nRS[0], nRS[1]) for l in range(n_layers)]
        nO = trial.suggest_int('nO', nOS[0], nOS[1])
        N1 = trial.suggest_int('N1', N1S[0], N1S[1])
        N2 = trial.suggest_int('N2', N2S[0], N2S[1])
        N3 = trial.suggest_int('N3', N3S[0], N3S[1])
        τ = trial.suggest_int('τ', τS[0], τS[1])
        act = trial.suggest_int('act', actS[0], actS[1])
        mO = trial.suggest_int('mO', mOS[0], mOS[1])
        w_ = trial.suggest_int('w', wS[0], wS[1])          # janela do DSI (fixa quando wS = None)
        if mO > nI: raise TrialPruned()
        prev = seen_cur.get(_Key(trial.params))
        if prev is not None: return _Repeat(trial, prev)   # combinação já testada: reutiliza o resultado, sem simular

        X, Y, Z, kX = PrepareData(RS, HI, w_, nI, nO, mO)
        try: ref, J, L = RefEoL(sig, eol, kX, strict=True)
        except ValueError: raise TrialPruned()   # o fluxo não alcança o EoL (não deve ocorrer com nOS limitado)

        teda = AutoCloud(m=m,nI=nI,nR=nR,nO=nO,mO=mO,ηS=[N1,N2,N3],act=act,
                         tau=τ,eol=eol,ref=ref,wtaG=wtaG,wtaP=wtaP,feedback=feedback)
        for j,_ in enumerate(X[:J]):
            teda.run(X[j])
            teda.RUL_Prediction(Y[j],mode='single',lim=L)   # WAPE_RUL do passo
            teda.Adapt(Y[j],Z[j])                            # WAPE_HI do passo e treino das RNNs
            wR, wH = teda.wape_RUL, teda.wape_HI

            if not np.isfinite(teda.hiP[-1]): raise TrialPruned()
            if not OptPrune: continue

            if j >=40 and j%5==0:
                if wR > 0.6: raise TrialPruned()
                if wH > 0.3: raise TrialPruned()

            if j%35==0:
                trial.report(wR+wH, j)
                if trial.should_prune():
                    raise TrialPruned()

        return teda.wape_RUL + teda.wape_HI

    for nL in range(nLS[0], nLS[1] + 1):
        nL_cur = nL
        base_name = f'{prefix}L{nL}-{nL}'
        print(f'\n===== nº de camadas: {nL} ({n_study} sessão(ões) x {n_trials} trials) =====')
        for i in range(n_study):
            study_name = f'{base_name}_s{i+1}'
            print(f'iteration: {i+1} | study: {study_name}')
            if OptReset and study_name in optuna.get_all_study_names(storage):
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f'  {study_name}: estudo anterior apagado (OptReset=True).')

            seed, _ = _Seed(OptSeed, nL, i, study_name, storage)   # depende de n_prev: retomada sem repetir sugestões
            study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),sampler=SelSampler(mode=OptSampler,seed=seed))
            n_prev, done = len(study.trials), _Complete(study)
            seen_cur.clear(); seen_cur.update(_Seen(study))   # inclui as execuções anteriores (retomada)

            if OptWarm:
                # Começa pelo melhor trial das outras sessões desta configuração
                others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
                cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
                if cand:
                    best = min(cand, key=lambda t: t.value)
                    study.enqueue_trial(best.params, skip_if_exists=True)
                    print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {best.value:.4f}).')

            # Parada antecipada conta trials sem superar o melhor resultado já armazenado
            callback = EarlyStoppingCallback(patience=patience, context={
                'nL': [nL, nL], 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
                'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout,
                'storage': storage, 'prefix': prefix})   # melhor trial por nL (todas as sessões/fases)
            if done:
                best_prev = min(done, key=lambda t: t.value)
                callback.best_score = best_prev.value
                callback.best_number = best_prev.number
                callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {best_prev.value:.4f} | Params: {best_prev.params}'
            print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else '')
                  + f' | semente do sampler: {seed}')
            study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout,
                           callbacks=[_Track(seen_cur), callback])

            done = _Complete(study)
            summary.append((study_name, n_prev, len(study.trials), min(t.value for t in done) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor {best:.4f}')

    # Tabela reconstruída do banco: todos os trials concluídos (todas as sessões, fases e nº de camadas),
    # mantendo para cada nL os n_best menores erros
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['act'],p.get('w', w)]
            rows.append([t.value] + prm + [p['n_layers']])
    if not rows:
        print('Nenhum trial concluído: Optimization.csv não foi gerado.')
        return pd.DataFrame(columns=names)
    df_all = pd.DataFrame(rows, columns=names + ['_nL'])
    # Mesmo conjunto de parâmetros conta uma vez (mantém o menor erro)
    df_all = df_all.sort_values(names[0]).drop_duplicates(subset=prm_names)
    df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best)
    df_stdy = df_stdy.sort_values(['_nL', names[0]], ascending=[True, False])

    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas ({out_path})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp[names[0]].min():.4f} | pior mantido {grp[names[0]].max():.4f}')

    df_stdy = df_stdy.drop(columns=['_nL']).reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


def OptimizeSeq(
        data=None,
        FileName=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,actS=None,
        feedback='symmetric',wtaG=False,wtaP=False,
        eol=0.3,w=1,wS=None,agg='mean',state='keep',ref_win=10,
        norm='ref',obj='cheb',lims=(0.3,0.2),alpha=0.5,eps=0.05,lim_prune=None,
        OptReset=False,OptWarm=False,n_best=10,
        folder='Optimization',write_table=True,worker=0,n_workers=1):

    '''
        Versão de Optimize para uma SEQUÊNCIA de rolamentos assimilada pelo mesmo modelo (AutoCloud_V4_6.restart):
            o 1º rolamento é simulado até o EoL; antes de cada rolamento seguinte, teda.restart(ref) zera RUL/HI/WAPE e
            mantém grânulos, RNNs e modelo do erro, e a simulação prossegue (mesmo procedimento da última célula).

        data: lista ordenada de (nome, RS, HI), ex.: [('B1_1', RS11, HI11['PC1']), ('B1_2', RS12, HI12['PC1'])].
        Objetivo: para cada rolamento b, WAPE_RUL e WAPE_HI finais -> (r, h) normalizados (norm) -> erro e_b (obj);
        objetivo do trial = agregação de e_b sobre os rolamentos (agg).
        norm : 'none' -> r = WAPE_RUL, h = WAPE_HI;
               'lim'  -> r = WAPE_RUL/lims[0], h = WAPE_HI/lims[1] (fração do erro admissível);
               'ref'  -> r = WAPE_RUL/WAPE_RUL_ref, h = WAPE_HI/WAPE_HI_ref, referências de RefWAPE (persistência para o
                         HI, tendência linear para a RUL), padrão.
        obj  : 'sum'  -> e = alpha*r + (1 - alpha)*h;
               'prod' -> e = r*h (uma métrica pode esconder a outra: h ~ 0 anula r);
               'cheb' -> e = max(r, h) + eps*(r + h) (Chebyshev aumentado, padrão).
        agg  : 'mean' (padrão), 'sum' ou 'max' (pior rolamento) de e_b sobre os rolamentos.
        lims : (lim_RUL, lim_HI) de norm='lim' e de lim_prune. alpha: peso da RUL em obj='sum'. eps: desempate de obj='cheb'.
        lim_prune: True -> poda o trial se, ao fim de um rolamento, WAPE_RUL > lims[0] ou WAPE_HI > lims[1];
               None (padrão) -> True somente com norm='lim'.
        ref_win: janela (nº de amostras) da tendência linear da RUL de referência em norm='ref' (0 = todo o histórico).
        Os estudos usam o prefixo ObjPrefix(...) (toda a configuração do objetivo).
        state: estado recorrente das RNNs no início de cada rolamento seguinte (teda.restart):
              'keep' (estado do fim do rolamento anterior), 'reset' (estado inicial) ou 'fixed' (equilíbrio para a
              primeira janela Y[0] do novo rolamento). Pastas: ..._restart_... ('keep') ou ..._restart-<state>_...
        Cada rolamento usa o próprio HI em RefEoL (ref, J, L locais). WAPE_RUL e WAPE_HI de cada rolamento são os valores
        finais (ponderados por (k - k_seg)^2, locais ao rolamento), lidos antes do restart seguinte.
        Demais argumentos: iguais a Optimize (faixas de busca, w/wS, sessões, nLS com n_trials por nº de camadas, OptPrune,
        OptReset, OptWarm, n_best, folder). O mesmo w é usado em todos os rolamentos do trial.

        Resultados em <folder>/<FileName>_<B1_1-B1_2>_restart[-<state>]_<feedback>_G<wtaG>P<wtaP>/ (optuna.db e mono/Optimization.csv).
        Optimization.csv: coluna do objetivo (OBJ_<obj>_<agg>), os 10 parâmetros (mesmo formato de Optimize/ParseParams),
        w e, de cada rolamento, WAPE_RUL_<nome>, WAPE_HI_<nome> (brutos), R_<nome>, H_<nome> (normalizados), E_<nome>
        (erro do rolamento) e, com norm='ref', REF_RUL_<nome> e REF_HI_<nome>.
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
    if not data or len(data) < 2: raise ValueError('Informe data com pelo menos dois rolamentos: [(nome, RS, HI), ...].')
    spec = _Spec(agg=agg, norm=norm, obj=obj, lims=lims, alpha=alpha, eps=eps, ref_win=ref_win, lim_prune=lim_prune)
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

    obj_name = _ObjName(spec)
    study_dir = os.path.join(folder, FileName, 'mono')
    os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')

    db_path = os.path.abspath(os.path.join(folder, FileName, 'optuna.db'))
    storage = _Storage(db_path)
    prefix = ObjPrefix(**spec)                     # configurações do objetivo diferentes não se misturam no mesmo banco
    summary = []
    nL_cur = None                                   # nº de camadas da rodada atual (laço abaixo)
    seen_cur = {}                                   # combinações já testadas no estudo atual (_Seen; atualizado por _Track)
    Agg = lambda v: float(_AGGF[spec['agg']](v))  # agregação sobre os rolamentos (inclusive objetivo parcial)
    refs = {}                                       # norm='ref': (rolamento, kX[0]) -> (WAPE_RUL_ref, WAPE_HI_ref)

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
        mO = trial.suggest_int('mO', mOS[0], mOS[1])
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
            if spec['norm'] == 'ref' and (nm, int(kX[0])) not in refs:   # depende só de kX[0] (J e L decorrem dele)
                try: refs[(nm, int(kX[0]))] = RefWAPE(s_, eol, kX, J, L, ref_win)
                except ValueError: raise TrialPruned()       # nenhum passo avaliado
            ds.append((nm, X, Y, Z, ref, J, L, int(kX[0])))

        teda = AutoCloud(m=m,nI=nI,nR=nR,nO=nO,mO=mO,ηS=[N1,N2,N3],act=act,
                         tau=τ,eol=eol,ref=ds[0][4],wtaG=wtaG,wtaP=wtaP,feedback=feedback)
        errs = []
        for b, (nm, X, Y, Z, ref, J, L, k0) in enumerate(ds):
            if b > 0: teda.restart(ref, state=state, x0=Y[0])   # mantém grânulos, pesos e modelo do erro; zera RUL/HI/WAPE
            for j in range(J):
                teda.run(X[j])
                teda.RUL_Prediction(Y[j],mode='single',lim=L)   # WAPE_RUL do passo
                teda.Adapt(Y[j],Z[j])                            # WAPE_HI do passo e treino das RNNs
                if not np.isfinite(teda.hiP[-1]): raise TrialPruned()
                if OptPrune and j >= 40 and j%5 == 0:
                    if teda.wape_RUL > 0.6 or teda.wape_HI > 0.3: raise TrialPruned()
            wR, wH = float(teda.wape_RUL), float(teda.wape_HI)
            if spec['lim_prune'] and (wR > spec['lims'][0] or wH > spec['lims'][1]): raise TrialPruned()
            trial.set_user_attr(f'WAPE_RUL_{nm}', wR)
            trial.set_user_attr(f'WAPE_HI_{nm}', wH)
            ref_b = refs.get((nm, k0))
            if ref_b is not None:
                trial.set_user_attr(f'REF_RUL_{nm}', ref_b[0])
                trial.set_user_attr(f'REF_HI_{nm}', ref_b[1])
            e, r, h = _Combine(wR, wH, ref_b, spec)
            trial.set_user_attr(f'R_{nm}', r)
            trial.set_user_attr(f'H_{nm}', h)
            trial.set_user_attr(f'E_{nm}', e)
            errs.append(e)
            if OptPrune:
                trial.report(Agg(errs), b)           # objetivo parcial ao fim de cada rolamento
                if trial.should_prune(): raise TrialPruned()
        return Agg(errs)

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
            study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,direction='minimize',
                                        pruner=optuna.pruners.HyperbandPruner(),sampler=_Sampler(OptSampler,seed,n_workers))
            n_prev, done = len(study.trials), _Complete(study)
            seen_cur.clear(); seen_cur.update(_Seen(study))   # inclui as execuções anteriores (retomada)

            if OptWarm and worker == 0:
                others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
                cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
                if cand:
                    best = min(cand, key=lambda t: t.value)
                    study.enqueue_trial(best.params, skip_if_exists=True)
                    print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {best.value:.4f}).')

            callback = EarlyStoppingCallback(patience=patience, context={
                'nL': [nL, nL], 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
                'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout,
                'storage': storage, 'prefix': prefix})
            if done:
                best_prev = min(done, key=lambda t: t.value)
                callback.best_score = best_prev.value
                callback.best_number = best_prev.number
                callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {best_prev.value:.4f} | Params: {best_prev.params}'
            print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else '')
                  + f' | semente do sampler: {seed}' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
            study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout,
                           callbacks=[_Track(seen_cur), _SharedBest(callback, n_workers), callback])

            done = _Complete(study)
            summary.append((study_name, n_prev, len(study.trials), min(t.value for t in done) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials (todos os workers) | melhor {best:.4f}')

    if not write_table:
        print('write_table=False: Optimization.csv não foi atualizado (use SeqTable ao fim de todas as janelas).')
        return None
    return SeqTable(db_path, names_b, n_best=n_best, out_path=out_path, w=w, **spec)


def SeqTable(db_path, names_b, agg='mean', n_best=10, out_path=None, w=1, ref_win=10,
             norm='ref', obj='cheb', lims=(0.3, 0.2), alpha=0.5, eps=0.05, lim_prune=None):
    '''
    Reconstrói o Optimization.csv de OptimizeSeq a partir do banco do Optuna (estudos com o prefixo ObjPrefix), sem executar
    trials: n_best melhores por nº de camadas, com w e o WAPE de cada rolamento. Usado ao fim de OptimizeSeq e, nas
    execuções em paralelo (uma janela por faixa de nº de camadas, mesmo banco), uma única vez depois que todas terminam.
        db_path : caminho do optuna.db.
        names_b : nomes dos rolamentos, na ordem (ex.: ['B1_1', 'B1_2']).
        out_path: caminho do .csv; None = <pasta do banco>/mono/Optimization.csv.
        w       : valor da coluna w para trials sem o parâmetro 'w' (bancos da V4_7 ou anteriores).
        agg, norm, obj, lims, alpha, eps, ref_win, lim_prune: configuração do objetivo (mesma de OptimizeSeq), que
              identifica os estudos (ObjPrefix); norm='ref' acrescenta REF_RUL_<b> e REF_HI_<b>.
    '''
    spec = _Spec(agg=agg, norm=norm, obj=obj, lims=lims, alpha=alpha, eps=eps, ref_win=ref_win, lim_prune=lim_prune)
    if not os.path.exists(db_path): raise FileNotFoundError(f'{db_path} não encontrado.')
    if out_path is None:
        os.makedirs(os.path.join(os.path.dirname(db_path), 'mono'), exist_ok=True)
        out_path = os.path.join(os.path.dirname(db_path), 'mono', 'Optimization.csv')
    storage = _Storage(db_path)
    prefix = ObjPrefix(**spec)
    obj_name = _ObjName(spec)
    prm_names = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation', 'w']
    seg_cols = [c for nm in names_b for c in (f'WAPE_RUL_{nm}', f'WAPE_HI_{nm}', f'R_{nm}', f'H_{nm}', f'E_{nm}')]
    if spec['norm'] == 'ref': seg_cols += [c for nm in names_b for c in (f'REF_RUL_{nm}', f'REF_HI_{nm}')]
    names = [obj_name] + prm_names

    # Tabela reconstruída do banco: n_best melhores por nº de camadas, com o WAPE de cada rolamento
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['act'],p.get('w', w)]
            rows.append([t.value] + prm + [t.user_attrs.get(c, np.nan) for c in seg_cols] + [p['n_layers']])
    if not rows:
        print('Nenhum trial concluído: Optimization.csv não foi gerado.')
        return pd.DataFrame(columns=names + seg_cols)
    df_all = pd.DataFrame(rows, columns=names + seg_cols + ['_nL'])
    df_all = df_all.sort_values(obj_name).drop_duplicates(subset=prm_names)
    df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best)
    df_stdy = df_stdy.sort_values(['_nL', obj_name], ascending=[True, False])

    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas ({out_path})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp[obj_name].min():.4f} | pior mantido {grp[obj_name].max():.4f}')

    df_stdy = df_stdy.drop(columns=['_nL']).reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


def _Parse10(vals10, w):
    # Utils.ParseParams atual exige e devolve 11 itens [m, nI, nR, nO, mO, N1, N2, N3, τ, act, w]: acrescenta w e separa
    p = ParseParams(list(vals10) + [w])
    return p[:10], int(p[10])


def BestParams(csv_path, w=1):
    '''Melhor linha (menor erro, 1ª coluna) de um Optimization.csv de Optimize/OptimizeSeq, no formato
    [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (sem w; ver BestParamsW).'''
    return BestParamsW(csv_path, w)[0]


def BestParamsW(csv_path, w=1):
    '''Melhor linha (menor erro, 1ª coluna) de um Optimization.csv de Optimize/OptimizeSeq: (params, w), com params no
    formato [m, nI, nR, nO, mO, N1, N2, N3, τ, act] e w da coluna 'w' da mesma linha
    (tabelas sem a coluna 'w', da V4_7 ou anteriores: devolve o w informado).'''
    df = pd.read_csv(csv_path)
    row = df.sort_values(df.columns[0], ascending=True).iloc[0]
    return _Parse10(row.iloc[1:11].values, int(row['w']) if 'w' in df.columns else int(w))


def SimSeq(params, data, state='keep', mode_rul='interval', rho=0.1, intr_state='carry',
           eol=0.3, w=1, feedback='symmetric', wtaG=False, wtaP=False, store=True, on_end=None):
    '''
    Simula uma sequência de rolamentos com parâmetros fixos, no mesmo procedimento de OptimizeSeq:
        1º rolamento até o EoL (SimGranularRUL); antes de cada rolamento seguinte, teda.restart(ref, state, Y[0]).
    params  : [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (ParseParams/BestParams).
    data    : lista ordenada de (nome, RS, HI).
    mode_rul: 'interval' (RUL e HI intervalares, com rho e intr_state) ou 'single' (apenas pontuais).
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
                             wtaG=wtaG, wtaP=wtaP, store=store, feedback=feedback, rho=rho, intr_state=intr_state)
        else:
            teda.restart(ref, state=state, x0=Y[0])
        teda.SimGranularRUL(J, L, X, Y, Z, mode_rul)
        rows.append(dict(ordem=i + 1, rolamento=nm, passos=len(teda.rulP), granulos=len(teda.c),
                         WAPE_RUL=float(teda.wape_RUL), WAPE_HI=float(teda.wape_HI),
                         V_HI=float(teda.V), cov_HI=float(teda.cov), sp_HI=float(teda.sp)))
        if on_end is not None: on_end(teda, nm, i + 1)
    return teda, rows
