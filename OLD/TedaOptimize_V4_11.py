'''
Otimização (Optuna) do AutoCloud_V4_6 com rho estático.

    Optimize    : um rolamento; minimiza WAPE_RUL + WAPE_HI (predições pontuais).
    OptimizeSeq : sequência de rolamentos assimilada pelo mesmo modelo (AutoCloud.restart entre rolamentos);
                  mode='single': minimiza a agregação (agg) sobre os rolamentos do erro combinado (norm, obj);
                  mode='multi' : minimiza os dois objetivos (F_RUL, F_HI) = agregação de (r_b, h_b) (frente de Pareto).
    BestParams  : melhor linha de um Optimization.csv no formato de ParseParams.
    BestParamsW : idem, com a janela w da mesma linha: (params, w).
    SimSeq      : simulação final de uma sequência de rolamentos com parâmetros fixos (mesmo procedimento de OptimizeSeq).

    RefWAPE     : WAPE_RUL e WAPE_HI de preditores de referência (normalização norm='ref').
    ObjPrefix   : prefixo dos estudos no banco para uma configuração do objetivo (mode, norm, obj, agg, ...).
    ParetoTable : todos os trials de uma configuração multi, com a frente de Pareto global e o escore de seleção.

Nas duas otimizações, nLS = [min, max] executa n_study sessões de n_trials trials para CADA nº de camadas.
Uso no notebook (na pasta principal): from Functions.TedaOptimize_V4_11 import *

Mudanças em relação à V4_10:
    - OptimizeSeq(mode='single' | 'multi'). 'single' (padrão): igual à V4_10. 'multi': otimização multiobjetivo, com
      dois objetivos minimizados (Optuna, directions = [minimize, minimize]):
        F_RUL = agg_b(r_b) e F_HI = agg_b(h_b), com r_b, h_b normalizados por norm (como na V4_10) e agg sobre os rolamentos;
      obj e alpha não se aplicam (a combinação de r e h é substituída pela frente de Pareto);
      - poda: trial.report/should_prune não existem no Optuna multiobjetivo, então a poda por objetivo parcial
        (Hyperband) é desativada; continuam as podas por limiares (OptPrune: 0.6/0.3 dentro do rolamento; lim_prune);
      - escolha do modelo na frente: escore de seleção SEL = max(F_RUL, F_HI) + eps*(F_RUL + F_HI) (Chebyshev
        aumentado sobre os objetivos já normalizados); o mínimo de SEL é sempre um ponto da frente de Pareto (eps > 0).
        Optimization.csv tem SEL_cheb na 1ª coluna (BestParams/BestParamsW escolhem esse ponto), os parâmetros, w e,
        por rolamento, WAPE_RUL_<b>, WAPE_HI_<b> finais; por nº de camadas, as n_best linhas de menor (front, SEL), com
        front = posição na ordenação não dominada do nº de camadas (1 = frente). mono/Pareto.csv: todos os trials
        concluídos, mesmas colunas, ordenados por SEL. ParetoTable (DataFrame, não gravado) acrescenta OBJ_R_<agg>,
        OBJ_H_<agg>, nL e pareto (1 = frente de Pareto global), usados no gráfico da frente;
      - colunas dos .csv (single e multi): objetivo/SEL_cheb, parâmetros, w e WAPE_RUL_<b>, WAPE_HI_<b> de cada rolamento,
        na ordem da sequência (valores no último passo de cada rolamento enviado ao modelo, no EoL; _CsvCols);
      - parada antecipada (_ParetoStop): melhora = trial concluído que entra na frente de Pareto (não dominado pelos
        trials concluídos anteriores do estudo) ou reduz o melhor SEL; com vários workers, frente com escore SEL
        melhor encontrada por outro worker também zera a contagem (_SharedBest);
      - estudos com o prefixo mseq_<norm>_<agg>[_cut...]_ (ObjPrefix): não se misturam com os estudos single (seq_...).
      eps é usado apenas na escolha (tabela), não na otimização: não faz parte do prefixo.

Mudanças da V4_10 em relação à V4_9:
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
from Functions.Utils_OPT_R2 import clear_output

__all__ = ['Optimize', 'OptimizeSeq', 'SeqTable', 'ParetoTable', 'BestParams', 'BestParamsW', 'SimSeq', 'RefWAPE',
           'ObjPrefix']

_AGGS = ('mean', 'sum', 'max')
_NORMS = ('none', 'lim', 'ref')
_OBJS = ('sum', 'prod', 'cheb')
_MODES = ('single', 'multi')
_AGGF = {'mean': np.mean, 'sum': np.sum, 'max': np.max}


def _Spec(agg='mean', norm='ref', obj='cheb', lims=(0.3, 0.2), alpha=0.5, eps=0.05, ref_win=10, lim_prune=None,
          mode='single'):
    '''Valida e completa a configuração do objetivo de OptimizeSeq / SeqTable (ver OptimizeSeq). Retorna um dict.
    mode='multi': obj é fixado em 'cheb' (E_<b> de cada rolamento e escolha na frente pelo mesmo critério).'''
    if mode not in _MODES: raise ValueError(f'mode deve ser um de {_MODES}.')
    if mode == 'multi': obj = 'cheb'
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
                lim_prune=bool(lim_prune), mode=mode)


def ObjPrefix(**kw):
    '''Prefixo dos estudos no banco para uma configuração do objetivo (argumentos de _Spec): configurações diferentes
    não se misturam. Ex.: seq_ref10_cheb0.05_mean_ ; seq_lim0.3-0.2_prod_mean_cut0.3-0.2_ ; mseq_ref10_mean_ (multi)'''
    s = _Spec(**kw)
    lim = f"{s['lims'][0]:g}-{s['lims'][1]:g}"
    nt = {'none': 'none', 'lim': f'lim{lim}', 'ref': f"ref{s['ref_win']}"}[s['norm']]
    ot = {'sum': f"sum{s['alpha']:g}", 'prod': 'prod', 'cheb': f"cheb{s['eps']:g}"}[s['obj']]
    cut = f'_cut{lim}' if s['lim_prune'] else ''
    if s['mode'] == 'multi': return f"mseq_{nt}_{s['agg']}{cut}_"     # eps só escolhe o ponto na frente: fora do prefixo
    return f"seq_{nt}_{ot}_{s['agg']}{cut}_"


def _ObjName(s):
    return 'SEL_cheb' if s['mode'] == 'multi' else f"OBJ_{s['obj']}_{s['agg']}"


def _Sel(values, eps):
    # Escore de seleção na frente (modo multi): Chebyshev aumentado sobre (F_RUL, F_HI) normalizados
    R, H = values
    return float(max(R, H) + eps * (R + H))


def _ScoreFn(s):
    # Escore escalar de um trial concluído: valor do objetivo (single) ou SEL (multi)
    if s['mode'] == 'multi': return lambda t: _Sel(t.values, s['eps'])
    return lambda t: t.value


def _NonDom(V):
    '''Máscara dos pontos não dominados (minimização) de V (n x 2): varredura por F_RUL crescente. Pontos iguais
    não se dominam (ambos ficam na frente).'''
    V = np.asarray(V, dtype=float)
    n = len(V)
    nd = np.zeros(n, dtype=bool)
    if n == 0: return nd
    order = np.lexsort((V[:, 1], V[:, 0]))
    best_h, i = np.inf, 0
    while i < n:
        j = i
        while j < n and V[order[j], 0] == V[order[i], 0]: j += 1
        g = order[i:j]
        hmin = V[g, 1].min()
        if hmin < best_h: nd[g] = V[g, 1] == hmin
        best_h = min(best_h, hmin)
        i = j
    return nd


def _FrontRank(V, need=None):
    '''Ordenação não dominada: 1 = frente de Pareto, 2 = frente após remover a 1ª, ... Para quando há pelo menos
    need pontos ordenados (os demais ficam com posição inf).'''
    V = np.asarray(V, dtype=float)
    rank = np.full(len(V), np.inf)
    left, r = np.arange(len(V)), 1
    while len(left) and (need is None or np.isfinite(rank).sum() < need):
        nd = _NonDom(V[left])
        rank[left[nd]] = r
        left, r = left[~nd], r + 1
    return rank


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
    return prev.value if len(prev.values) == 1 else tuple(prev.values)   # multi: os dois objetivos


def _Lookup(seen, trial, n_workers):
    # Combinação já testada? Com vários workers no mesmo estudo, os trials concluídos pelos outros processos só estão
    # no banco: na falta no dicionário local, relê o estudo (uma consulta por trial novo) antes de simular
    key = _Key(trial.params)
    prev = seen.get(key)
    if prev is None and n_workers > 1:
        seen.update({k: t for k, t in _Seen(trial.study).items() if k not in seen})
        prev = seen.get(key)
    return prev


def _SharedBest(callback, n_workers, score=None):
    # Callback (antes do EarlyStoppingCallback): com vários workers, melhora encontrada por outro processo também zera
    # a contagem de patience deste processo (patience = trials deste worker sem melhora do ESTUDO).
    # score = None: single (study.best_trial); score = função do trial: multi (melhor SEL da frente, study.best_trials)
    def cb(study, t):
        if n_workers <= 1: return
        try:
            if score is None:
                best = study.best_trial
                sb = best.value
            else:
                front = study.best_trials
                if not front: return
                best = min(front, key=score)
                sb = score(best)
        except ValueError: return                    # nenhum trial concluído ainda
        if best.number != t.number and sb < callback.best_score:
            callback.best_score, callback.best_number = sb, best.number
            callback.best_trial_info = f'Trial #{best.number:03d} (outro worker) | Score: {sb:.4f} | Params: {best.params}'
            callback.trials_without_improvement = 0
    return cb


class _ParetoStop(EarlyStoppingCallback):
    '''
    EarlyStoppingCallback para o modo multi. Melhora (zera a contagem de patience): trial concluído que entra na frente
    de Pareto (não dominado pelos trials concluídos anteriores do estudo; valores iguais a um trial anterior não contam)
    ou que reduz o melhor escore de seleção SEL = max(F_RUL, F_HI) + eps*(F_RUL + F_HI). A exibição usa SEL.
    '''
    def __init__(self, patience, eps, context=None):
        self.eps = eps                       # antes de super().__init__: LoadBestPerNL usa UpdateBestPerNL
        self.V = None                        # objetivos dos trials concluídos do estudo (lidos na 1ª chamada)
        super().__init__(patience, context=context)

    def Sel(self, t):
        return _Sel(t.values, self.eps)

    def UpdateBestPerNL(self, study_name, t):
        nL = t.params.get('n_layers')
        if nL is None or t.values is None or len(t.values) != 2: return
        score = self.Sel(t)
        if nL not in self.nl_best or score < self.nl_best[nL][0]:
            self.nl_best[nL] = (score, study_name, t.number, t.params)

    def PrintBestPerNL(self, is_multi):
        nLc = self.context.get('nL')
        running = set(range(nLc[0], nLc[1] + 1)) if isinstance(nLc, (list, tuple)) else {nLc}
        print("=" * 80)
        print("BEST TRIAL PER nL (SEL = max(F_RUL, F_HI) + eps*(F_RUL + F_HI))")
        print("=" * 80)
        for nL in sorted(set(self.nl_best) | running):
            mark = '   <-- running' if nL in running else ''
            if nL in self.nl_best:
                score, name, number, params = self.nl_best[nL]
                print(f"nL={nL} | {name} #{number:04d} | SEL: {score:.4f} | Params: {params}{mark}")
            else:
                print(f"nL={nL} | no completed trials yet{mark}")
        if self.best_score < float('inf'):
            number = f"#{self.best_number:04d}" if self.best_number is not None else "#----"
            print(f"Current session best: {number} | SEL: {self.best_score:.4f}")
        else:
            print("Current session best: no completed trials yet")

    def __call__(self, study, trial):
        COMPLETE = optuna.trial.TrialState.COMPLETE
        self.n_run += 1
        if self.V is None:
            self.V = [list(t.values) for t in study.get_trials(deepcopy=False, states=(COMPLETE,))
                      if t.number != trial.number]
        if trial.state == COMPLETE:
            v, sel = list(trial.values), self.Sel(trial)
            self.last_trials.append(f"Trial #{trial.number:03d} [COMPLETE] | F_RUL, F_HI: {[round(x, 4) for x in v]} | "
                                    f"SEL: {sel:.4f} | Params: {trial.params}")
            V = np.asarray(self.V, dtype=float).reshape(-1, 2)
            dominated = bool(np.any(np.all(V <= v, axis=1) & np.any(V < v, axis=1))) if len(V) else False
            equal = bool(np.any(np.all(V == v, axis=1))) if len(V) else False
            self.V.append(v)
            if self.per_nl: self.UpdateBestPerNL(study.study_name, trial)
            improved = not dominated and not equal
            if sel < self.best_score:
                self.best_score, self.best_number, improved = sel, trial.number, True
                self.best_trial_info = f"Trial #{trial.number:03d} | F_RUL, F_HI: {[round(x, 4) for x in v]} | SEL: {sel:.4f} | Params: {trial.params}"
            self.trials_without_improvement = 0 if improved else self.trials_without_improvement + 1
        else:
            self.last_trials.append(f"Trial #{trial.number:03d} [{trial.state.name}]")

        clear_output(wait=True)
        if self.context: self.PrintContext(study)
        if self.per_nl: self.PrintBestPerNL(True)
        else:
            print("=" * 80); print("BEST TRIAL (SEL)"); print("=" * 80)
            print(self.best_trial_info or "NO COMPLETED TRIALS FINISHED YET.")
        if self.V: print(f"-> SOLUTIONS IN THE PARETO FRONT (this study): {int(_NonDom(np.asarray(self.V, dtype=float)).sum())}")
        print("\n" + "=" * 80); print(f"LAST {self.last_trials.maxlen} TRIALS"); print("=" * 80)
        for log in self.last_trials: print(log)
        print("-" * 80)
        print(f"STOPPING STATUS: {self.trials_without_improvement}/{int(self.patience)} ITERATIONS WITHOUT IMPROVEMENT "
              f"(improvement = new Pareto point or lower SEL).")
        print("-" * 80)
        if self.trials_without_improvement >= self.patience:
            print(f"\n>>> STUDY INTERRUPTED! NO IMPROVEMENTS FOR {int(self.patience)} ITERATIONS.")
            study.stop()

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
        norm='ref',obj='cheb',lims=(0.3,0.2),alpha=0.5,eps=0.05,lim_prune=None,mode='single',
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
        mode : 'single' (padrão) -> um objetivo, como acima;
               'multi'  -> dois objetivos minimizados, F_RUL = agg_b(r_b) e F_HI = agg_b(h_b) (obj e alpha não se
                           aplicam; sem poda por objetivo parcial; parada antecipada por frente de Pareto, _ParetoStop).
                           O modelo escolhido na frente (1ª linha de menor SEL_cheb do Optimization.csv) minimiza
                           SEL = max(F_RUL, F_HI) + eps*(F_RUL + F_HI).
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
        OptPrune: poda ao fim de cada rolamento (trial.report do objetivo parcial; só em mode='single') e por limiares
              dentro do rolamento.
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
    spec = _Spec(agg=agg, norm=norm, obj=obj, lims=lims, alpha=alpha, eps=eps, ref_win=ref_win, lim_prune=lim_prune,
                 mode=mode)
    multi = spec['mode'] == 'multi'
    Score = _ScoreFn(spec)                          # escore escalar de um trial concluído (valor ou SEL)
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
        errs, rs, hs = [], [], []
        for b, (nm, X, Y, Z, ref, J, L, k0) in enumerate(ds):
            if b > 0: teda.restart(ref, state=state, x0=Y[0])   # mantém grânulos, pesos e modelo do erro; zera RUL/HI/WAPE
            for j in range(J):
                teda.run(X[j])
                teda.RUL_Prediction(Y[j],mode='single',lim=L)   # WAPE_RUL do passo
                teda.Adapt(Y[j],Z[j])                            # WAPE_HI do passo e treino das RNNs
                if not np.isfinite(teda.hiP[-1]): raise TrialPruned()
                if OptPrune and j >= 40 and j%5 == 0:
                    if teda.wape_RUL > 0.6 or teda.wape_HI > 0.3: raise TrialPruned()
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
            errs.append(e); rs.append(r); hs.append(h)
            if OptPrune and not multi:               # multiobjetivo: o Optuna não permite report/should_prune
                trial.report(Agg(errs), b)           # objetivo parcial ao fim de cada rolamento
                if trial.should_prune(): raise TrialPruned()
        if multi: return Agg(rs), Agg(hs)            # (F_RUL, F_HI)
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
            if multi:
                study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                            directions=['minimize','minimize'],pruner=optuna.pruners.NopPruner(),
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
                    best = min(cand, key=Score)
                    study.enqueue_trial(best.params, skip_if_exists=True)
                    print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {Score(best):.4f}).')

            context = {'nL': [nL, nL], 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
                       'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout, 'storage': storage, 'prefix': prefix}
            callback = _ParetoStop(patience, spec['eps'], context) if multi else EarlyStoppingCallback(patience=patience, context=context)
            if done:
                best_prev = min(done, key=Score)
                callback.best_score = Score(best_prev)
                callback.best_number = best_prev.number
                callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {Score(best_prev):.4f} | Params: {best_prev.params}'
            print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else '')
                  + f' | semente do sampler: {seed}' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
            study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout,
                           callbacks=[_Track(seen_cur), _SharedBest(callback, n_workers, Score if multi else None), callback])

            done = _Complete(study)
            summary.append((study_name, n_prev, len(study.trials), min(Score(t) for t in done) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')' + (f' | worker {worker + 1}/{n_workers}' if n_workers > 1 else ''))
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials (todos os workers) | melhor {"SEL " if multi else ""}{best:.4f}')

    if not write_table:
        print('write_table=False: Optimization.csv não foi atualizado (use SeqTable ao fim de todas as janelas).')
        return None
    return SeqTable(db_path, names_b, n_best=n_best, out_path=out_path, w=w, **spec)   # spec inclui mode


def SeqTable(db_path, names_b, agg='mean', n_best=10, out_path=None, w=1, ref_win=10,
             norm='ref', obj='cheb', lims=(0.3, 0.2), alpha=0.5, eps=0.05, lim_prune=None, mode='single'):
    '''
    Reconstrói o Optimization.csv de OptimizeSeq a partir do banco do Optuna (estudos com o prefixo ObjPrefix), sem executar
    trials: n_best melhores por nº de camadas, com w e o WAPE de cada rolamento. Usado ao fim de OptimizeSeq e, nas
    execuções em paralelo (uma janela por faixa de nº de camadas, mesmo banco), uma única vez depois que todas terminam.
        db_path : caminho do optuna.db.
        names_b : nomes dos rolamentos, na ordem (ex.: ['B1_1', 'B1_2']).
        out_path: caminho do .csv; None = <pasta do banco>/mono/Optimization.csv.
        w       : valor da coluna w para trials sem o parâmetro 'w' (bancos da V4_7 ou anteriores).
        agg, norm, obj, lims, alpha, eps, ref_win, lim_prune: configuração do objetivo (mesma de OptimizeSeq), que
              identifica os estudos (ObjPrefix).
        Colunas gravadas: objetivo, parâmetros, w e WAPE_RUL_<b>, WAPE_HI_<b> finais de cada rolamento (_CsvCols).
        mode='multi': 1ª coluna SEL_cheb (escolha na frente); por nº de camadas, as n_best linhas de menor
              (front, SEL_cheb). Grava também <pasta>/Pareto.csv (todos os trials, mesmas colunas, ordenados por SEL_cheb).
    '''
    spec = _Spec(agg=agg, norm=norm, obj=obj, lims=lims, alpha=alpha, eps=eps, ref_win=ref_win, lim_prune=lim_prune,
                 mode=mode)
    if spec['mode'] == 'multi':
        return _SeqTableMulti(db_path, names_b, spec, n_best, out_path, w)
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

    df_stdy = df_stdy[_CsvCols(obj_name, names_b)].reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


_PRM_NAMES = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation', 'w']


def _CsvCols(first, names_b):
    # Colunas gravadas nos .csv (Optimization.csv e Pareto.csv): objetivo/escore, parâmetros, w e, por rolamento na
    # ordem da sequência, WAPE_RUL_<b> e WAPE_HI_<b> finais (último passo do rolamento enviado ao modelo, no EoL)
    return [first] + _PRM_NAMES + [c for nm in names_b for c in (f'WAPE_RUL_{nm}', f'WAPE_HI_{nm}')]


def _MultiRows(db_path, names_b, spec, w):
    # Todos os trials concluídos dos estudos multi da configuração (um por combinação de parâmetros)
    storage = _Storage(db_path)
    prefix = ObjPrefix(**spec)
    ag = spec['agg']
    seg_cols = [c for nm in names_b for c in (f'WAPE_RUL_{nm}', f'WAPE_HI_{nm}', f'R_{nm}', f'H_{nm}', f'E_{nm}')]
    if spec['norm'] == 'ref': seg_cols += [c for nm in names_b for c in (f'REF_RUL_{nm}', f'REF_HI_{nm}')]
    cols = ['SEL_cheb'] + _PRM_NAMES + [f'OBJ_R_{ag}', f'OBJ_H_{ag}'] + seg_cols + ['_nL']
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['act'],p.get('w', w)]
            rows.append([_Sel(t.values, spec['eps'])] + prm + list(t.values) + [t.user_attrs.get(c, np.nan) for c in seg_cols]
                        + [p['n_layers']])
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values('SEL_cheb').drop_duplicates(subset=_PRM_NAMES).reset_index(drop=True)


def ParetoTable(db_path, names_b, w=1, **obj_kw):
    '''
    Modo multi: todos os trials concluídos (uma linha por combinação de parâmetros) da configuração do objetivo obj_kw
    (argumentos de OptimizeSeq: agg, norm, lims, eps, ref_win, lim_prune; mode é sempre 'multi'), com OBJ_R_<agg>,
    OBJ_H_<agg>, SEL_cheb, nL (nº de camadas) e pareto (1 = frente de Pareto global, todos os nº de camadas).
    Ordenada por SEL_cheb: a 1ª linha é o modelo escolhido. Usada para o gráfico da frente.
    '''
    spec = _Spec(**{**obj_kw, 'mode': 'multi'})
    df = _MultiRows(db_path, names_b, spec, w)
    ag = spec['agg']
    df['pareto'] = _NonDom(df[[f'OBJ_R_{ag}', f'OBJ_H_{ag}']].to_numpy()).astype(int) if len(df) else []
    return df.rename(columns={'_nL': 'nL'})


def _SeqTableMulti(db_path, names_b, spec, n_best, out_path, w):
    if not os.path.exists(db_path): raise FileNotFoundError(f'{db_path} não encontrado.')
    if out_path is None:
        os.makedirs(os.path.join(os.path.dirname(db_path), 'mono'), exist_ok=True)
        out_path = os.path.join(os.path.dirname(db_path), 'mono', 'Optimization.csv')
    ag = spec['agg']
    df = _MultiRows(db_path, names_b, spec, w)
    if df.empty:
        print('Nenhum trial concluído: Optimization.csv não foi gerado.')
        return df.drop(columns=['_nL'])
    V_cols = [f'OBJ_R_{ag}', f'OBJ_H_{ag}']
    df['pareto'] = _NonDom(df[V_cols].to_numpy()).astype(int)            # frente global (todos os nº de camadas)
    df['front'] = np.inf
    for nL_, idx in df.groupby('_nL').groups.items():                       # ordenação não dominada por nº de camadas
        df.loc[idx, 'front'] = _FrontRank(df.loc[idx, V_cols].to_numpy(), need=n_best)
    df_sel = (df.sort_values(['_nL', 'front', 'SEL_cheb']).groupby('_nL', group_keys=False).head(n_best)
                .sort_values(['_nL', 'front', 'SEL_cheb'], ascending=[True, False, False]))
    seg = [c for c in df.columns if c not in ['SEL_cheb'] + _PRM_NAMES + V_cols + ['_nL', 'pareto', 'front']]
    out = df_sel[['SEL_cheb'] + _PRM_NAMES + V_cols + ['front', 'pareto'] + seg + ['_nL']]

    best = df.iloc[0]
    print(f'\nOptimization.csv (multi): até {n_best} linhas por nº de camadas, por (front, SEL_cheb) ({out_path})')
    for nL_, grp in out.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | pontos na frente do nL: {int((df["_nL"].eq(nL_) & df["front"].eq(1)).sum())} | '
              f'melhor SEL {grp["SEL_cheb"].min():.4f}')
    print(f'  frente de Pareto global: {int(df["pareto"].sum())} pontos de {len(df)} trials | escolhido (menor SEL): '
          f'nL={best["_nL"]} | {V_cols[0]}={best[V_cols[0]]:.4f} | {V_cols[1]}={best[V_cols[1]]:.4f} | SEL={best["SEL_cheb"]:.4f}')
    cols = _CsvCols('SEL_cheb', names_b)
    out = out[cols].reset_index(drop=True)
    out.to_csv(out_path, index=False)
    par_path = os.path.join(os.path.dirname(out_path), 'Pareto.csv')
    df[cols].to_csv(par_path, index=False)
    print(f'  todos os trials (ordenados por SEL_cheb): {par_path}')
    return out


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
