import ast
import os
import optuna
import ipynbname
import pandas as pd
import numpy as np
from optuna.exceptions import TrialPruned
from Functions.AutoCloud_V4_5 import *
from Functions.Utils import *
from Functions.Graphs import *
from Functions.TedaGraphs import *
from Functions.Utils_OPT_R2 import *
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--instancia", default="principal")
args = parser.parse_args()

FileName = (
    os.path.splitext(os.path.basename(__file__))[0]
    + "_" + args.instancia
)

# Configuração do experimento
FEEDBACK = 'symmetric'      # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = False, False   # configuração principal: soma ponderada dos grânulos (eq. 36)
EOL = 0.3                   # limiar de fim de vida (EoL) do HI

# rho dinâmico (AutoCloud_V4_2): começa em 0 e é ajustado a cada iteração maximizando V = média(cov)*média(sp)
RHO_TARGET = 'RUL'           # 'HI' (causal, HI um passo à frente) ou 'RUL' (oráculo: usa a RUL real; n_rho previsões por passo)
RHO_MIN, RHO_MAX, N_RHO = 1e-3, 1, 4   # grade log-espaçada de candidatos
LAM = 1.0                   # fator de esquecimento (1.0 = todos os passos com o mesmo peso)

def RunTag(feedback=None, wtaG=None, wtaP=None):
    # Uma pasta de resultados por configuração (feedback, wtaG, wtaP)
    feedback = FEEDBACK if feedback is None else feedback
    wtaG = WTAG if wtaG is None else wtaG
    wtaP = WTAP if wtaP is None else wtaP
    return f'{FileName}_{feedback}_G{int(wtaG)}P{int(wtaP)}'

def RefEoL(sig, eol, nI, mO, mode):
    '''
    Referência da RUL real baseada no limiar de EoL (eq. 12): RUL_real = iEoL - k_dado.
    O AutoCloud calcula rulR = ref - k, com k = j + 2 na iteração j, e o PrepareData
    alinha a iteração j à amostra k_dado = k0 + j. Logo ref = iEoL - k0 + 2.
    Retorna ref e J, o número de iterações com RUL real >= 1.
    '''
    sig = np.asarray(sig, dtype=float)
    if not np.any(sig <= eol): raise ValueError('O HI nunca atinge o limiar de EoL.')
    iEoL = int(np.argmax(sig <= eol))
    k0 = (max(nI, 1, mO) if mode in ('ahead', 1) else max(nI, 1)) - 1
    return iEoL - k0 + 2, iEoL - k0

def nOMax(sig, eol):
    '''
    Maior nO (modo 'ahead') para o qual o fluxo do PrepareData alcança a última amostra antes do EoL.
    A última janela fica em k = K - nO - 2 (o PrepareData descarta a última); exigir k >= iEoL - 1
    dá nO <= K - iEoL - 1, independente de nI e mO. No modo 'past' o fluxo sempre alcança o EoL.
    '''
    sig = np.asarray(sig, dtype=float)
    return len(sig) - int(np.argmax(sig <= eol)) - 1

out_path = os.path.join('Optimization', RunTag(), 'multi', 'Optimization.csv')
dfs = [pd.read_csv(out_path)] if os.path.exists(out_path) else []
RS = pd.read_excel(r'Dataset\RS.xlsx')
HI = pd.read_excel(r'Dataset\HI.xlsx')
sig = HI['PC1'].values

def WAPE_Aligned(teda, acc):
    '''
    WAPE ponderado por k^2 (mesma forma de WAPE_RUL/WAPE_HI do AutoCloud), acumulado em acc = [dR, sR, dH, sH].
    Chamar após Adapt: compara a predição do passo k com o valor real do MESMO passo.
    (teda_WAPE, chamado dentro de RUL_Prediction antes de Adapt, compara hiP_k com hiR_{k-1}: o wape_HI do
    módulo fica deslocado de um passo; a RUL não tem esse problema.) Retorna (wape_RUL, wape_HI).
    '''
    k = teda.k
    if k > 2:
        t = k**2
        acc[0] += t*abs(teda.rulR[-1] - teda.rulP[-1]); acc[1] += t*abs(teda.rulR[-1])
        acc[2] += t*abs(teda.hiR[-1] - teda.hiP[-1]);   acc[3] += t*abs(teda.hiR[-1])
    wR = acc[0]/acc[1] if acc[1] > 0 else 0.0
    wH = acc[2]/acc[3] if acc[3] > 0 else 0.0
    return wR, wH

def Optimize(
        FileName=None,OptDim=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,mdS=None,actS=None,
        feedback=None,wtaG=None,wtaP=None,
        OptReset=False,OptWarm=False,n_best=5,
        rho=0.5,intr_state='reset'):

    '''
        OptSampler modes available: None, Auto, GP, NSGAII, Random, TPE
    
        None: sampler default, suport multivariate optimization

        GP: suport multivariate optimization

        Auto: suport multivariate optimization
        
        NSGAII: poorly suport multivariate optimization
        
        Random: poorly suport multivariate optimization

        TPE: suport multivariate optimization

        feedback, wtaG, wtaP: model configuration (defaults: FEEDBACK, WTAG, WTAP from cell 0).
        Results are saved in Optimization/<FileName>_<feedback>_G<wtaG>P<wtaP>/.

        Persistent storage (SQLite, Optimization/<...>/optuna.db):
            each of the n_study sessions is an independent study named obj<1|2>_L<nLS[0]>-<nLS[1]>_s<i>,
            resumed on every execution (n_trials = number of NEW trials per execution).
        OptReset: delete and recreate the studies of these sessions (use after changing the objective/code).
        OptWarm : enqueue the best trial of the other sessions of the same configuration before optimizing.
        OptSeed : base seed; session i uses OptSeed + i (sessions stay independent).
        n_best  : rows kept per number of layers (nL) in Optimization.csv. The table is rebuilt from ALL
                  trials stored in the database (all sessions/phases), keeping the n_best smallest errors
                  of distinct parameter sets per nL; rows are grouped by nL (worst -> best within each group).

        Objective (AutoCloud_V4_4): RUL_Prediction(mode='single') -> point HI and RUL predictions only.
            Error = WAPE_RUL + WAPE_HI (k^2-weighted), computed by WAPE_Aligned after Adapt (HI prediction of step k
            against the true HI of step k). rho is fixed (rho, rho_dyn=False) and does not affect the point
            predictions; rho and its parameters are optimized afterwards by OptimizeRho.
            Note: trials stored with the previous objective (module wape_HI, shifted by one step) are not
            comparable -> use OptReset=True (or another FileName) when reusing an old database.
    '''

    if feedback is None: feedback = FEEDBACK
    if wtaG is None: wtaG = WTAG
    if wtaP is None: wtaP = WTAP
    FileName = f'{FileName}_{feedback}_G{int(wtaG)}P{int(wtaP)}'

    if OptDim == 1 or OptDim is None: SingleObj = True
    elif OptDim > 1: SingleObj = False    
    if n_study is None: n_study = 1
    if timeout is None: timeout = 60
    if n_trials is None and patience is None: n_trials = 25
    if patience is None:
        if n_trials < 0.5e4: patience = 500
        else: patience = int(0.02*n_trials)

    if mS  is None: mS  = [1.75,4.25]
    if nIS is None: nIS = [2,40]
    if nLS is None: nLS = [1,5]
    if nRS is None: nRS = [1,60]
    if nOS is None: nOS = [1,40]
    if mOS is None: mOS = [0,40]
    if N1S is None: N1S = [1,9]
    if N2S is None: N2S = [1,9]
    if N3S is None: N3S = [1,9]
    if τS  is None: τS  = [1,25]
    if mdS is None: mdS = [0,1]
    if actS is None: actS = [0,1,2]

    # Limita nO para que o fluxo sempre alcance o EoL: a RUL é avaliada até a última amostra antes
    # do EoL em todos os trials (evita favorecer nO grande, que encurta o trecho avaliado)
    if mdS[1] == 1:
        nOS = [nOS[0], min(nOS[1], nOMax(sig, EOL))]
        if nOS[0] > nOS[1]: raise ValueError(f'nOS inválido: o limite para alcançar o EoL é nO <= {nOMax(sig, EOL)}.')
        print(f'nO limitado a {nOS} para que o fluxo alcance o EoL.')

    if SingleObj: names = ['MAPE_RUL+MAPE_HI', 'm', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3','TAU','past/ahead','activation']
    else: names = ['MAPE_RUL','MAPE_HI', 'm', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3','TAU','past/ahead','activation']
    study_dir, out_path = df_ParamsTable(names,FileName)

    # Armazenamento persistente: um banco SQLite por configuração; cada sessão é um estudo independente e retomável
    db_path = os.path.abspath(os.path.join('Optimization', FileName, 'optuna.db'))
    storage = optuna.storages.RDBStorage('sqlite:///' + db_path.replace(os.sep, '/'))
    base_name = f'obj{1 if SingleObj else 2}_L{nLS[0]}-{nLS[1]}'
    Score = lambda t: t.value if SingleObj else sum(t.values)
    Complete = lambda st: [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = []

    for i in range(n_study):
        study_name = f'{base_name}_s{i+1}'
        print(f'iteration: {i+1} | study: {study_name}')
        if OptReset and study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f'  {study_name}: estudo anterior apagado (OptReset=True).')

        def objective(trial):
            m = trial.suggest_float('m', mS[0], mS[1],step=0.25)
            nI = trial.suggest_int('nI', nIS[0], nIS[1]) 
            n_layers = trial.suggest_int('n_layers', nLS[0], nLS[1])
            nR = [trial.suggest_int(f'nR_layer_{l}', nRS[0], nRS[1]) for l in range(n_layers)]
            nO = trial.suggest_int('nO', nOS[0], nOS[1]) 
            N1 = trial.suggest_int('N1', N1S[0], N1S[1])
            N2 = trial.suggest_int('N2', N2S[0], N2S[1])
            N3 = trial.suggest_int('N3', N3S[0], N3S[1])
            τ = trial.suggest_int('τ', τS[0], τS[1])
            mode = trial.suggest_int('mode', mdS[0], mdS[1])  
            act = trial.suggest_int('act', actS[0], actS[1])  

            # Limites de nI e nO controlados apenas por nIS e nOS
            if mode == 'past' or mode == 0:
                mO = trial.suggest_int('mO', 0, 0) 
                if nO > nI: raise TrialPruned()

            elif mode == 'ahead' or mode == 1:
                mO = trial.suggest_int('mO', mOS[0], mOS[1]) 
                if mO > nI: raise TrialPruned()
                
            X, Y, Z = PrepareData(RS, HI, nI, nO, mO, 1, mode)
            ref, J = RefEoL(sig, EOL, nI, mO, mode)
            if len(X) < J: raise TrialPruned()   # o fluxo não alcança o EoL (não deve ocorrer com nOS limitado)
            
            # Predição pontual ('single'): a predição pontual é idêntica à de 'interval' (rho só define os limites)
            nI2 = len(Y[0]); lim = len(sig) - nI2 + 1          # mesmo lim da célula 1 (nI = len(Y[0]))
            teda=AutoCloud(m=m,nI=nI2,nR=nR,nO=nO+mO,ηS=[N1,N2,N3],mode=mode,act=act,
                           tau=τ,eol=EOL,ref=ref,wtaG=wtaG,wtaP=wtaP,feedback=feedback,mO=mO,
                           rho=rho,lam=LAM,intr_state=intr_state,rho_dyn=False)
            acc = np.zeros(4)
            for j,_ in enumerate(X[:J]):
                teda.run(X[j])
                teda.RUL_Prediction(Y[j],mode='single',lim=lim)
                teda.Adapt(Y[j],Z[j])
                wR, wH = WAPE_Aligned(teda, acc)

                if not np.isfinite(teda.hiP[-1]): raise TrialPruned()
                if not OptPrune: continue

                if j >=40 and j%5==0:
                    if wR > 0.6: raise TrialPruned()
                    if wH > 0.3: raise TrialPruned()

                if SingleObj and j%35==0:
                    trial.report(wR+wH, j)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if SingleObj: return wR + wH
            else: return wR, wH

        # Sessões independentes: sementes distintas quando OptSeed é informado
        seed = None if OptSeed is None else OptSeed + i
        pruner=optuna.pruners.HyperbandPruner()
        if SingleObj: study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                                  direction='minimize',pruner=pruner,sampler=SelSampler(mode=OptSampler,seed=seed))
        else: study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                          directions=['minimize','minimize'],sampler=SelSampler(mode=OptSampler,seed=seed))
        n_prev, done = len(study.trials), Complete(study)

        if OptWarm:
            # Começa pelo melhor trial das outras sessões desta configuração
            others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
            cand = [t for n in others for t in Complete(optuna.load_study(study_name=n, storage=storage))]
            if cand:
                best = min(cand, key=Score)
                study.enqueue_trial(best.params, skip_if_exists=True)
                print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {Score(best):.4f}).')

        # Parada antecipada conta trials sem superar o melhor resultado já armazenado
        # Contexto exibido acima de BEST TRIAL: nº de camadas, sessão, estudo, configuração e progresso
        callback = EarlyStoppingCallback(patience=patience, context={
            'nL': nLS, 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
            'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout,
            'storage': storage, 'prefix': f'obj{1 if SingleObj else 2}_'})   # melhor trial por nL (todas as sessões/fases)
        if done:
            best_prev = min(done, key=Score)
            callback.best_score = Score(best_prev)
            callback.best_number = best_prev.number
            callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {Score(best_prev):.4f} | Params: {best_prev.params}'
        print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else ''))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])

        done = Complete(study)
        summary.append((study_name, n_prev, len(study.trials), min(map(Score, done)) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor {best:.4f}')

    # Tabela de resultados reconstruída do banco: todos os trials concluídos deste tipo de objetivo
    # (todas as sessões, fases e nº de camadas), mantendo para cada nL os n_best menores erros
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(f'obj{1 if SingleObj else 2}_'): continue
        for t in Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['mode'],p['act']]
            rows.append(list(t.values[:1 if SingleObj else 2]) + prm + [p['n_layers']])
    df_all = pd.DataFrame(rows, columns=names + ['_nL'])
    df_all['_err'] = df_all[names[0]] if SingleObj else df_all[names[0]] + df_all[names[1]]
    # Mesmo conjunto de parâmetros conta uma vez (mantém o menor erro)
    df_all = df_all.sort_values('_err').drop_duplicates(subset=names[1 if SingleObj else 2:])
    df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best)
    df_stdy = df_stdy.sort_values(['_nL', '_err'], ascending=[True, False])

    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas ({out_path})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp["_err"].min():.4f} | pior mantido {grp["_err"].max():.4f}')

    df_stdy = df_stdy.drop(columns=['_nL', '_err']).reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy

lS,lE = 3,4
for nL in range(lS,lE+1):
    df = Optimize(FileName=FileName, OptDim=1, OptSampler='random', OptSeed=100*nL, OptPrune=False,
                  n_study=3, n_trials=3e3, timeout=1200, patience=1000,
                  mS=[2.0,4.5], nLS=[nL,nL], nRS=[1,70], nOS=[1,9], mOS=[0,10], mdS=[0,1], actS=[0,1],
                  feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, OptReset=False)