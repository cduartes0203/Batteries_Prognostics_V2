import optuna
import pandas as pd
import numpy as np
from Functions.AutoCloud_V3 import *
from Functions.Utils import *
from Functions.Graphs import *
from Functions.TedaGraphs import *
from Functions.Utils_OPT import *

RS = pd.read_excel(r'Dataset\RS.xlsx')
HI = pd.read_excel(r'Dataset\HI.xlsx')
sig = HI['PC1'].values

def Optimize(
        FileName=None,OptDim=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,mdS=None,actS=None):

    '''
        OptSampler modes available: None, Auto, GP, NSGAII, Random, TPE
    
        None: sampler default, suport multivariate optimization

        GP: suport multivariate optimization

        Auto: suport multivariate optimization
        
        NSGAII: poorly suport multivariate optimization
        
        Random: poorly suport multivariate optimization

        TPE: suport multivariate optimization
    '''

    if OptDim == 1 or OptDim is None: SingleObj = True
    elif OptDim > 1: SingleObj = False    
    if n_study is None: n_study = 1
    if timeout is None: timeout = 60
    if n_trials is None and patience is None: 
        n_trials = 25
        patience = 25
    elif n_trials < 100 and patience is None:
        patience = int(0.25*n_trials)
    elif n_trials >= 1e2 and n_trials < 1e3 and patience is None:
        patience = int(0.20*n_trials)
    elif n_trials >= 1e3 and n_trials < 5e3 and patience is None:
        patience = int(0.150*n_trials)
    elif n_trials >= 5e3 and n_trials < 1e4 and patience is None:
        patience = int(0.125*n_trials)    
    elif n_trials >= 1e4 and patience is None:
        patience = int(0.03*n_trials)   

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

    if SingleObj: names = ['MAPE_RUL+MAPE_HI', 'm', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3','TAU','past/ahead','activation']
    else: names = ['MAPE_RUL','MAPE_HI', 'm', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3','TAU','past/ahead','activation']
    study_dir, out_path = df_ParamsTable(names,FileName)

    for i in range(n_study):
        print('iteration:',i+1)
        def objective(trial):
            
            m = trial.suggest_float('m', mS[0], mS[1],step=0.25)
            τ = trial.suggest_int('τ', τS[0], τS[1])
            mode = trial.suggest_int('mode', mdS[0], mdS[1])  
            act = trial.suggest_int('act', actS[0], actS[1])
            N1 = trial.suggest_int('N1', N1S[0], N1S[1])
            N2 = trial.suggest_int('N2', N2S[0], N2S[1])
            N3 = trial.suggest_int('N3', N3S[0], N3S[1])
            nI = trial.suggest_int('nI', nIS[0], nIS[1]) 
            nO = trial.suggest_int('nO', nOS[0], nOS[1]) 
            if mode == 'past' or mode == 0:
                mO = trial.suggest_int('mO', 0, 0) 
                if nO > nI: raise TrialPruned()
            elif mode == 'ahead' or mode == 1:
                mO = trial.suggest_int('mO', mOS[0], mOS[1]) 
                if   nI > 20 or nO > 20: raise TrialPruned()
                elif mO > nI: raise TrialPruned()
            nL = trial.suggest_int('nL', nLS[0], nLS[1])
            nR = [trial.suggest_int(f'nL{l}', nRS[0], nRS[1]) for l in range(nL)]

            X, Y, Z = PrepareData(RS, HI, nI, nO, mO, 1, mode)
            
            teda=AutoCloud(m=m,nI=len(Y[0]),nR=nR,nO=nO+mO,ηS=[N1,N2,N3],mode=mode,act=act,
                           tau=τ,rho=0.0,eol=0.3,ref=len(sig)-nI+1,wtaG=True,wtaP=True) 
            for j,_ in enumerate(X[:]):
                teda.run(X[j])
                teda.RUL_Prediction(Y[j],mode='single',lim=len(sig)-nI+1)
                teda.Adapt(Y[j],Z[j])
                #teda.WAPE_HI2(Y[j],Z[j])

                if np.isinf(teda.hiP[-1]) or np.isnan(teda.hiP[-1]): raise TrialPruned()
                if not OptPrune: continue
        
                if j >=60 and j%5==0:
                    if teda.wape_RUL > 0.6: raise TrialPruned()
                    if teda.wape_HI  > 0.3: raise TrialPruned()

                if SingleObj and j%35==0:
                    if np.mean(teda.rulP) == 1: raise TrialPruned()
                    trial.report(teda.wape_RUL+teda.wape_HI, j)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if SingleObj: return teda.wape_RUL + teda.wape_HI
            else: return teda.wape_RUL,teda.wape_HI

        pruner=optuna.pruners.HyperbandPruner()
        if SingleObj: study = optuna.create_study(direction='minimize',pruner=pruner,sampler=SelSampler(mode=OptSampler,seed=OptSeed))
        else: study = optuna.create_study(directions=['minimize','minimize'],sampler=SelSampler(mode=OptSampler,seed=OptSeed))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, 
                       callbacks=[EarlyStoppingCallback(patience=patience)]
                       )

        vec = []
        if SingleObj:
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            for trial in trials:
                p = trial.params
                nR_list = [p[f'nL{l}'] for l in range(p['nL'])]
                row = [trial.values[0],p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['mode'],p['act']]   
                vec.append(row)
            df2 = pd.DataFrame(vec,columns=names)
            df2 = df2.sort_values(by=df2.columns[0], ascending=False)[-5:]

        elif not SingleObj: 
            for trial in study.best_trials:
                p = trial.params
                nR_list = [p[f'nL{l}'] for l in range(p['nL'])]
                row = [trial.values[0],trial.values[1],p['m'],p['nI'],str(nR_list),p['nO'],p['mO'],p['N1'],p['N2'],p['N3'],p['τ'],p['mode'],p['act']]
                vec.append(row)
            df2 = pd.DataFrame(vec,columns=names)
        
    if os.path.isfile(out_path) and out_path.startswith(study_dir):
        df1 = pd.read_csv(out_path)
        df_stdy = pd.concat([df1, df2], ignore_index=True)
    
    else: df_stdy = df2
    
    df_stdy = df_stdy.drop_duplicates()
    df_stdy.to_csv(out_path, index=False)

    if not SingleObj: 
        Sum = df_stdy.iloc[:, 0] + df_stdy.iloc[:, 1]
        df_stdy = df_stdy.loc[Sum.sort_values(ascending=False).index]
    elif SingleObj:
        Sum = df_stdy.iloc[:, 0] + 0
        df_stdy = df_stdy.loc[Sum.sort_values(ascending=False).index]
    
    return df_stdy


