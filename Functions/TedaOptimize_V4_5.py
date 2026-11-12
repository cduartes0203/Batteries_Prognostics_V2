'''
Otimização (Optuna) do modelo AutoCloud_V4_5, independente dos notebooks TedaOpt_R0x.

    Optimize    : hiperparâmetros do modelo (predições pontuais de HI e RUL, rho fixo); minimiza WAPE_RUL + WAPE_HI.
    OptimizeRho : rho dinâmico e modelo do erro do HI para um modelo fixo; minimiza 1/Vsum.

Os dados (RS, HI) e a configuração do experimento (eol, w, lam, feedback, wtaG, wtaP) são argumentos:
nada é lido de variáveis globais do notebook. Os resultados ficam em Optimization/<FileName>_<feedback>_G<wtaG>P<wtaP>/
(mono/ ou multi/ para Optimize, rho/ para OptimizeRho), com estudos persistentes em SQLite (optuna.db), retomados a cada execução.
'''
import os
import hashlib
import numpy as np
import pandas as pd
import optuna
from optuna.exceptions import TrialPruned
from Functions.AutoCloud_V4_5 import AutoCloud, RefEoL
from Functions.Utils import PrepareData, ParseParams
from Functions.Utils_OPT_R2 import EarlyStoppingCallback, SelSampler


def RunTag(filename=None, feedback='symmetric', wtaG=False, wtaP=False):
    '''Nome da pasta de resultados de uma configuração: <filename>_<feedback>_G<wtaG>P<wtaP>.'''
    return f'{filename}_{feedback}_G{int(wtaG)}P{int(wtaP)}'


def WAPE_Aligned(teda, acc):
    '''
    WAPE ponderado por k^2 (mesma forma de WAPE_RUL/WAPE_HI do AutoCloud), acumulado em acc = [dR, sR, dH, sH].
    Chamar após Adapt: compara a predição do passo k com o valor real do MESMO passo. Retorna (wape_RUL, wape_HI).
    (Mantido para que o objetivo seja idêntico ao dos trials já armazenados.)
    '''
    k = teda.k
    if k > 2:
        t = k**2
        acc[0] += t*abs(teda.rulR[-1] - teda.rulP[-1]); acc[1] += t*abs(teda.rulR[-1])
        acc[2] += t*abs(teda.hiR[-1] - teda.hiP[-1]);   acc[3] += t*abs(teda.hiR[-1])
    wR = acc[0]/acc[1] if acc[1] > 0 else 0.0
    wH = acc[2]/acc[3] if acc[3] > 0 else 0.0
    return wR, wH


def GranV(yR, yL, yU, Range):
    '''V = média(cov*sp): cov = 1 se o valor real está em [yL, yU]; sp = max(0, 1 - largura/Range). Retorna (V, cov, sp).'''
    yR, yL, yU = np.asarray(yR), np.asarray(yL), np.asarray(yU)
    cov = ((yR >= yL) & (yR <= yU)).astype(float)
    sp = np.maximum(0.0, 1.0 - (yU - yL) / Range)
    return (cov * sp).mean(), cov.mean(), sp.mean()


def _Signal(HI):
    sig = np.asarray(HI, dtype=float)
    if sig.ndim > 1 and sig.shape[1] != 1:
        raise ValueError('HI deve ter uma única coluna.')
    return sig.ravel()


def _Complete(study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def Optimize(
        RS=None, HI=None,
        FileName=None,OptDim=None,OptSampler=None,OptSeed=None,OptPrune=False,
        n_study=None,timeout=None,n_trials=None,
        patience=None,
        mS =None,nIS=None,nLS=None,nRS=None,nOS =None,mOS=None,
        N1S=None,N2S=None,N3S=None,τS =None,actS=None,
        feedback='symmetric',wtaG=False,wtaP=False,
        eol=0.3,w=1,lam=1.0,
        OptReset=False,OptWarm=False,n_best=5,
        rho=0.5,intr_state='reset'):

    '''
        Otimiza os hiperparâmetros do modelo [m, nI, nR, nO, mO, N1, N2, N3, τ, act].

        RS, HI : DSI (uma amostra por linha) e HI (uma coluna) do experimento.
        eol    : limiar de EoL do HI;  w: janela do DSI em X_k (PrepareData);  lam: fator de esquecimento do AutoCloud.
        nO / mO: valores futuros / passados (incluindo h_k) de Z_k; h_{k+1} = Z_k[mO] (o antigo modo 'past' é nO = 1).
        Faixas de busca (xxS = [min, max]): mS, nIS, nLS (nº de camadas), nRS (neurônios por camada), nOS, mOS,
            N1S, N2S, N3S (taxas de aprendizado, expoentes), τS, actS.

        OptSampler: None, Auto, GP, NSGAII, Random, TPE
            None/GP/Auto/TPE: suportam otimização multivariada; NSGAII/Random: suporte fraco.
        OptDim: 1 (ou None) -> mono-objetivo (WAPE_RUL + WAPE_HI); > 1 -> multiobjetivo (WAPE_RUL, WAPE_HI).

        feedback, wtaG, wtaP: configuração do modelo. Resultados em Optimization/<FileName>_<feedback>_G<wtaG>P<wtaP>/.
        Armazenamento persistente (SQLite, Optimization/<...>/optuna.db):
            cada uma das n_study sessões é um estudo independente chamado obj<1|2>_L<nLS[0]>-<nLS[1]>_s<i>,
            retomado a cada execução (n_trials = nº de trials NOVOS por execução).
        OptReset: apaga e recria os estudos destas sessões (use após mudar o objetivo/código).
        OptWarm : enfileira o melhor trial das outras sessões da mesma configuração antes de otimizar.
        OptSeed : semente base; a sessão i usa OptSeed + i (sessões independentes).
        n_best  : linhas mantidas por nº de camadas (nL) em mono/ (ou multi/) Optimization.csv. A tabela é reconstruída
                  com TODOS os trials do banco (todas as sessões/fases), mantendo os n_best menores erros de conjuntos de
                  parâmetros distintos por nL; linhas agrupadas por nL (pior -> melhor em cada grupo).

        Objetivo: RUL_Prediction(mode='single') -> apenas predições pontuais de HI e RUL.
            Erro = WAPE_RUL + WAPE_HI (ponderados por k^2), calculados após Adapt (HI do passo k contra o real do passo k).
            rho é fixo (rho, rho_dyn=False) e não afeta as predições pontuais; rho e seus parâmetros são otimizados
            depois por OptimizeRho.
    '''
    if RS is None or HI is None: raise ValueError('Informe RS e HI.')
    sig = _Signal(HI)
    FileName = RunTag(FileName, feedback, wtaG, wtaP)

    SingleObj = OptDim is None or OptDim == 1
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
    if actS is None: actS = [0,1,2]

    # Limita nO para que o fluxo sempre alcance o EoL: a RUL é avaliada até a última amostra antes
    # do EoL em todos os trials (evita favorecer nO grande, que encurta o trecho avaliado)
    nO_max = len(sig) - int(np.argmax(sig <= eol))   # último instante k = K-1-nO >= iEoL-1
    if nOS[1] > nO_max:
        nOS = [nOS[0], nO_max]
        print(f'nO limitado a {nOS} para que o fluxo alcance o EoL.')
    if nOS[0] > nOS[1]: raise ValueError(f'nOS inválido: o limite para alcançar o EoL é nO <= {nO_max}.')

    prm_names = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation']
    names = (['MAPE_RUL+MAPE_HI'] if SingleObj else ['MAPE_RUL', 'MAPE_HI']) + prm_names
    study_dir = os.path.join('Optimization', FileName, 'mono' if SingleObj else 'multi')
    os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')

    # Armazenamento persistente: um banco SQLite por configuração; cada sessão é um estudo independente e retomável
    db_path = os.path.abspath(os.path.join('Optimization', FileName, 'optuna.db'))
    storage = optuna.storages.RDBStorage('sqlite:///' + db_path.replace(os.sep, '/'))
    prefix = f'obj{1 if SingleObj else 2}_'
    base_name = f'{prefix}L{nLS[0]}-{nLS[1]}'
    Score = lambda t: t.value if SingleObj else sum(t.values)
    summary = []

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
        act = trial.suggest_int('act', actS[0], actS[1])
        mO = trial.suggest_int('mO', mOS[0], mOS[1])
        if mO > nI: raise TrialPruned()

        X, Y, Z, kX = PrepareData(RS, HI, w, nI, nO, mO)
        try: ref, J, L = RefEoL(sig, eol, kX, strict=True)
        except ValueError: raise TrialPruned()   # o fluxo não alcança o EoL (não deve ocorrer com nOS limitado)

        # Predição pontual ('single'): idêntica à de 'interval' (rho só define os limites)
        teda=AutoCloud(m=m,nI=nI,nR=nR,nO=nO,ηS=[N1,N2,N3],act=act,
                       tau=τ,eol=eol,ref=ref,wtaG=wtaG,wtaP=wtaP,feedback=feedback,mO=mO,
                       rho=rho,lam=lam,intr_state=intr_state,rho_dyn=False)
        acc = np.zeros(4)
        for j,_ in enumerate(X[:J]):
            teda.run(X[j])
            teda.RUL_Prediction(Y[j],mode='single',lim=L)
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
                    raise TrialPruned()

        if SingleObj: return wR + wH
        else: return wR, wH

    for i in range(n_study):
        study_name = f'{base_name}_s{i+1}'
        print(f'iteration: {i+1} | study: {study_name}')
        if OptReset and study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f'  {study_name}: estudo anterior apagado (OptReset=True).')

        # Sessões independentes: sementes distintas quando OptSeed é informado
        seed = None if OptSeed is None else OptSeed + i
        if SingleObj: study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,direction='minimize',
                                                  pruner=optuna.pruners.HyperbandPruner(),sampler=SelSampler(mode=OptSampler,seed=seed))
        else: study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                          directions=['minimize','minimize'],sampler=SelSampler(mode=OptSampler,seed=seed))
        n_prev, done = len(study.trials), _Complete(study)

        if OptWarm:
            # Começa pelo melhor trial das outras sessões desta configuração
            others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
            cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
            if cand:
                best = min(cand, key=Score)
                study.enqueue_trial(best.params, skip_if_exists=True)
                print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {Score(best):.4f}).')

        # Parada antecipada conta trials sem superar o melhor resultado já armazenado
        callback = EarlyStoppingCallback(patience=patience, context={
            'nL': nLS, 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
            'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout,
            'storage': storage, 'prefix': prefix})   # melhor trial por nL (todas as sessões/fases)
        if done:
            best_prev = min(done, key=Score)
            callback.best_score = Score(best_prev)
            callback.best_number = best_prev.number
            callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {Score(best_prev):.4f} | Params: {best_prev.params}'
        print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else ''))
        study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout, callbacks=[callback])

        done = _Complete(study)
        summary.append((study_name, n_prev, len(study.trials), min(map(Score, done)) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor {best:.4f}')

    # Tabela reconstruída do banco: todos os trials concluídos deste tipo de objetivo
    # (todas as sessões, fases e nº de camadas), mantendo para cada nL os n_best menores erros
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            # trials antigos no modo 'past' (Z_k com nO valores terminando em h_{k+1}) -> nO = 1 futuro e mO = nO - 1 passados
            nO_, mO_ = (1, p['nO'] - 1) if p.get('mode', 1) == 0 else (p['nO'], p['mO'])
            prm = [p['m'],p['nI'],str(nR_list),nO_,mO_,p['N1'],p['N2'],p['N3'],p['τ'],p['act']]
            rows.append(list(t.values[:1 if SingleObj else 2]) + prm + [p['n_layers']])
    df_all = pd.DataFrame(rows, columns=names + ['_nL'])
    df_all['_err'] = df_all[names[0]] if SingleObj else df_all[names[0]] + df_all[names[1]]
    # Mesmo conjunto de parâmetros conta uma vez (mantém o menor erro)
    df_all = df_all.sort_values('_err').drop_duplicates(subset=prm_names)
    df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best)
    df_stdy = df_stdy.sort_values(['_nL', '_err'], ascending=[True, False])

    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas ({out_path})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp["_err"].min():.4f} | pior mantido {grp["_err"].max():.4f}')

    df_stdy = df_stdy.drop(columns=['_nL', '_err']).reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


def _RhoTag(params, w):
    '''
    Identificador do modelo no nome dos estudos de OptimizeRho. Usa a representação antiga da lista de parâmetros
    (com mode = 1, equivalente) para que os estudos já armazenados sejam retomados; w entra só quando w != 1.
    '''
    m, nI, nR, nO, mO, N1, N2, N3, τ, act = params
    py = lambda v: v.item() if isinstance(v, np.generic) else v
    legacy = [py(m), py(nI), [py(n) for n in nR], py(nO), py(mO), py(N1), py(N2), py(N3), py(τ), 1, py(act)]
    if w != 1: legacy = legacy + [w]
    return hashlib.md5(str(legacy).encode()).hexdigest()[:8]


def OptimizeRho(
        RS=None, HI=None,
        FileName=None,params=None,OptSampler=None,OptSeed=None,
        n_study=None,timeout=None,n_trials=None,patience=None,
        rhoS=None,etaS=None,TS=None,betaS=None,nfS=None,ffS=None,dtS=None,
        rho_min=1e-2,rho_max=10,intr_state='reset',Vtarget='HI',rls_mode='pol',
        rul_bounds='network',rul_growth='sqrt',rho_rulS=None,rul_eclip='none',rul_frac=False,
        feedback='symmetric',wtaG=False,wtaP=False,
        eol=0.3,w=1,lam=1.0,
        OptReset=False,OptWarm=False,n_best=5):

    '''
        Otimiza o rho dinâmico (subida de gradiente online em V) e o modelo RLS do erro do HI para um modelo fixo.

        RS, HI : DSI e HI do experimento;  eol, w, lam: como em Optimize.
        params : [m, nI, nR, nO, mO, N1, N2, N3, τ, act] do modelo (o formato antigo, com mode, também é aceito).
                 None: melhor linha de Optimization/<FileName>_<feedback>_G<wtaG>P<wtaP>/mono/Optimization.csv (Optimize).
        Variáveis otimizadas (faixas [min, max]; use [a, a] para fixar):
            rhoS  : rho inicial (log)                  padrão [0.01, 3]
            etaS  : passo da subida de gradiente (log)  padrão [1e-3, 1]
            TS    : temperatura da cobertura suavizada (log, adimensional: escala do escore |e|/e_est)  padrão [0.05, 2]
            betaS : momento (média do gradiente com esquecimento)   padrão [0, 1]
            nfS   : grau do polinômio do RLS do erro do HI (inteiro; só com rls_mode='pol')   padrão [1, 6]
            ffS   : fator de esquecimento do RLS do erro do HI      padrão [0.7, 1.0]
            dtS   : P inicial do RLS do erro do HI (log)            padrão [1e1, 1e6]
        rho_min, rho_max, intr_state: fixos.
        rul_bounds, rul_growth: limites da RUL intervalar (AutoCloud.RULInterv): 'network' (análise intervalar da rede)
                  ou 'error' (predição -/+ rho * erro estimado * g(n), g = 'const' | 'sqrt' | 'lin'). Com 'error' o nome
                  do estudo recebe o sufixo _err<rul_growth> (estudos 'network' mantêm os nomes já armazenados).
        rho_rulS: faixa [min, max] (log) do rho_rul, a escala separada da incerteza da RUL intervalar. None: rho_rul não é
                  otimizado (a RUL usa o rho do HI, como antes). Com rho_rulS o nome do estudo recebe o sufixo _rr e a
                  tabela ganha a coluna rho_rul. Só afeta o objetivo com Vtarget = 'RUL' ou 'HI*RUL'.
        rul_eclip, rul_frac: como em AutoCloud ('observed'/'last': modelo de erro sem extrapolação nas trajetórias da RUL;
                  rul_frac=True: RUL fracionária). Fora do padrão o nome do estudo recebe _ec<modo> e/ou _fr.
                  Não alteram o V do HI.
        rls_mode: modelo do erro do HI: 'pol' (polinômio de grau nf_rls com termo afim) ou 'log'
                  (b0 + b1*ln x + b2*ln^2 x; nf_rls não se aplica e não é otimizado).
        Vtarget: 'HI'     -> V do HI (HI_Granularity, Range = 1; causal)
                 'RUL'    -> V da RUL (Range = max(rulR) - min(rulR); usa a RUL real, conhecida só no fim da vida)
                 'HI*RUL' -> produto dos dois
        Score minimizado pelo Optuna: 1/Vsum, com Vsum = soma de cov*sp nos passos (= V * nº de passos).
            Vtarget='HI': Vsum do HI (teda.Vsum, mesmo valor dos trials já armazenados); 'RUL'/'HI*RUL': V * nº de passos.
            A tabela mostra V (média de cov*sp).
        Armazenamento: Optimization/<...>/rho/optuna.db, um estudo por sessão: rho_<Vtarget>[_pol]_<id do modelo>_s<i>
            (o sufixo _pol só com rls_mode='pol', como nos estudos já armazenados). OptReset/OptWarm/OptSeed como em Optimize.
        n_best: linhas mantidas em Optimization/<...>/rho/Optimization.csv (maiores V, todas as sessões).
                Colunas: V e os parâmetros otimizados (rho, eta, T, beta, [nf_rls,] ff_rls, dt_rls).
                V_HI, cov/sp do HI, V_RUL, cov/sp da RUL, Vsum e rho mediano/final ficam em trial.user_attrs no banco.
        Os pesos da rede e a predição pontual não dependem do rho: só o modelo do erro e o rho mudam os intervalos.
    '''
    if RS is None or HI is None: raise ValueError('Informe RS e HI.')
    if Vtarget not in ('HI', 'RUL', 'HI*RUL'): raise ValueError("Vtarget deve ser 'HI', 'RUL' ou 'HI*RUL'.")
    if rls_mode not in ('pol', 'log'): raise ValueError("rls_mode deve ser 'pol' ou 'log'.")
    sig = _Signal(HI)
    FileName = RunTag(FileName, feedback, wtaG, wtaP)
    if n_study is None: n_study = 1
    if timeout is None: timeout = 600
    if n_trials is None and patience is None: n_trials = 50
    if patience is None: patience = 100 if n_trials is None else max(20, int(0.5*n_trials))

    if rhoS  is None: rhoS  = [0.01, 3.0]
    if etaS  is None: etaS  = [1e-3, 1.0]
    if TS    is None: TS    = [0.05, 2.0]
    if betaS is None: betaS = [0.0, 1.0]
    if nfS   is None: nfS   = [1, 6]
    if ffS   is None: ffS   = [0.7, 1.0]
    if dtS   is None: dtS   = [1e1, 1e6]

    # Modelo fixo: parâmetros informados ou melhor resultado de Optimize (modo mono-objetivo)
    if params is None:
        opt_csv = os.path.join('Optimization', FileName, 'mono', 'Optimization.csv')
        if not os.path.exists(opt_csv): raise FileNotFoundError(f'{opt_csv} não encontrado: informe params ou rode Optimize antes.')
        best = pd.read_csv(opt_csv).sort_values('MAPE_RUL+MAPE_HI').iloc[0]
        params = [best['m'],best['nI'],best['nR'],best['nO'],best['mO'],int(best['N1']),int(best['N2']),int(best['N3']),int(best['TAU'])] \
                 + ([int(best['past/ahead'])] if 'past/ahead' in best.index else []) + [int(best['activation'])]   # tabelas antigas têm mode
        print(f"Modelo: melhor linha de {opt_csv} (MAPE_RUL+MAPE_HI = {best['MAPE_RUL+MAPE_HI']:.4f})")
    params = ParseParams(params)
    m,nI,nR,nO,mO,N1,N2,N3,τ,act = params
    print(f'params = {params}')

    X, Y, Z, kX = PrepareData(RS, HI, w, nI, nO, mO)
    ref, J, L = RefEoL(sig, eol, kX, strict=True, frac=rul_frac)   # erro se o fluxo não alcança o EoL; EoL fracionário com rul_frac

    study_dir = os.path.join('Optimization', FileName, 'rho'); os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')
    db_path = os.path.abspath(os.path.join(study_dir, 'optuna.db'))
    storage = optuna.storages.RDBStorage('sqlite:///' + db_path.replace(os.sep, '/'))
    base_name = (f'rho_{Vtarget.replace("*", "x")}{"_pol" if rls_mode == "pol" else ""}'
                 f'{"_err" + rul_growth if rul_bounds == "error" else ""}{"_rr" if rho_rulS is not None else ""}'
                 f'{"_ec" + rul_eclip if rul_eclip != "none" else ""}{"_fr" if rul_frac else ""}_{_RhoTag(params, w)}')
    if rho_rulS is not None and Vtarget == 'HI':
        print("Aviso: rho_rul não altera o V do HI; use Vtarget='RUL' ou 'HI*RUL' para otimizá-lo.")
    Vof = lambda t: t.user_attrs.get('V', float('nan'))      # V do trial (o score do Optuna é 1/Vsum)
    summary = []

    def objective(trial):
        rho  = trial.suggest_float('rho',  rhoS[0],  rhoS[1],  log=True)
        eta  = trial.suggest_float('eta',  etaS[0],  etaS[1],  log=True)
        T    = trial.suggest_float('T',    TS[0],    TS[1],    log=True)
        beta = trial.suggest_float('beta', betaS[0], betaS[1])
        nf   = trial.suggest_int('nf_rls', int(nfS[0]), int(nfS[1])) if rls_mode == 'pol' else 0
        ff   = trial.suggest_float('ff_rls', ffS[0], ffS[1])
        dt   = trial.suggest_float('dt_rls', dtS[0], dtS[1],   log=True)
        rr   = trial.suggest_float('rho_rul', rho_rulS[0], rho_rulS[1], log=True) if rho_rulS is not None else None

        teda=AutoCloud(m=m,nI=nI,nR=nR,nO=nO,ηS=[N1,N2,N3],act=act,
                       tau=τ,eol=eol,ref=ref,wtaG=wtaG,wtaP=wtaP,feedback=feedback,mO=mO,
                       rho=rho,lam=lam,intr_state=intr_state,
                       rho_dyn=True,rho_eta=eta,rho_T=T,rho_beta=beta,rho_min=rho_min,rho_max=rho_max,
                       rls_mode=rls_mode,nf_rls=nf,ff_rls=ff,dt_rls=dt,rul_bounds=rul_bounds,rul_growth=rul_growth,rho_rul=rr,
                       rul_eclip=rul_eclip,rul_frac=rul_frac)
        for j,_ in enumerate(X[:J]):
            teda.run(X[j])
            teda.RUL_Prediction(Y[j],mode='interval',lim=L)
            teda.Adapt(Y[j],Z[j])
            teda.UpdateRLS(mode='HI')
            if not np.isfinite(teda.hiP[-1]) or not np.isfinite(teda.hiL[-1]) or not np.isfinite(teda.hiU[-1]): raise TrialPruned()

        V_HI, cov_HI, sp_HI = teda.V, teda.cov, teda.sp
        V_RUL, cov_RUL, sp_RUL = GranV(teda.rulR, teda.rulL, teda.rulU, max(np.max(teda.rulR) - np.min(teda.rulR), 1e-9))
        V = {'HI': V_HI, 'RUL': V_RUL, 'HI*RUL': V_HI * V_RUL}[Vtarget]
        Vsum = teda.Vsum if Vtarget == 'HI' else V * len(teda.rulR)
        for k_, v_ in dict(V=V, Vsum=Vsum, V_HI=V_HI, cov_HI=cov_HI, sp_HI=sp_HI, V_RUL=V_RUL, cov_RUL=cov_RUL, sp_RUL=sp_RUL,
                           rho_median=float(np.median(teda.rho_used)), rho_final=teda.rho).items():
            trial.set_user_attr(k_, float(v_))
        if not Vsum > 0: raise TrialPruned()        # nenhum passo coberto: 1/Vsum indefinido
        return 1/Vsum

    for i in range(n_study):
        study_name = f'{base_name}_s{i+1}'
        print(f'iteration: {i+1} | study: {study_name}')
        if OptReset and study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f'  {study_name}: estudo anterior apagado (OptReset=True).')
        seed = None if OptSeed is None else OptSeed + i
        study = optuna.create_study(study_name=study_name,storage=storage,load_if_exists=True,
                                    direction='minimize',sampler=SelSampler(mode=OptSampler,seed=seed))
        n_prev, done = len(study.trials), _Complete(study)

        if OptWarm:
            others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
            cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
            if cand:
                best_t = min(cand, key=lambda t: t.value)
                study.enqueue_trial(best_t.params, skip_if_exists=True)
                print(f'  OptWarm: melhor trial das outras sessões enfileirado (V = {Vof(best_t):.4f}).')

        callback = EarlyStoppingCallback(patience=patience, context={
            'nL': len(nR), 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': FileName,
            'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout})
        if done:
            best_prev = min(done, key=lambda t: t.value)
            callback.best_score, callback.best_number = best_prev.value, best_prev.number
            callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | score: {best_prev.value:.4f} | V: {Vof(best_prev):.4f} | Params: {best_prev.params}'
        print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor V {Vof(best_prev):.4f}' if done else ''))
        study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout, callbacks=[callback])
        done = _Complete(study)
        summary.append((study_name, n_prev, len(study.trials), Vof(min(done, key=lambda t: t.value)) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best_V in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor V {best_V:.4f}')

    # Tabela reconstruída do banco: todos os trials concluídos deste modelo e deste Vtarget (todas as sessões)
    pnames = ['rho', 'eta', 'T', 'beta'] + (['nf_rls'] if rls_mode == 'pol' else []) + ['ff_rls', 'dt_rls'] \
             + (['rho_rul'] if rho_rulS is not None else [])
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(base_name + '_s'): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            rows.append([Vof(t)] + [t.params[k_] for k_ in pnames])
    cols = ['V'] + pnames
    df_rho = pd.DataFrame(rows, columns=cols).sort_values(cols[0], ascending=False).head(n_best).reset_index(drop=True)
    df_rho.to_csv(out_path, index=False)
    print(f'\nOptimization.csv: {len(df_rho)} melhores V ({out_path})')
    return df_rho



# ======================================================================================================================
# Sequência de rolamentos (TedaRNN_Opt_R2): o modelo aprende com um rolamento e prossegue para o seguinte
# ======================================================================================================================
PRM_NAMES = ['m', 'nI', 'nR', 'nO', 'mO', 'N1', 'N2', 'N3', 'TAU', 'activation']


def SeqTag(FileName, names, transfer):
    '''Identificador da configuração de uma sequência: <FileName>_<B1_1-B1_2-...>_<transfer>.'''
    return f'{FileName}_{"-".join(names)}_{transfer}'


def PrepareSeq(data, w, nI, nO, mO, eol=0.3, frac=False, strict=True):
    '''
    data: lista [(nome, RS, HI), ...] na ordem da sequência. Retorna a lista de dicts usada por AutoCloud.SimSequence:
    {'name', 'X', 'Y', 'Z', 'kX', 'J', 'L', 'ref'}, com janelas, EoL, J e L calculados separadamente para cada
    rolamento (nenhuma janela mistura dados de dois rolamentos). strict: erro se um fluxo não alcança o EoL.
    '''
    ds = []
    for name, RS, HI in data:
        X, Y, Z, kX = PrepareData(RS, HI, w, nI, nO, mO)
        ref, J, L = RefEoL(_Signal(HI), eol, kX, strict=strict, frac=frac)
        ds.append(dict(name=name, X=X, Y=Y, Z=Z, kX=kX, J=J, L=L, ref=ref))
    return ds


def BestParams(src):
    '''Melhor linha (menor erro, 1ª coluna) de uma tabela de OptimizeSeq/Optimize (DataFrame ou caminho do CSV),
    como [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (ParseParams).'''
    df = pd.read_csv(src) if isinstance(src, str) else src
    best = df.sort_values(df.columns[0], ascending=True).iloc[0]
    return ParseParams([best[c] for c in PRM_NAMES])


def MakeTeda(params, ref, prho=None, eol=0.3, feedback='symmetric', wtaG=False, wtaP=False, lam=1.0,
             intr_state='reset', rho_min=1e-2, rho_max=10, rls_mode='pol',
             rul_bounds='error', rul_growth='const', rho_rul=None, rul_eclip='none', rul_frac=False):
    '''
    AutoCloud para o modelo params com os parâmetros de rho prho (dict: rho, eta, T, beta, [nf_rls,] ff_rls, dt_rls
    [, rho_rul]; ex. linha de OptimizeRhoSeq). prho=None: rho fixo 0.5, sem rho dinâmico (como em Optimize).
    '''
    m, nI, nR, nO, mO, N1, N2, N3, τ, act = params
    kw = dict(m=m, nI=nI, nR=nR, nO=nO, ηS=[N1, N2, N3], act=act, tau=τ, eol=eol, ref=ref, wtaG=wtaG, wtaP=wtaP,
              feedback=feedback, mO=mO, lam=lam, intr_state=intr_state)
    if prho is None:
        return AutoCloud(rho=0.5, rho_dyn=False, **kw)
    if rho_rul is None and 'rho_rul' in prho and np.isfinite(prho['rho_rul']): rho_rul = float(prho['rho_rul'])
    return AutoCloud(rho=prho['rho'], rho_dyn=True, rho_eta=prho['eta'], rho_T=prho['T'], rho_beta=prho['beta'],
                     rho_min=rho_min, rho_max=rho_max, rls_mode=rls_mode,
                     nf_rls=int(prho['nf_rls']) if rls_mode == 'pol' else None, ff_rls=prho['ff_rls'], dt_rls=prho['dt_rls'],
                     rul_bounds=rul_bounds, rul_growth=rul_growth, rho_rul=rho_rul, rul_eclip=rul_eclip, rul_frac=rul_frac, **kw)


def OptimizeSeq(
        data=None, FileName=None, transfer='adapt',
        OptSampler=None, OptSeed=None, OptPrune=False,
        n_study=None, timeout=None, n_trials=None, patience=None,
        mS=None, nIS=None, nLS=None, nRS=None, nOS=None, mOS=None,
        N1S=None, N2S=None, N3S=None, τS=None, actS=None,
        feedback='symmetric', wtaG=False, wtaP=False,
        eol=0.3, w=1, lam=1.0,
        OptReset=False, OptWarm=False, n_best=5, rho=0.5, intr_state='reset'):
    '''
    Como Optimize, mas para uma sequência de rolamentos simulados com o MESMO modelo (AutoCloud.SimSequence).

    data    : [(nome, RS, HI), ...] na ordem da sequência (ex.: [('B1_1', RS11, HI11), ('B1_2', RS12, HI12)]).
    transfer: 'adapt'   -> o modelo continua aprendendo (grânulos e RNNs) nos rolamentos seguintes (SimSequence);
              'freeze'  -> o modelo é congelado ao fim do primeiro rolamento (AutoCloud.Freeze);
              'restart' -> como em Teda_R00.ipynb: AutoCloud.restart(ref) entre os rolamentos (ver OptimizeSeq2).
    Cada rolamento: janelas, EoL, RUL real e WAPE próprios; termina no EoL; estado recorrente das RNNs reiniciado
    no início de cada rolamento (pesos mantidos). Os hiperparâmetros [m, nI, nR, nO, mO, N1, N2, N3, τ, act] são os
    mesmos em toda a sequência.
    Objetivo (minimizado): média, sobre os rolamentos, de WAPE_RUL + WAPE_HI (ponderados por k^2 locais; predições
    pontuais, rho fixo, como em Optimize). WAPEs de cada rolamento ficam em trial.user_attrs e na tabela.
    Resultados em Optimization/<SeqTag>_<feedback>_G<wtaG>P<wtaP>/ (optuna.db e mono/Optimization.csv), com
    SeqTag = <FileName>_<nomes>_<transfer>: cada sequência/transfer tem estudos próprios.
    Demais argumentos: como em Optimize (apenas mono-objetivo).
    '''
    if not data: raise ValueError('Informe data = [(nome, RS, HI), ...].')
    if transfer not in ('adapt', 'freeze', 'restart'): raise ValueError("transfer deve ser 'adapt', 'freeze' ou 'restart'.")
    names = [d[0] for d in data]
    if len(set(names)) != len(names): raise ValueError(f'Nomes de rolamento repetidos: {names}')
    tag = RunTag(SeqTag(FileName, names, transfer), feedback, wtaG, wtaP)
    if n_study is None: n_study = 1
    if timeout is None: timeout = 60
    if n_trials is None and patience is None: n_trials = 25
    if patience is None: patience = 500 if n_trials < 0.5e4 else int(0.02*n_trials)
    if mS  is None: mS  = [1.75, 4.25]
    if nIS is None: nIS = [2, 40]
    if nLS is None: nLS = [1, 5]
    if nRS is None: nRS = [1, 60]
    if nOS is None: nOS = [1, 40]
    if mOS is None: mOS = [0, 40]
    if N1S is None: N1S = [1, 9]
    if N2S is None: N2S = [1, 9]
    if N3S is None: N3S = [1, 9]
    if τS  is None: τS  = [1, 25]
    if actS is None: actS = [0, 1, 2]

    # nO limitado pelo rolamento mais restritivo: todos os fluxos alcançam o EoL
    nO_max = min(len(_Signal(H)) - int(np.argmax(_Signal(H) <= eol)) for _, _, H in data)
    if nOS[1] > nO_max:
        nOS = [nOS[0], nO_max]
        print(f'nO limitado a {nOS} para que todos os fluxos alcancem o EoL.')
    if nOS[0] > nOS[1]: raise ValueError(f'nOS inválido: o limite para alcançar o EoL é nO <= {nO_max}.')

    study_dir = os.path.join('Optimization', tag, 'mono'); os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')
    db_path = os.path.abspath(os.path.join('Optimization', tag, 'optuna.db'))
    storage = optuna.storages.RDBStorage('sqlite:///' + db_path.replace(os.sep, '/'))
    prefix, base_name = 'obj1_', f'obj1_L{nLS[0]}-{nLS[1]}'
    summary = []
    print(f'Sequência: {" -> ".join(names)} | transfer = {transfer} | resultados em Optimization/{tag}')

    def objective(trial):
        m = trial.suggest_float('m', mS[0], mS[1], step=0.25)
        nI = trial.suggest_int('nI', nIS[0], nIS[1])
        n_layers = trial.suggest_int('n_layers', nLS[0], nLS[1])
        nR = [trial.suggest_int(f'nR_layer_{l}', nRS[0], nRS[1]) for l in range(n_layers)]
        nO = trial.suggest_int('nO', nOS[0], nOS[1])
        N1 = trial.suggest_int('N1', N1S[0], N1S[1])
        N2 = trial.suggest_int('N2', N2S[0], N2S[1])
        N3 = trial.suggest_int('N3', N3S[0], N3S[1])
        τ = trial.suggest_int('τ', τS[0], τS[1])
        act = trial.suggest_int('act', actS[0], actS[1])
        mO = trial.suggest_int('mO', mOS[0], mOS[1])
        if mO > nI: raise TrialPruned()
        try: ds = PrepareSeq(data, w, nI, nO, mO, eol=eol, strict=True)
        except ValueError: raise TrialPruned()

        teda = AutoCloud(m=m, nI=nI, nR=nR, nO=nO, ηS=[N1, N2, N3], act=act, tau=τ, eol=eol, ref=ds[0]['ref'],
                         wtaG=wtaG, wtaP=wtaP, feedback=feedback, mO=mO, rho=rho, lam=lam, intr_state=intr_state, rho_dyn=False)
        errs = []
        def check(t, b, j):
            if not np.isfinite(t.hiP[-1]): raise TrialPruned()
        def on_end(t, name):
            seg = t.segments[name]
            errs.append(seg['wape_RUL'] + seg['wape_HI'])
            trial.set_user_attr(f'WAPE_RUL_{name}', seg['wape_RUL'])
            trial.set_user_attr(f'WAPE_HI_{name}', seg['wape_HI'])
            if OptPrune and len(errs) < len(ds):      # poda com a média parcial ao fim de cada rolamento
                trial.report(float(np.mean(errs)), len(errs))
                if trial.should_prune(): raise TrialPruned()
        if transfer == 'restart':
            # Abordagem de Teda_R00.ipynb: o 1º rolamento cria o modelo; antes de cada rolamento seguinte, restart(ref)
            # (reinicia RUL/HI/WAPE e os registros dos grânulos, mantendo grânulos, RNNs, modelo do erro e rho)
            for b, d in enumerate(ds):
                if b > 0: teda.restart(d['ref'])
                X, Y, Z = d['X'], d['Y'], d['Z']
                for j in range(d['J']):
                    teda.run(X[j])
                    teda.RUL_Prediction(Y[j], mode='single', lim=d['L'])
                    teda.Adapt(Y[j], Z[j])
                    check(teda, b, j)
                teda.segments[d['name']] = dict(name=d['name'], wape_RUL=float(teda.wape_RUL), wape_HI=float(teda.wape_HI))
                on_end(teda, d['name'])
        else:
            teda.SimSequence(ds, transfer=transfer, mode='single', update_rls=False, on_end=on_end, check=check)
        return float(np.mean(errs))

    for i in range(n_study):
        study_name = f'{base_name}_s{i+1}'
        print(f'iteration: {i+1} | study: {study_name}')
        if OptReset and study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f'  {study_name}: estudo anterior apagado (OptReset=True).')
        seed = None if OptSeed is None else OptSeed + i
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='minimize',
                                    pruner=optuna.pruners.HyperbandPruner(), sampler=SelSampler(mode=OptSampler, seed=seed))
        n_prev, done = len(study.trials), _Complete(study)
        if OptWarm:
            others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
            cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
            if cand:
                best = min(cand, key=lambda t: t.value)
                study.enqueue_trial(best.params, skip_if_exists=True)
                print(f'  OptWarm: melhor trial das outras sessões enfileirado (score {best.value:.4f}).')
        callback = EarlyStoppingCallback(patience=patience, context={
            'nL': nLS, 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': tag,
            'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout, 'storage': storage, 'prefix': prefix})
        if done:
            best_prev = min(done, key=lambda t: t.value)
            callback.best_score, callback.best_number = best_prev.value, best_prev.number
            callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | Score: {best_prev.value:.4f} | Params: {best_prev.params}'
        print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor {callback.best_score:.4f}' if done else ''))
        study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout, callbacks=[callback])
        done = _Complete(study)
        summary.append((study_name, n_prev, len(study.trials), min(t.value for t in done) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor {best:.4f}')

    # Tabela: todos os trials concluídos (todas as sessões e nº de camadas), n_best menores erros por nL
    wcols = [f'WAPE_{q}_{n}' for n in names for q in ('RUL', 'HI')]
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(prefix): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            p = t.params
            nR_list = [p[f'nR_layer_{l}'] for l in range(p['n_layers'])]
            prm = [p['m'], p['nI'], str(nR_list), p['nO'], p['mO'], p['N1'], p['N2'], p['N3'], p['τ'], p['act']]
            rows.append([t.value] + prm + [t.user_attrs.get(c, np.nan) for c in wcols] + [p['n_layers']])
    df_all = pd.DataFrame(rows, columns=['WAPE_mean'] + PRM_NAMES + wcols + ['_nL'])
    df_all = df_all.sort_values('WAPE_mean').drop_duplicates(subset=PRM_NAMES)
    df_stdy = df_all.groupby('_nL', group_keys=False).head(n_best).sort_values(['_nL', 'WAPE_mean'], ascending=[True, False])
    print(f'\nOptimization.csv: {n_best} melhores por nº de camadas ({out_path})')
    for nL_, grp in df_stdy.groupby('_nL'):
        print(f'  nL={nL_}: {len(grp)} linhas | melhor {grp["WAPE_mean"].min():.4f} | pior mantido {grp["WAPE_mean"].max():.4f}')
    df_stdy = df_stdy.drop(columns=['_nL']).reset_index(drop=True)
    df_stdy.to_csv(out_path, index=False)
    return df_stdy


def OptimizeSeq2(data=None, FileName=None, **kw):
    '''
    Otimização em sequência de rolamentos com a abordagem de Teda_R00.ipynb (AutoCloud.restart):
      - o 1º rolamento de data cria e treina o modelo (granulação + RNNs) até o EoL;
      - antes de cada rolamento seguinte, teda.restart(ref) reinicia os vetores de RUL e HI (rulP/L/U/R, hiP/L/U/R),
        o WAPE e os registros de amostras dos grânulos, e o tempo/RUL real/pesos do WAPE ficam locais ao rolamento
        (k continua global). Grânulos, pesos das RNNs, estado recorrente, modelo do erro e rho são mantidos, e o
        modelo continua aprendendo (como 'adapt').
    Diferença para OptimizeSeq(transfer='adapt'): restart não reinicia o estado recorrente das RNNs nem os traços de
    elegibilidade (NewSegment os reinicia) e apaga os registros de amostras dos grânulos (cloud.x, cloud.t).
    data: [(nome, RS, HI), ...] na ordem da sequência; cada rolamento usa o próprio HI no EoL (RefEoL).
    Objetivo, faixas de busca, sessões, poda e tabela: como OptimizeSeq (média, sobre os rolamentos, de
    WAPE_RUL + WAPE_HI das predições pontuais; WAPE de cada rolamento na tabela).
    Resultados em Optimization/<FileName>_<nomes>_restart_<feedback>_G<wtaG>P<wtaP>/ (estudos próprios).
    Demais argumentos (kw): os de OptimizeSeq (exceto transfer).
    '''
    if 'transfer' in kw: raise ValueError('OptimizeSeq2 usa sempre restart; para adapt/freeze use OptimizeSeq.')
    return OptimizeSeq(data=data, FileName=FileName, transfer='restart', **kw)


def OptimizeRhoSeq(
        data=None, FileName=None, transfer='adapt', params=None,
        OptSampler=None, OptSeed=None, n_study=None, timeout=None, n_trials=None, patience=None,
        rhoS=None, etaS=None, TS=None, betaS=None, nfS=None, ffS=None, dtS=None,
        rho_min=1e-2, rho_max=10, intr_state='reset', Vtarget='HI', rls_mode='pol',
        rul_bounds='error', rul_growth='const', rho_rulS=None, rul_eclip='none', rul_frac=False,
        feedback='symmetric', wtaG=False, wtaP=False, eol=0.3, w=1, lam=1.0,
        OptReset=False, OptWarm=False, n_best=5):
    '''
    Como OptimizeRho, para a sequência de rolamentos de OptimizeSeq (mesmos data, FileName, transfer).
    params=None: melhor linha de Optimization/<SeqTag>_.../mono/Optimization.csv (OptimizeSeq).
    Score minimizado: 1 / média, sobre os rolamentos, do V do alvo (Vtarget 'HI' | 'RUL' | 'HI*RUL').
    Com transfer='freeze', o modelo do erro e o rho ficam fixos a partir do segundo rolamento (o rho inicial,
    eta, T e beta atuam no primeiro). Resultados em Optimization/<SeqTag>_.../rho/ (V de cada rolamento na tabela).
    '''
    if not data: raise ValueError('Informe data = [(nome, RS, HI), ...].')
    if Vtarget not in ('HI', 'RUL', 'HI*RUL'): raise ValueError("Vtarget deve ser 'HI', 'RUL' ou 'HI*RUL'.")
    if rls_mode not in ('pol', 'log'): raise ValueError("rls_mode deve ser 'pol' ou 'log'.")
    names = [d[0] for d in data]
    tag = RunTag(SeqTag(FileName, names, transfer), feedback, wtaG, wtaP)
    if params is None:
        opt_csv = os.path.join('Optimization', tag, 'mono', 'Optimization.csv')
        if not os.path.exists(opt_csv): raise FileNotFoundError(f'{opt_csv} não encontrado: informe params ou rode OptimizeSeq antes.')
        params = BestParams(opt_csv)
        print(f'Modelo: melhor linha de {opt_csv}')
    params = ParseParams(params)
    m, nI, nR, nO, mO, N1, N2, N3, τ, act = params
    print(f'params = {params}')
    if n_study is None: n_study = 1
    if timeout is None: timeout = 600
    if n_trials is None and patience is None: n_trials = 50
    if patience is None: patience = 100 if n_trials is None else max(20, int(0.5*n_trials))
    if rhoS  is None: rhoS  = [0.01, 3.0]
    if etaS  is None: etaS  = [1e-3, 1.0]
    if TS    is None: TS    = [0.05, 2.0]
    if betaS is None: betaS = [0.0, 1.0]
    if nfS   is None: nfS   = [1, 6]
    if ffS   is None: ffS   = [0.7, 1.0]
    if dtS   is None: dtS   = [1e1, 1e6]
    ds = PrepareSeq(data, w, nI, nO, mO, eol=eol, frac=rul_frac, strict=True)

    study_dir = os.path.join('Optimization', tag, 'rho'); os.makedirs(study_dir, exist_ok=True)
    out_path = os.path.join(study_dir, 'Optimization.csv')
    db_path = os.path.abspath(os.path.join(study_dir, 'optuna.db'))
    storage = optuna.storages.RDBStorage('sqlite:///' + db_path.replace(os.sep, '/'))
    base_name = (f'rho_{Vtarget.replace("*", "x")}{"_pol" if rls_mode == "pol" else ""}'
                 f'{"_err" + rul_growth if rul_bounds == "error" else ""}{"_rr" if rho_rulS is not None else ""}'
                 f'{"_ec" + rul_eclip if rul_eclip != "none" else ""}{"_fr" if rul_frac else ""}_{_RhoTag(params, w)}')
    vcols = [f'V_{n}' for n in names]
    cfg = dict(eol=eol, feedback=feedback, wtaG=wtaG, wtaP=wtaP, lam=lam, intr_state=intr_state, rho_min=rho_min,
               rho_max=rho_max, rls_mode=rls_mode, rul_bounds=rul_bounds, rul_growth=rul_growth,
               rul_eclip=rul_eclip, rul_frac=rul_frac)
    summary = []

    def objective(trial):
        prho = dict(rho=trial.suggest_float('rho', rhoS[0], rhoS[1], log=True),
                    eta=trial.suggest_float('eta', etaS[0], etaS[1], log=True),
                    T=trial.suggest_float('T', TS[0], TS[1], log=True),
                    beta=trial.suggest_float('beta', betaS[0], betaS[1]))
        if rls_mode == 'pol': prho['nf_rls'] = trial.suggest_int('nf_rls', int(nfS[0]), int(nfS[1]))
        prho['ff_rls'] = trial.suggest_float('ff_rls', ffS[0], ffS[1])
        prho['dt_rls'] = trial.suggest_float('dt_rls', dtS[0], dtS[1], log=True)
        if rho_rulS is not None: prho['rho_rul'] = trial.suggest_float('rho_rul', rho_rulS[0], rho_rulS[1], log=True)
        teda = MakeTeda(params, ds[0]['ref'], prho, **cfg)
        Vs = []
        def check(t, b, j):
            if not (np.isfinite(t.hiP[-1]) and np.isfinite(t.hiL[-1]) and np.isfinite(t.hiU[-1])): raise TrialPruned()
        def on_end(t, name):
            V_HI = t.V
            V_RUL = GranV(t.rulR, t.rulL, t.rulU, max(np.max(t.rulR) - np.min(t.rulR), 1e-9))[0]
            V = {'HI': V_HI, 'RUL': V_RUL, 'HI*RUL': V_HI * V_RUL}[Vtarget]
            Vs.append(V)
            for k_, v_ in {f'V_{name}': V, f'V_HI_{name}': V_HI, f'V_RUL_{name}': V_RUL,
                           f'WAPE_RUL_{name}': t.wape_RUL, f'WAPE_HI_{name}': t.wape_HI}.items():
                trial.set_user_attr(k_, float(v_))
        teda.SimSequence(ds, transfer=transfer, mode='interval', update_rls=True, on_end=on_end, check=check)
        V = float(np.mean(Vs)); trial.set_user_attr('V', V)
        if not V > 0: raise TrialPruned()
        return 1/V

    Vof = lambda t: t.user_attrs.get('V', float('nan'))
    for i in range(n_study):
        study_name = f'{base_name}_s{i+1}'
        print(f'iteration: {i+1} | study: {study_name}')
        if OptReset and study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f'  {study_name}: estudo anterior apagado (OptReset=True).')
        seed = None if OptSeed is None else OptSeed + i
        study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                    direction='minimize', sampler=SelSampler(mode=OptSampler, seed=seed))
        n_prev, done = len(study.trials), _Complete(study)
        if OptWarm:
            others = [n for n in optuna.get_all_study_names(storage) if n.startswith(base_name + '_s') and n != study_name]
            cand = [t for n in others for t in _Complete(optuna.load_study(study_name=n, storage=storage))]
            if cand: study.enqueue_trial(min(cand, key=lambda t: t.value).params, skip_if_exists=True)
        callback = EarlyStoppingCallback(patience=patience, context={
            'nL': len(nR), 'session': i+1, 'n_study': n_study, 'study_name': study_name, 'config': tag,
            'n_prev': n_prev, 'n_trials': n_trials, 'timeout': timeout})
        if done:
            best_prev = min(done, key=lambda t: t.value)
            callback.best_score, callback.best_number = best_prev.value, best_prev.number
            callback.best_trial_info = f'Trial #{best_prev.number:03d} (sessões anteriores) | V: {Vof(best_prev):.4f} | Params: {best_prev.params}'
        print(f'  {study_name}: {n_prev} trials armazenados' + (f', melhor V {Vof(best_prev):.4f}' if done else ''))
        study.optimize(objective, n_trials=None if n_trials is None else int(n_trials), timeout=timeout, callbacks=[callback])
        done = _Complete(study)
        summary.append((study_name, n_prev, len(study.trials), Vof(min(done, key=lambda t: t.value)) if done else float('nan')))

    print('\nResumo das sessões (banco: ' + db_path + ')')
    for name, n0, n1, best_V in summary:
        print(f'  {name}: {n0} -> {n1} trials | melhor V médio {best_V:.4f}')
    pnames = ['rho', 'eta', 'T', 'beta'] + (['nf_rls'] if rls_mode == 'pol' else []) + ['ff_rls', 'dt_rls'] \
             + (['rho_rul'] if rho_rulS is not None else [])
    rows = []
    for name in optuna.get_all_study_names(storage):
        if not name.startswith(base_name + '_s'): continue
        for t in _Complete(optuna.load_study(study_name=name, storage=storage)):
            rows.append([Vof(t)] + [t.params[k_] for k_ in pnames] + [t.user_attrs.get(c, np.nan) for c in vcols])
    df_rho = pd.DataFrame(rows, columns=['V'] + pnames + vcols).sort_values('V', ascending=False).head(n_best).reset_index(drop=True)
    df_rho.to_csv(out_path, index=False)
    print(f'\nOptimization.csv: {len(df_rho)} melhores V médios ({out_path})')
    return df_rho
