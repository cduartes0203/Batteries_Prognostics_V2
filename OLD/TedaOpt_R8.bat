@echo off
rem Executa a otimizacao em sequencia de rolamentos (TedaOpt_R8.py, AutoCloud_V4_6 com restart, TedaOptimize_V4_11) com o Python do .venv.
rem Duplo clique (sem argumentos): pede cada parametro (Enter = valor padrao entre colchetes) e confirma antes de iniciar.
rem Ordem: instancia, etapa, objetivo (mode, norm, obj, agg, lims, alpha, eps, lim_prune, ref_win), rolamentos; parametros do Optuna; parametros do modelo TEDA-RNN.
rem Os valores padrao (set abaixo) podem ser editados neste arquivo.
rem Com argumentos (terminal): repassados ao script, ex.:
rem     TedaOpt_R8.bat --bearings 1_1 1_2 --instancia teste1 --sampler tpe --n_trials 500 --nLS 1 3
rem Ajuda (todos os argumentos): TedaOpt_R8.bat --help
rem Com paralelo = 1, esta janela abre workers janelas por par de n. de camadas, espera todas, gera o Optimization.csv
rem e (etapa all) executa a etapa plot. n_trials e dividido entre os workers. Use no maximo ~1 processo por nucleo.
rem Faixas = "minimo maximo". wS = "1 1" equivale a w = 1 fixo. actS: 0 = tanh, 1 = relu, 2 = sigmoid.
setlocal
cd /d "%~dp0"

if not "%~1"=="" (
    ".venv\Scripts\python.exe" -X utf8 TedaOpt_R8.py %*
    goto fim
)

rem ---- Valores padrao: execucao
set "INST=principal"
set "STG=all"
set "MODE=single"
set "NORM=ref"
set "OBJ=cheb"
set "AGG=mean"
set "LIMS=0.3 0.2"
set "ALPHA=0.5"
set "EPS=0.05"
set "LPR=-1"
set "REFW=10"
set "BRG=1_1 1_2"
rem ---- Valores padrao: Optuna (OptimizeSeq)
set "SAMP=random"
set "SEED=1"
set "PRUNE=1"
set "NST=1"
set "NTR=1000"
set "PAT=500"
set "TMO=3600"
set "RESET=0"
set "WARM=0"
set "NBEST=10"
set "PAR=1"
set "WRK=4"
rem ---- Valores padrao: modelo TEDA-RNN (configuracao e faixas de busca)
set "STATE=keep"
set "FB=symmetric"
set "WTAG=0"
set "WTAP=0"
set "EOL=0.3"
set "MS=1.75 2.05"
set "NLS=1 5"
set "NRS=1 75"
set "NIS=2 40"
set "NOS=1 40"
set "MOS=0 40"
set "N1S=1 9"
set "N2S=1 9"
set "N3S=1 9"
set "TAUS=1 50"
set "ACTS=0 1"
set "WS=1 20"
set "RHO=0.1"
set "IST=carry"
set "OK=S"

echo Otimizacao TedaOpt_R8 - sequencia de rolamentos com restart  (Enter mantem o valor entre colchetes; faixas = "minimo maximo")
echo.
echo ---- Execucao
set /p "INST=Instancia (nome da pasta de resultados) [%INST%]: "
set /p "STG=Etapa: all, model, table, plot [%STG%]: "
echo Objetivo: por rolamento, r = WAPE_RUL/a_R e h = WAPE_HI/a_H (norm), erro do rolamento (obj), agregacao sobre os rolamentos (agg)
set /p "MODE=mode: single (um objetivo) ou multi (frente de Pareto RUL x HI; obj e alpha ignorados) [%MODE%]: "
set /p "NORM=norm: ref (referencias), lim (limites lims) ou none (WAPE brutos) [%NORM%]: "
set /p "OBJ=obj: cheb = max(r,h)+eps*(r+h), sum = alpha*r+(1-alpha)*h, prod = r*h [%OBJ%]: "
set /p "AGG=agg: mean (media), sum (soma) ou max (pior rolamento) [%AGG%]: "
set /p "LIMS=lims: limites de WAPE_RUL e WAPE_HI (norm lim e lim_prune) [%LIMS%]: "
set /p "ALPHA=alpha: peso da RUL em obj sum, em [0, 1] [%ALPHA%]: "
set /p "EPS=eps: termo de desempate de obj cheb [%EPS%]: "
set /p "LPR=lim_prune: 1 = podar se WAPE acima de lims ao fim de um rolamento, 0 = nao, -1 = automatico (so norm lim) [%LPR%]: "
set /p "REFW=ref_win: amostras da tendencia linear da RUL de referencia (norm ref; 0 = todo o historico) [%REFW%]: "
set /p "BRG=Rolamentos, em ordem (ex.: 1_1 1_2) [%BRG%]: "
echo.
echo ---- Optuna
set /p "SAMP=OptSampler: random, tpe, gp, nsgaii, auto, none [%SAMP%]: "
set /p "SEED=OptSeed (semente base) [%SEED%]: "
set /p "PRUNE=OptPrune: 1 = podar trials ruins, 0 = nao [%PRUNE%]: "
set /p "NST=n_study: sessoes por n. de camadas [%NST%]: "
set /p "NTR=n_trials por sessao e por n. de camadas [%NTR%]: "
set /p "PAT=patience: trials sem melhora para parar [%PAT%]: "
set /p "TMO=timeout por sessao, em segundos [%TMO%]: "
set /p "RESET=OptReset: 1 = APAGAR os estudos desta instancia antes de otimizar, 0 = continuar [%RESET%]: "
set /p "WARM=OptWarm: 1 = comecar pelo melhor trial das outras sessoes, 0 = nao [%WARM%]: "
set /p "NBEST=n_best: linhas por n. de camadas no Optimization.csv [%NBEST%]: "
set /p "PAR=Janelas em paralelo, por par de n. de camadas (ex.: nLS 1 6 -> 1-2, 3-4, 5-6): 1 = sim, 0 = nao [%PAR%]: "
set /p "WRK=workers: janelas por par de n. de camadas, nos mesmos estudos (com paralelo = 1) [%WRK%]: "
echo.
echo ---- Modelo TEDA-RNN
set /p "STATE=Estado das RNNs no inicio de cada rolamento seguinte: keep, reset, fixed [%STATE%]: "
set /p "FB=feedback: symmetric (gradiente) ou random (ablacao RFLO + DFA) [%FB%]: "
set /p "WTAG=wtaG: 1 = winner-takes-all nos granulos, 0 = soma ponderada (eq. 36) [%WTAG%]: "
set /p "WTAP=wtaP: 1 = winner-takes-all na predicao da RUL, 0 = nao [%WTAP%]: "
set /p "EOL=eol: limiar de fim de vida do HI [%EOL%]: "
set /p "MS=mS, faixa de m (limiar do TEDA) [%MS%]: "
set /p "NLS=nLS, faixa do n. de camadas [%NLS%]: "
set /p "NRS=nRS, faixa de neuronios por camada [%NRS%]: "
set /p "NIS=nIS, faixa de nI (valores passados do HI na entrada) [%NIS%]: "
set /p "NOS=nOS, faixa de nO (valores futuros de Z_k) [%NOS%]: "
set /p "MOS=mOS, faixa de mO (valores passados de Z_k) [%MOS%]: "
set /p "N1S=N1S, faixa de N1 (expoente da taxa de aprendizado 1) [%N1S%]: "
set /p "N2S=N2S, faixa de N2 (expoente da taxa de aprendizado 2) [%N2S%]: "
set /p "N3S=N3S, faixa de N3 (expoente da taxa de aprendizado 3) [%N3S%]: "
set /p "TAUS=tauS, faixa de tau (constante de tempo das RNNs) [%TAUS%]: "
set /p "ACTS=actS, faixa da ativacao (0 = tanh, 1 = relu, 2 = sigmoid) [%ACTS%]: "
set /p "WS=wS, faixa da janela w do DSI em X_k (1 1 = w fixo em 1) [%WS%]: "
set /p "RHO=rho estatico da simulacao final (etapa plot) [%RHO%]: "
set /p "IST=intr_state da RUL intervalar (etapa plot): carry, reset [%IST%]: "
echo.
echo   ---- Execucao
echo   instancia  = %INST%   (resultados em Optimization\TedaOpt_R8_%INST%_... e IMAGES\TedaOpt_R8\, apenas .png)
echo   etapa      = %STG%   rolamentos = %BRG%
echo   objetivo   = mode %MODE%   norm %NORM%   obj %OBJ%   agg %AGG%   lims %LIMS%   alpha %ALPHA%   eps %EPS%   lim_prune %LPR%   ref_win %REFW%
echo   ---- Optuna
echo   OptSampler = %SAMP%   OptSeed = %SEED%   OptPrune = %PRUNE%
echo   n_study    = %NST%   n_trials = %NTR%   patience = %PAT%   timeout = %TMO%
echo   OptReset   = %RESET%   OptWarm = %WARM%   n_best = %NBEST%
echo   paralelo   = %PAR%   workers = %WRK%  (n_trials dividido entre os workers)
echo   ---- Modelo TEDA-RNN
echo   state      = %STATE%   feedback = %FB%   wtaG = %WTAG%   wtaP = %WTAP%   eol = %EOL%
echo   mS         = %MS%   nLS = %NLS%   nRS = %NRS%
echo   nIS        = %NIS%   nOS = %NOS%   mOS = %MOS%
echo   N1S        = %N1S%   N2S = %N2S%   N3S = %N3S%   tauS = %TAUS%   actS = %ACTS%
echo   wS         = %WS%   rho = %RHO%   intr_state = %IST%
set /p "OK=Iniciar? (S/N) [%OK%]: "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
if "%RESET%"=="1" (
    set "OK=N"
    set /p "OK=OptReset = 1 APAGA os trials ja armazenados desta instancia (faixa nLS). Confirma? (S/N) [N]: "
)
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
echo.
".venv\Scripts\python.exe" -X utf8 TedaOpt_R8.py --instancia "%INST%" --stage %STG% --mode %MODE% --norm %NORM% --obj %OBJ% --agg %AGG% --lims %LIMS% --alpha %ALPHA% --eps %EPS% --lim_prune %LPR% --ref_win %REFW% --bearings %BRG% --sampler %SAMP% --seed %SEED% --prune %PRUNE% --n_study %NST% --n_trials %NTR% --patience %PAT% --timeout %TMO% --reset %RESET% --warm %WARM% --n_best %NBEST% --par %PAR% --workers %WRK% --state %STATE% --feedback %FB% --wtaG %WTAG% --wtaP %WTAP% --eol %EOL% --mS %MS% --nLS %NLS% --nRS %NRS% --nIS %NIS% --nOS %NOS% --mOS %MOS% --N1S %N1S% --N2S %N2S% --N3S %N3S% --tauS %TAUS% --actS %ACTS% --wS %WS% --rho %RHO% --intr_state %IST%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
