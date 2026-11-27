@echo off
rem Executa a otimizacao do modelo (OptimizeSeq, etapa model) em sequencia de rolamentos (TedaOpt_R7.py, AutoCloud_V5 com
rem restart, TedaOptimize_V5) com o Python do .venv. Apenas OptimizeSeq: sem etapa plot e sem otimizacao do rho.
rem Este .bat fica em TedaOpt\ e roda da raiz do projeto: cd /d "%~dp0.." entra na pasta acima,
rem onde estao TedaOpt_R7.py, .venv\ e Dataset\ (todos os caminhos do script sao relativos a raiz).
rem Duplo clique (sem argumentos): pede cada parametro (Enter = valor padrao entre colchetes) e confirma antes de iniciar.
rem Ordem: instancia, modo, objetivo, rolamentos; parametros do Optuna; parametros do modelo TEDA-RNN.
rem Os valores padrao (set abaixo) podem ser editados neste arquivo.
rem Com argumentos (terminal): repassados ao script com --stage model, ex.:
rem     TedaOpt\TedaOpt_R7.bat --bearings 1_1 1_2 --instancia teste1 --sampler tpe --n_trials 500 --nLS 1 3
rem Ajuda (todos os argumentos): TedaOpt\TedaOpt_R7.bat --help
rem Com paralelo = 1, esta janela abre workers janelas por n. de camadas, espera todas e gera o Optimization.csv
rem (SeqTable). n_trials e dividido entre os workers. Use no maximo ~1 processo por nucleo.
rem Etapas table e plot: rode TedaOpt_R7.py diretamente (--stage table / --stage plot).
rem Faixas = "minimo maximo". wS = "1 1" equivale a w = 1 fixo. actS: 0 = tanh, 1 = relu, 2 = sigmoid.
setlocal
cd /d "%~dp0.."

if not "%~1"=="" (
    ".venv\Scripts\python.exe" -X utf8 TedaOpt_R7.py --stage model %*
    goto fim
)

rem ---- Valores padrao: execucao
set "INST=T3"
set "MODE=multi"
set "AGG=mean"
set "BRG=1_1 1_2"
rem ---- Valores padrao: Optuna (OptimizeSeq)
set "SAMP=random"
set "SEED=3"
set "PRUNE=0"
set "NST=1"
set "NTR=1000"
set "PAT=1500"
set "TMO=3600"
set "RESET=0"
set "WARM=0"
set "NBEST=8"
set "PAR=1"
set "WRK=1"
rem ---- Valores padrao: modelo TEDA-RNN (configuracao e faixas de busca)
set "STATE=keep"
set "FB=symmetric"
set "WTAG=0"
set "WTAP=0"
set "EOL=0.3"
set "MS=1.75 2.05"
set "NLS=1 4"
set "NRS=10 75"
set "NIS=5 40"
set "NOS=1 10"
set "MOS=0 25"
set "N1S=1 7"
set "N2S=1 9"
set "N3S=1 9"
set "TAUS=1 50"
set "ACTS=0 1"
set "WS=1 20"
set "OK=S"

echo Otimizacao TedaOpt_R7 (OptimizeSeq) - sequencia de rolamentos com restart  (Enter mantem o valor entre colchetes; faixas = "minimo maximo")
echo.
echo ---- Execucao
set /p "INST=Instancia (nome da pasta de resultados) [%INST%]: "
set /p "MODE=mode: single (um objetivo, WAPE_RUL*WAPE_HI) ou multi (F_RUL e F_HI separados, frente de Pareto) [%MODE%]: "
set /p "AGG=Agregacao sobre os rolamentos: mean (media) ou sum (soma) [%AGG%]: "
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
set /p "PAR=Janelas em paralelo, uma por n. de camadas (ex.: nLS 1 6 -> 1, 2, 3, 4, 5, 6): 1 = sim, 0 = nao [%PAR%]: "
set /p "WRK=workers: janelas por n. de camadas, nos mesmos estudos (com paralelo = 1) [%WRK%]: "
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
echo.
echo   ---- Execucao
echo   instancia  = %INST%   (resultados em Optimization\TedaOpt_R7_%INST%_...)
echo   etapa      = model (OptimizeSeq)   mode = %MODE%   agg = %AGG%   rolamentos = %BRG%
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
echo   wS         = %WS%
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
".venv\Scripts\python.exe" -X utf8 TedaOpt_R7.py --instancia "%INST%" --stage model --mode %MODE% --agg %AGG% --bearings %BRG% --sampler %SAMP% --seed %SEED% --prune %PRUNE% --n_study %NST% --n_trials %NTR% --patience %PAT% --timeout %TMO% --reset %RESET% --warm %WARM% --n_best %NBEST% --par %PAR% --workers %WRK% --state %STATE% --feedback %FB% --wtaG %WTAG% --wtaP %WTAP% --eol %EOL% --mS %MS% --nLS %NLS% --nRS %NRS% --nIS %NIS% --nOS %NOS% --mOS %MOS% --N1S %N1S% --N2S %N2S% --N3S %N3S% --tauS %TAUS% --actS %ACTS% --wS %WS%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
