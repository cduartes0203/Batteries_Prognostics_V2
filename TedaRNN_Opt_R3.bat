@echo off
rem Executa a otimizacao em sequencia de rolamentos (TedaRNN_Opt_R3.py, AutoCloud_V4_6 com restart) com o Python do .venv.
rem Duplo clique (sem argumentos): pede cada parametro (Enter = valor padrao) e confirma antes de iniciar.
rem Com argumentos (terminal): repassados ao script, ex.:
rem     TedaRNN_Opt_R3.bat --bearings 1_1 1_2 --instancia teste1 --sampler tpe --n_trials 500 --nLS 1 3
rem Ajuda: TedaRNN_Opt_R3.bat --help
setlocal
cd /d "%~dp0"

if not "%~1"=="" (
    ".venv\Scripts\python.exe" -X utf8 TedaRNN_Opt_R3.py %*
    goto fim
)

set "BRG=1_1 1_2"
set "STATE=keep"
set "AGG=mean"
set "STG=all"
set "INST=principal"
set "SAMP=tpe"
set "SEED=1"
set "PRUNE=1"
set "NST=1"
set "NTR=1000"
set "PAT=500"
set "TMO=3600"
set "MS=1.75 2.05"
set "NLS=1 3"
set "NIS=2 40"
set "NRS=1 60"
set "NOS=1 40"
set "MOS=0 40"
set "RHO=0.1"
set "IST=carry"
set "OK="
echo Otimizacao TedaRNN_Opt_R3 - sequencia de rolamentos com restart  (Enter mantem o valor entre colchetes; faixas = "minimo maximo")
echo.
set /p "BRG=Rolamentos, em ordem (ex.: 1_1 1_2) [%BRG%]: "
set /p "STATE=Estado das RNNs no inicio de cada rolamento seguinte: keep, reset, fixed [%STATE%]: "
set /p "AGG=Objetivo: mean (media) ou sum (soma) de WAPE_RUL+WAPE_HI dos rolamentos [%AGG%]: "
set /p "STG=Etapa: all, model, plot [%STG%]: "
set /p "INST=Instancia (nome da pasta de resultados) [%INST%]: "
set /p "SAMP=OptSampler: random, tpe, gp, nsgaii, auto, none [%SAMP%]: "
set /p "SEED=OptSeed (semente base) [%SEED%]: "
set /p "PRUNE=OptPrune: 1 = podar trials ruins, 0 = nao [%PRUNE%]: "
set /p "NST=n_study: sessoes por n. de camadas [%NST%]: "
set /p "NTR=n_trials por sessao e por n. de camadas [%NTR%]: "
set /p "PAT=patience: trials sem melhora para parar [%PAT%]: "
set /p "TMO=timeout por sessao, em segundos [%TMO%]: "
set /p "MS=mS, faixa de m [%MS%]: "
set /p "NLS=nLS, faixa do n. de camadas [%NLS%]: "
set /p "NIS=nIS, faixa de nI [%NIS%]: "
set /p "NRS=nRS, faixa de neuronios por camada [%NRS%]: "
set /p "NOS=nOS, faixa de nO (valores futuros de Z_k) [%NOS%]: "
set /p "MOS=mOS, faixa de mO (valores passados de Z_k) [%MOS%]: "
set /p "RHO=rho estatico da simulacao final (etapa plot) [%RHO%]: "
set /p "IST=intr_state da RUL intervalar (etapa plot): carry, reset [%IST%]: "
echo.
echo   rolamentos = %BRG%
echo   state      = %STATE%
echo   agg        = %AGG%
echo   etapa      = %STG%
echo   instancia  = %INST%   (resultados em Optimization\TedaRNN_Opt_R3_%INST%_... e IMAGES\)
echo   OptSampler = %SAMP%   OptSeed = %SEED%   OptPrune = %PRUNE%
echo   n_study    = %NST%   n_trials = %NTR%   patience = %PAT%   timeout = %TMO%
echo   mS         = %MS%
echo   nLS        = %NLS%
echo   nIS        = %NIS%
echo   nRS        = %NRS%
echo   nOS        = %NOS%
echo   mOS        = %MOS%
echo   rho        = %RHO%   intr_state = %IST%
set /p "OK=Iniciar? (S/N): "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
echo.
".venv\Scripts\python.exe" -X utf8 TedaRNN_Opt_R3.py --bearings %BRG% --state %STATE% --agg %AGG% --stage %STG% --instancia "%INST%" --sampler %SAMP% --seed %SEED% --prune %PRUNE% --n_study %NST% --n_trials %NTR% --patience %PAT% --timeout %TMO% --mS %MS% --nLS %NLS% --nIS %NIS% --nRS %NRS% --nOS %NOS% --mOS %MOS% --rho %RHO% --intr_state %IST%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
