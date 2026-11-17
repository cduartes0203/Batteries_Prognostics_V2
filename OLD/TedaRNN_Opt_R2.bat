@echo off
rem Executa a otimizacao em sequencia de rolamentos (TedaRNN_Opt_R2.py) com o Python do .venv, a partir da pasta principal.
rem Duplo clique (sem argumentos): pede cada parametro (Enter = valor padrao) e confirma antes de iniciar.
rem Com argumentos (terminal): repassados ao script, ex.:
rem     TedaRNN_Opt_R2.bat --bearings 1_1 1_2 --transfer adapt --instancia teste1 --sampler tpe --n_trials 500
rem Ajuda: TedaRNN_Opt_R2.bat --help
setlocal
cd /d "%~dp0"

if not "%~1"=="" (
    ".venv\Scripts\python.exe" TedaRNN_Opt_R2.py %*
    goto fim
)

set "BRG=1_1 1_2"
set "TRF=adapt"
set "STG=all"
set "INST=principal"
set "SAMP=random"
set "PRUNE=0"
set "NTR=5000"
set "RTR=300"
set "MS=1.65 2.05"
set "NLS=3 4"
set "NOS=1 9"
set "MOS=0 10"
set "OK="
echo Otimizacao TedaRNN_Opt_R2 - sequencia de rolamentos  (Enter mantem o valor entre colchetes; faixas = "minimo maximo")
echo.
set /p "BRG=Rolamentos, em ordem (ex.: 1_1 1_2) [%BRG%]: "
set /p "TRF=Transferencia: adapt (continua aprendendo) ou freeze (congela apos o 1o) [%TRF%]: "
set /p "STG=Etapa: all, model, rho, plot [%STG%]: "
set /p "INST=Instancia (nome da pasta de resultados) [%INST%]: "
set /p "SAMP=OptSampler: random, tpe, gp, nsgaii, auto, none [%SAMP%]: "
set /p "PRUNE=OptPrune: 1 = podar trials ruins, 0 = nao [%PRUNE%]: "
set /p "NTR=n_trials por sessao e por n. de camadas (etapa model) [%NTR%]: "
set /p "RTR=rho_trials (etapa rho) [%RTR%]: "
set /p "MS=mS, faixa de m [%MS%]: "
set /p "NLS=nLS, faixa do n. de camadas [%NLS%]: "
set /p "NOS=nOS, faixa de nO (valores futuros de Z_k) [%NOS%]: "
set /p "MOS=mOS, faixa de mO (valores passados de Z_k) [%MOS%]: "
echo.
echo   rolamentos = %BRG%
echo   transfer   = %TRF%
echo   etapa      = %STG%
echo   instancia  = %INST%   (resultados em Optimization\TedaRNN_Opt_R2_%INST%_... e IMAGES\)
echo   OptSampler = %SAMP%
echo   OptPrune   = %PRUNE%
echo   n_trials   = %NTR%
echo   rho_trials = %RTR%
echo   mS         = %MS%
echo   nLS        = %NLS%
echo   nOS        = %NOS%
echo   mOS        = %MOS%
set /p "OK=Iniciar? (S/N): "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
echo.
".venv\Scripts\python.exe" TedaRNN_Opt_R2.py --bearings %BRG% --transfer %TRF% --stage %STG% --instancia "%INST%" --sampler %SAMP% --prune %PRUNE% --n_trials %NTR% --rho_trials %RTR% --mS %MS% --nLS %NLS% --nOS %NOS% --mOS %MOS%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
