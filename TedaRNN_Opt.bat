@echo off
rem Executa a otimizacao (TedaRNN_Opt.py) com o Python do .venv, a partir da pasta principal.
rem Duplo clique (sem argumentos): pede cada parametro (Enter = valor padrao) e confirma antes de iniciar.
rem Com argumentos (terminal): repassados ao script, ex.:
rem     TedaRNN_Opt.bat --instancia teste1 --sampler tpe --prune 0 --n_trials 500 --mS 2.0 4.5 --nLS 1 2 --nOS 1 9 --mOS 0 10
rem Ajuda: TedaRNN_Opt.bat --help
setlocal
cd /d "%~dp0"

if not "%~1"=="" (
    ".venv\Scripts\python.exe" TedaRNN_Opt.py %*
    goto fim
)

set "INST=principal"
set "SAMP=random"
set "PRUNE=0"
set "NTR=3000"
set "MS=2.0 4.5"
set "NLS=3 4"
set "NOS=1 9"
set "MOS=0 10"
set "OK="
echo Otimizacao TedaRNN_Opt  (Enter mantem o valor entre colchetes; faixas = "minimo maximo")
echo.
set /p "INST=Instancia (nome da pasta de resultados) [%INST%]: "
set /p "SAMP=OptSampler: random, tpe, gp, nsgaii, auto, none [%SAMP%]: "
set /p "PRUNE=OptPrune: 1 = podar trials ruins, 0 = nao [%PRUNE%]: "
set /p "NTR=n_trials por sessao e por n. de camadas [%NTR%]: "
set /p "MS=mS, faixa de m [%MS%]: "
set /p "NLS=nLS, faixa do n. de camadas [%NLS%]: "
set /p "NOS=nOS, faixa de nO (valores futuros de Z_k) [%NOS%]: "
set /p "MOS=mOS, faixa de mO (valores passados de Z_k) [%MOS%]: "
echo.
echo   instancia  = %INST%   (resultados em Optimization\TedaRNN_Opt_%INST%_...)
echo   OptSampler = %SAMP%
echo   OptPrune   = %PRUNE%
echo   n_trials   = %NTR%
echo   mS         = %MS%
echo   nLS        = %NLS%
echo   nOS        = %NOS%
echo   mOS        = %MOS%
set /p "OK=Iniciar a otimizacao? (S/N): "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
echo.
".venv\Scripts\python.exe" TedaRNN_Opt.py --instancia "%INST%" --sampler %SAMP% --prune %PRUNE% --n_trials %NTR% --mS %MS% --nLS %NLS% --nOS %NOS% --mOS %MOS%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
