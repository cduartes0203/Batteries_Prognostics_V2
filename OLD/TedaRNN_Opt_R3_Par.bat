@echo off
rem Otimizacao em sequencia de rolamentos (TedaRNN_Opt_R3.py) com UMA JANELA POR FAIXA de n. de camadas, em paralelo.
rem Ex.: faixas "1-2 3-4 5-6" -> tres janelas: --nLS 1 2, --nLS 3 4 e --nLS 5 6, com os mesmos demais parametros.
rem Todas as janelas usam a mesma pasta de resultados e o mesmo optuna.db (estudos distintos por n. de camadas).
rem Esta janela (controle) espera todas terminarem, gera o Optimization.csv uma unica vez (--stage table) e,
rem se pedido, executa a etapa plot com o melhor modelo entre todas as faixas.
rem Uso: duplo clique (pede cada parametro; Enter = valor padrao).
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if "%~1"=="__worker" goto worker

set "BRG=1_1 1_2"
set "FAIXAS=1-2 3-4 5-6"
set "STATE=keep"
set "AGG=mean"
set "INST=par"
set "SAMP=random"
set "SEED=1"
set "PRUNE=1"
set "NST=1"
set "NTR=1000"
set "PAT=500"
set "TMO=3600"
set "MS=1.75 2.05"
set "NIS=2 40"
set "NRS=1 75"
set "NOS=1 40"
set "MOS=0 40"
set "PLOT=S"
set "RHO=0.1"
set "IST=carry"
set "OK=S"
echo Otimizacao TedaRNN_Opt_R3 em paralelo - uma janela por faixa de n. de camadas
echo (Enter mantem o valor entre colchetes; faixas de busca = "minimo maximo")
echo.
set /p "BRG=Rolamentos, em ordem (ex.: 1_1 1_2) [%BRG%]: "
set /p "FAIXAS=Faixas de n. de camadas, uma por janela, no formato min-max separadas por espaco [%FAIXAS%]: "
set /p "STATE=Estado das RNNs no inicio de cada rolamento seguinte: keep, reset, fixed [%STATE%]: "
set /p "AGG=Objetivo: mean (media) ou sum (soma) de WAPE_RUL+WAPE_HI dos rolamentos [%AGG%]: "
set /p "INST=Instancia (nome da pasta de resultados, comum a todas as janelas) [%INST%]: "
set /p "SAMP=OptSampler: random, tpe, gp, nsgaii, auto, none [%SAMP%]: "
set /p "SEED=OptSeed (semente base) [%SEED%]: "
set /p "PRUNE=OptPrune: 1 = podar trials ruins, 0 = nao [%PRUNE%]: "
set /p "NST=n_study: sessoes por n. de camadas [%NST%]: "
set /p "NTR=n_trials por sessao e por n. de camadas [%NTR%]: "
set /p "PAT=patience: trials sem melhora para parar [%PAT%]: "
set /p "TMO=timeout por sessao, em segundos [%TMO%]: "
set /p "MS=mS, faixa de m [%MS%]: "
set /p "NIS=nIS, faixa de nI [%NIS%]: "
set /p "NRS=nRS, faixa de neuronios por camada [%NRS%]: "
set /p "NOS=nOS, faixa de nO (valores futuros de Z_k) [%NOS%]: "
set /p "MOS=mOS, faixa de mO (valores passados de Z_k) [%MOS%]: "
set /p "PLOT=Executar a etapa plot ao fim de todas as janelas? (S/N) [%PLOT%]: "
if /i "%PLOT%"=="S" (
    set /p "RHO=rho estatico da simulacao final [!RHO!]: "
    set /p "IST=intr_state da RUL intervalar: carry, reset [!IST!]: "
)

rem Valida as faixas (min-max, inteiros, 1 <= min <= max) antes de abrir as janelas
set "NJ=0"
for %%I in (%FAIXAS%) do (
    set "A=" & set "B="
    for /f "tokens=1,2 delims=-" %%a in ("%%I") do (set "A=%%a" & set "B=%%b")
    set /a "A2=!A!+0, B2=!B!+0" 2>nul
    if "!B!"=="" (echo Faixa invalida: %%I ^(use min-max, ex.: 1-2^) & goto fim)
    if not "!A2!"=="!A!" (echo Faixa invalida: %%I & goto fim)
    if not "!B2!"=="!B!" (echo Faixa invalida: %%I & goto fim)
    if !A2! LSS 1 (echo Faixa invalida: %%I ^(min ^>= 1^) & goto fim)
    if !A2! GTR !B2! (echo Faixa invalida: %%I ^(min ^<= max^) & goto fim)
    set /a NJ+=1
)
if %NJ%==0 (echo Nenhuma faixa informada. & goto fim)

set "COMMON=--bearings %BRG% --state %STATE% --agg %AGG% --instancia "%INST%" --sampler %SAMP% --seed %SEED% --prune %PRUNE% --n_study %NST% --n_trials %NTR% --patience %PAT% --timeout %TMO% --mS %MS% --nIS %NIS% --nRS %NRS% --nOS %NOS% --mOS %MOS%"

echo.
echo   rolamentos = %BRG%
echo   faixas     = %FAIXAS%   (%NJ% janelas)
echo   state      = %STATE%   agg = %AGG%
echo   instancia  = %INST%   (resultados em Optimization\TedaRNN_Opt_R3_%INST%_... e IMAGES\)
echo   OptSampler = %SAMP%   OptSeed = %SEED%   OptPrune = %PRUNE%
echo   n_study    = %NST%   n_trials = %NTR%   patience = %PAT%   timeout = %TMO%
echo   mS = %MS%   nIS = %NIS%   nRS = %NRS%   nOS = %NOS%   mOS = %MOS%
if /i "%PLOT%"=="S" (echo   plot ao fim: rho = %RHO%   intr_state = %IST%) else (echo   plot ao fim: nao)
set /p "OK=Iniciar? (S/N): "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)

rem Pasta de controle desta execucao: <faixa>.lock (aberto pela janela enquanto roda) e <faixa>.done (fim normal)
set "CTRL=%TEMP%\TedaRNN_Opt_R3_Par_%RANDOM%%RANDOM%"
mkdir "%CTRL%"

echo.
for %%I in (%FAIXAS%) do (
    for /f "tokens=1,2 delims=-" %%a in ("%%I") do (
        set "WARGS=%COMMON% --stage model --table 0 --nLS %%a %%b"
        set "WLOCK=%CTRL%\L%%I.lock"
        set "WDONE=%CTRL%\L%%I.done"
        set "WTITLE=TedaRNN_Opt_R3 %INST% - camadas %%a a %%b"
        echo Abrindo janela: camadas %%a a %%b
        start "!WTITLE!" cmd /c ""%~f0" __worker"
        timeout /t 5 /nobreak >nul
    )
)

echo.
echo Aguardando as %NJ% janelas (verificacao a cada 30 s; Ctrl+C cancela apenas esta espera)...
:espera
timeout /t 30 /nobreak >nul
set "RODANDO=0"
for %%I in (%FAIXAS%) do (
    if not exist "%CTRL%\L%%I.done" (
        if not exist "%CTRL%\L%%I.lock" (
            set /a RODANDO+=1
        ) else (
            2>nul ( 9>>"%CTRL%\L%%I.lock" (call ) ) && (
                echo Aviso: a janela das camadas %%I terminou sem concluir ^(fechada ou com erro^).
                echo interrompida> "%CTRL%\L%%I.done"
            ) || set /a RODANDO+=1
        )
    )
)
if %RODANDO% GTR 0 goto espera

echo.
echo Todas as janelas terminaram. Gerando o Optimization.csv a partir do banco comum...
".venv\Scripts\python.exe" -X utf8 TedaRNN_Opt_R3.py %COMMON% --stage table
if errorlevel 1 goto limpa
if /i "%PLOT%"=="S" (
    echo.
    ".venv\Scripts\python.exe" -X utf8 TedaRNN_Opt_R3.py %COMMON% --stage plot --rho %RHO% --intr_state %IST%
)

:limpa
rmdir /s /q "%CTRL%" 2>nul
goto fim

:worker
rem Janela de uma faixa: roda a etapa model com o arquivo .lock aberto (liberado automaticamente se a janela for fechada)
title %WTITLE%
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
call :roda 9>"%WLOCK%"
echo.
echo Fim desta faixa (codigo de saida %RC%). O Optimization.csv sera gerado pela janela de controle.
pause
exit /b

:roda
echo %WTITLE%
echo.
".venv\Scripts\python.exe" -X utf8 TedaRNN_Opt_R3.py %WARGS%
set "RC=%ERRORLEVEL%"
rem .done gravado ainda com o .lock aberto: a janela de controle nunca ve o .lock livre sem o .done
echo %RC%> "%WDONE%"
exit /b

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
