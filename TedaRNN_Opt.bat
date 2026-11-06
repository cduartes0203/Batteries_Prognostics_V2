@echo off
rem Executa a otimizacao (TedaRNN_Opt.py) com o Python do .venv, a partir da pasta principal.
rem Duplo clique (sem argumentos): pede instancia, lS e lE (Enter = valor padrao) e confirma antes de iniciar.
rem Com argumentos (terminal):  TedaRNN_Opt.bat --instancia teste1 --lS 2 --lE 5   (ajuda: TedaRNN_Opt.bat --help)
setlocal
cd /d "%~dp0"

if not "%~1"=="" (
    ".venv\Scripts\python.exe" TedaRNN_Opt.py %*
    goto fim
)

set "INST=principal"
set "LS=3"
set "LE=4"
set "OK="
echo Otimizacao TedaRNN_Opt  (Enter mantem o valor entre colchetes)
set /p "INST=Instancia [%INST%]: "
set /p "LS=Menor n. de camadas, lS [%LS%]: "
set /p "LE=Maior n. de camadas, lE [%LE%]: "
echo.
echo   instancia = %INST%   (resultados em Optimization\TedaRNN_Opt_%INST%_...)
echo   camadas   = %LS% a %LE%
set /p "OK=Iniciar a otimizacao? (S/N): "
if /i not "%OK%"=="S" (
    echo Cancelado.
    goto fim
)
echo.
".venv\Scripts\python.exe" TedaRNN_Opt.py --instancia "%INST%" --lS %LS% --lE %LE%

:fim
echo.
echo Fim (codigo de saida %ERRORLEVEL%).
pause
