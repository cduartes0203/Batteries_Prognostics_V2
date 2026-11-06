'''
Otimização do modelo (Optimize, Functions/TedaOptimize_V4_5.py) executada fora dos notebooks.
Uso (na pasta principal):  .venv\Scripts\python.exe TedaRNN_Opt.py --instancia principal --lS 3 --lE 4
                           (ou TedaRNN_Opt.bat com os mesmos argumentos; --help lista os argumentos)
Resultados em Optimization/TedaRNN_Opt_<instancia>_<feedback>_G<wtaG>P<wtaP>/.
'''
import os
import argparse
import pandas as pd
from Functions.TedaOptimize_V4_5 import Optimize

parser = argparse.ArgumentParser(description='Otimização do modelo AutoCloud_V4_5 (Optimize), um nº de camadas por vez.')
parser.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
parser.add_argument("--lS", type=int, default=3, help="menor nº de camadas testado (padrão: 3)")
parser.add_argument("--lE", type=int, default=4, help="maior nº de camadas testado (padrão: 4)")
args = parser.parse_args()
if not 1 <= args.lS <= args.lE:
    parser.error('requer 1 <= lS <= lE')

FileName = os.path.splitext(os.path.basename(__file__))[0] + "_" + args.instancia

# Configuração do experimento
FEEDBACK = 'symmetric'      # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = False, False   # configuração principal: soma ponderada dos grânulos (eq. 36)
EOL = 0.3                   # limiar de fim de vida (EoL) do HI
W = 1                       # janela do DSI em cada X_k (nº de amostras do RS; PrepareData/WindowDSI)
LAM = 1.0                   # fator de esquecimento (1.0 = todos os passos com o mesmo peso)

RS = pd.read_excel(r'Dataset\RS.xlsx')
HI = pd.read_excel(r'Dataset\HI.xlsx')

if __name__ == '__main__':
    lS,lE = args.lS, args.lE
    for nL in range(lS,lE+1):
        df = Optimize(RS=RS, HI=HI, eol=EOL, w=W, lam=LAM,
                      FileName=FileName, OptDim=1, OptSampler='random', OptSeed=100*nL, OptPrune=False,
                      n_study=3, n_trials=3e3, timeout=1200, patience=1000,
                      mS=[2.0,4.5], nLS=[nL,nL], nRS=[1,70], nOS=[1,9], mOS=[0,10], actS=[0,1],
                      feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, OptReset=False)
