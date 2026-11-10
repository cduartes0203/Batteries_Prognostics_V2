'''
Otimização do modelo (Optimize, Functions/TedaOptimize_V4_5.py) executada fora dos notebooks.
Uso (na pasta principal):
    .venv\Scripts\python.exe TedaRNN_Opt.py --instancia teste1 --sampler random --prune 0 --n_trials 3000
                                            --mS 2.0 4.5 --nLS 3 4 --nOS 1 9 --mOS 0 10
    (ou TedaRNN_Opt.bat: sem argumentos pergunta cada valor; --help lista os argumentos)
Faixas (mS, nLS, nOS, mOS) são dois valores: mínimo e máximo. Cada nº de camadas em nLS é otimizado separadamente.
Resultados em Optimization/TedaRNN_Opt_<instancia>_<feedback>_G<wtaG>P<wtaP>/.
'''
import os
import argparse
import pandas as pd
from Functions.TedaOptimize_V4_5 import Optimize
from Functions.Utils import *


SAMPLERS = ['random', 'tpe', 'gp', 'nsgaii', 'auto', 'none']

parser = argparse.ArgumentParser(description='Otimização do modelo AutoCloud_V4_5 (Optimize), um nº de camadas por vez.')
parser.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
parser.add_argument("--sampler", default="random", type=str.lower, choices=SAMPLERS, help="OptSampler (padrão: random)")
parser.add_argument("--prune", default=0, type=int, choices=[0, 1], help="OptPrune: 1 = poda de trials ruins (padrão: 0)")
parser.add_argument("--n_trials", default=3000, type=int, help="nº de trials novos por sessão e por nº de camadas (padrão: 3000)")
parser.add_argument("--mS", nargs=2, type=float, default=[1.5, 4.5], metavar=('MIN', 'MAX'), help="faixa de m (padrão: 1.45 4.5)")
parser.add_argument("--nLS", nargs=2, type=int, default=[3, 4], metavar=('MIN', 'MAX'), help="faixa do nº de camadas (padrão: 3 4)")
parser.add_argument("--nOS", nargs=2, type=int, default=[1, 9], metavar=('MIN', 'MAX'), help="faixa de nO, valores futuros de Z_k (padrão: 1 9)")
parser.add_argument("--mOS", nargs=2, type=int, default=[0, 10], metavar=('MIN', 'MAX'), help="faixa de mO, valores passados de Z_k (padrão: 0 10)")
args = parser.parse_args()

def check(name, r, low):
    if not low <= r[0] <= r[1]:
        parser.error(f'{name}: requer {low} <= MIN <= MAX (recebido {r[0]} {r[1]})')
check('mS', args.mS, 0.25)       # m é sugerido em passos de 0.25
check('nLS', args.nLS, 1)
check('nOS', args.nOS, 1)
check('mOS', args.mOS, 0)
if args.n_trials < 1: parser.error('n_trials deve ser >= 1')

FileName = os.path.splitext(os.path.basename(__file__))[0] + "_" + args.instancia

# Configuração do experimento
FEEDBACK = 'symmetric'      # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = False, False   # configuração principal: soma ponderada dos grânulos (eq. 36)
EOL = 0.3                   # limiar de fim de vida (EoL) do HI
W = 1                       # janela do DSI em cada X_k (nº de amostras do RS; PrepareData/WindowDSI)
LAM = 1.0                   # fator de esquecimento (1.0 = todos os passos com o mesmo peso)

RS = pd.read_excel(r'Dataset\RS.xlsx')
HI = pd.read_excel(r'Dataset\HI.xlsx')

path = r'Dataset\DSIR1\Bearing1_1.csv'
RS = pd.read_csv(path).iloc[:,-6:-1]
RS = dfMovingAvg(RS,10)
RS = dfExpMovAvg(RS,0.3)
RS = normalize_df(RS,0,25)
path = r'Dataset\HI1\Bearing1_1.csv'
HI = pd.read_csv(path)

if __name__ == '__main__':
    print(f'{FileName}: sampler={args.sampler} | prune={bool(args.prune)} | n_trials={args.n_trials} | '
          f'mS={args.mS} | nLS={args.nLS} | nOS={args.nOS} | mOS={args.mOS}')
    lS,lE = args.nLS
    for nL in range(lS,lE+1):
        df = Optimize(RS=RS, HI=HI, eol=EOL, w=W, lam=LAM,
                      FileName=FileName, OptDim=1, OptSampler=args.sampler, OptSeed=100*nL, OptPrune=bool(args.prune),
                      n_study=3, n_trials=args.n_trials, timeout=1200, patience=1000,
                      mS=args.mS, nLS=[nL,nL], nRS=[1,70], nOS=args.nOS, mOS=args.mOS, actS=[0,1],
                      feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, OptReset=False)
