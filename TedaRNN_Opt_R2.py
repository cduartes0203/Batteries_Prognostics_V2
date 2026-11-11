'''
Otimização do modelo AutoCloud_V4_5 em uma SEQUÊNCIA de rolamentos (ex.: Bearing1_1 -> Bearing1_2): o mesmo modelo
assimila o primeiro rolamento até o EoL e prossegue para o seguinte com o conhecimento adquirido.

Etapas (--stage):
    model : OptimizeSeq    -> hiperparâmetros [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (um nº de camadas por vez);
                              objetivo = média, sobre os rolamentos, de WAPE_RUL + WAPE_HI (predições pontuais).
    rho   : OptimizeRhoSeq -> rho_HI dinâmico e modelo do erro para o melhor modelo (V médio do HI).
    plot  : simulação final (melhor modelo + melhor rho) e gráficos de PlotSimulation de cada rolamento em IMAGES/.
    all   : as três, em ordem (padrão).
Transferência entre rolamentos (--transfer):
    adapt  : o modelo continua aprendendo nos rolamentos seguintes (grânulos, RNNs, modelo do erro e rho);
    freeze : o modelo é congelado ao fim do primeiro rolamento (AutoCloud.Freeze).

Uso (na pasta principal):
    .venv\Scripts\python.exe TedaRNN_Opt_R2.py --bearings 1_1 1_2 --transfer adapt --instancia teste1
                                               --sampler tpe --n_trials 3000 --mS 1.5 4.5 --nLS 3 4 --nOS 1 9 --mOS 0 10
    (ou TedaRNN_Opt_R2.bat: sem argumentos pergunta cada valor; --help lista os argumentos)
Dados: Dataset\DSIR1n\B<b>.csv (DSI) e Dataset\HI4\Bearing<b>.csv (HI), b = 1_1, 1_2, ...
Resultados: Optimization\TedaRNN_Opt_R2_<instancia>_<B1_1-B1_2>_<transfer>_<feedback>_G<wtaG>P<wtaP>\
            (optuna.db, mono\Optimization.csv, rho\) e IMAGES\<mesmo nome>_<i>_<rolamento>.png/.pdf
'''
import os
import argparse
import matplotlib
matplotlib.use('Agg')                       # gráficos salvos em arquivo, sem janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Functions.TedaOptimize_V4_5 import (OptimizeSeq, OptimizeRhoSeq, PrepareSeq, BestParams, MakeTeda,
                                         RunTag, SeqTag, GranV)
from Functions.TedaGraphs import PlotSimulation


SAMPLERS = ['random', 'tpe', 'gp', 'nsgaii', 'auto', 'none']

parser = argparse.ArgumentParser(description='Otimização do AutoCloud_V4_5 em uma sequência de rolamentos.')
parser.add_argument("--bearings", nargs='+', default=['1_1', '1_2'], metavar='B',
                    help="rolamentos em ordem, ex.: 1_1 1_2 (padrão: 1_1 1_2)")
parser.add_argument("--transfer", default="adapt", choices=['adapt', 'freeze'],
                    help="adapt: continua aprendendo; freeze: congela após o 1º rolamento (padrão: adapt)")
parser.add_argument("--stage", default="all", choices=['all', 'model', 'rho', 'plot'], help="etapa(s) executada(s) (padrão: all)")
parser.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
parser.add_argument("--sampler", default="random", type=str.lower, choices=SAMPLERS, help="OptSampler (padrão: random)")
parser.add_argument("--prune", default=0, type=int, choices=[0, 1], help="OptPrune: 1 = poda ao fim de cada rolamento (padrão: 0)")
parser.add_argument("--n_trials", default=5000, type=int, help="trials novos por sessão e por nº de camadas, etapa model (padrão: 5000)")
parser.add_argument("--rho_trials", default=300, type=int, help="trials novos da etapa rho (padrão: 300)")
parser.add_argument("--mS", nargs=2, type=float, default=[1.65, 2.05], metavar=('MIN', 'MAX'), help="faixa de m (padrão: 1.65 2.05)")
parser.add_argument("--nLS", nargs=2, type=int, default=[3, 4], metavar=('MIN', 'MAX'), help="faixa do nº de camadas (padrão: 3 4)")
parser.add_argument("--nOS", nargs=2, type=int, default=[1, 9], metavar=('MIN', 'MAX'), help="faixa de nO (padrão: 1 9)")
parser.add_argument("--mOS", nargs=2, type=int, default=[0, 10], metavar=('MIN', 'MAX'), help="faixa de mO (padrão: 0 10)")
args = parser.parse_args()

def check(name, r, low):
    if not low <= r[0] <= r[1]:
        parser.error(f'{name}: requer {low} <= MIN <= MAX (recebido {r[0]} {r[1]})')
check('mS', args.mS, 0.025)
check('nLS', args.nLS, 1)
check('nOS', args.nOS, 1)
check('mOS', args.mOS, 0)
if args.n_trials < 1 or args.rho_trials < 1: parser.error('n_trials e rho_trials devem ser >= 1')
if len(args.bearings) < 2: parser.error('--bearings: informe pelo menos dois rolamentos')
if len(set(args.bearings)) != len(args.bearings): parser.error('--bearings: rolamentos repetidos')

FileName = os.path.splitext(os.path.basename(__file__))[0] + "_" + args.instancia

# Configuração do experimento
FEEDBACK = 'symmetric'      # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = False, False   # configuração principal: soma ponderada dos grânulos (eq. 36)
EOL = 0.3                   # limiar de fim de vida (EoL) do HI
W = 1                       # janela do DSI em cada X_k (nº de amostras do RS; PrepareData/WindowDSI)
LAM = 1.0                   # fator de esquecimento (1.0 = todos os passos com o mesmo peso)
# Etapas rho e plot: RUL intervalar (ver Teda_R01: 'error' + 'sqrt' + 'observed' deu o afunilamento mais claro)
RHO_CFG = dict(Vtarget='HI', rls_mode='pol', intr_state='carry', rho_min=1e-2, rho_max=10,
               rul_bounds='error', rul_growth='sqrt', rul_eclip='observed', rul_frac=False)
IMAGES = 'IMAGES'


def LoadData(bearings):
    data = []
    for b in bearings:
        rs_path, hi_path = os.path.join('Dataset', 'DSIR1n', f'B{b}.csv'), os.path.join('Dataset', 'HI4', f'Bearing{b}.csv')
        for p in (rs_path, hi_path):
            if not os.path.exists(p): raise FileNotFoundError(f'{p} não encontrado (rolamento {b}).')
        data.append((f'B{b}', pd.read_csv(rs_path), pd.read_csv(hi_path)))
    return data


def StageModel(data):
    lS, lE = args.nLS
    for nL in range(lS, lE + 1):
        OptimizeSeq(data=data, FileName=FileName, transfer=args.transfer, eol=EOL, w=W, lam=LAM,
                    OptSampler=args.sampler, OptSeed=100*nL, OptPrune=bool(args.prune),
                    n_study=3, n_trials=args.n_trials, timeout=1200, patience=1000,
                    mS=args.mS, nLS=[nL, nL], nRS=[1, 70], nOS=args.nOS, mOS=args.mOS, actS=[0, 1],
                    feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, OptReset=False)


def StageRho(data):
    OptimizeRhoSeq(data=data, FileName=FileName, transfer=args.transfer, params=None,   # melhor linha de OptimizeSeq
                   OptSampler='tpe', OptSeed=1, n_study=1, n_trials=args.rho_trials, timeout=3600,
                   feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, eol=EOL, w=W, lam=LAM, OptReset=False, **RHO_CFG)


def StagePlot(data):
    tag = RunTag(SeqTag(FileName, [d[0] for d in data], args.transfer), FEEDBACK, WTAG, WTAP)
    params = BestParams(os.path.join('Optimization', tag, 'mono', 'Optimization.csv'))
    prho = pd.read_csv(os.path.join('Optimization', tag, 'rho', 'Optimization.csv')).iloc[0].to_dict()
    print(f'Simulação final | params = {params} | rho = { {k: round(float(v), 5) for k, v in prho.items()} }')
    m, nI, nR, nO, mO = params[:5]
    ds = PrepareSeq(data, W, nI, nO, mO, eol=EOL, frac=RHO_CFG['rul_frac'], strict=True)
    cfg = {k: v for k, v in RHO_CFG.items() if k != 'Vtarget'}
    teda = MakeTeda(params, ds[0]['ref'], prho, eol=EOL, feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, lam=LAM, **cfg)
    os.makedirs(IMAGES, exist_ok=True)
    rows = []

    def on_end(t, name):                    # fim de um rolamento: gráficos e resumo deste rolamento
        i = len(rows) + 1
        for ext in ('png', 'pdf'):
            out = os.path.join(IMAGES, f'{tag}_{i}_{name}.{ext}')
            PlotSimulation(teda=t.PlotView(), mode='plt', out=out, xi=0, ftrs=5, dsi_mode='last3')
            plt.close('all')
        Vr, cr, sr = GranV(t.rulR, t.rulL, t.rulU, max(np.max(t.rulR) - np.min(t.rulR), 1e-9))
        rows.append(dict(ordem=i, rolamento=name, transfer=args.transfer, congelado=t.frozen, passos=len(t.rulP),
                         granulos=len(t.c), WAPE_RUL=t.wape_RUL, WAPE_HI=t.wape_HI, V_HI=t.V, cov_HI=t.cov, sp_HI=t.sp,
                         V_RUL=Vr, cov_RUL=cr, sp_RUL=sr))
        print(f'  {name}: WAPE_RUL={t.wape_RUL:.4f} | WAPE_HI={t.wape_HI:.4f} | V_HI={t.V:.4f} | V_RUL={Vr:.4f} | '
              f'grânulos={len(t.c)} | gráficos em {IMAGES}\\{tag}_{i}_{name}.png/.pdf')

    teda.SimSequence(ds, transfer=args.transfer, mode='interval', update_rls=True, on_end=on_end)
    out_csv = os.path.join(IMAGES, f'{tag}_resumo.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'Resumo: {out_csv}')


if __name__ == '__main__':
    data = LoadData(args.bearings)
    print(f'{FileName}: rolamentos={" -> ".join(d[0] for d in data)} | transfer={args.transfer} | stage={args.stage} | '
          f'sampler={args.sampler} | prune={bool(args.prune)} | n_trials={args.n_trials} | rho_trials={args.rho_trials} | '
          f'mS={args.mS} | nLS={args.nLS} | nOS={args.nOS} | mOS={args.mOS}')
    if args.stage in ('all', 'model'): StageModel(data)
    if args.stage in ('all', 'rho'): StageRho(data)
    if args.stage in ('all', 'plot'): StagePlot(data)
