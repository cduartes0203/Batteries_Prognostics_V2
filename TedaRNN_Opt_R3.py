'''
Otimização do modelo AutoCloud_V4_6 (rho estático) em uma SEQUÊNCIA de rolamentos (ex.: Bearing1_1 -> Bearing1_2):
o mesmo modelo assimila o primeiro rolamento até o EoL; antes de cada rolamento seguinte, AutoCloud.restart zera
RUL/HI/WAPE e mantém grânulos, pesos das RNNs e modelo do erro (procedimento de Teda_R03 / TedaOptimize_V4_6.OptimizeSeq).

Etapas (--stage):
    model : OptimizeSeq -> hiperparâmetros [m, nI, nR, nO, mO, N1, N2, N3, τ, act]; para CADA nº de camadas da faixa
            --nLS, n_study sessões de n_trials trials. Objetivo = média (--agg mean) ou soma (--agg sum), sobre os
            rolamentos, de WAPE_RUL + WAPE_HI final de cada rolamento (predições pontuais).
    plot  : simulação final com o melhor modelo (SimSeq, RUL intervalar com --rho e --intr_state) e gráficos de
            PlotSimulation de cada rolamento em IMAGES/, mais um resumo .csv.
    table : apenas reconstrói o mono\Optimization.csv a partir do optuna.db (SeqTable), sem executar trials; usado ao fim
            das execuções em paralelo (TedaRNN_Opt_R3_Par.bat: uma janela por faixa de nº de camadas, mesmo banco).
    all   : model e plot, em ordem (padrão).
Estado recorrente das RNNs no início de cada rolamento seguinte (--state): keep (padrão), reset ou fixed (AutoCloud.restart).

Uso (na pasta principal):
    .venv\\Scripts\\python.exe TedaRNN_Opt_R3.py --bearings 1_1 1_2 --instancia teste1 --sampler tpe
                                               --n_trials 1000 --nLS 1 3 --mS 1.75 2.05
    (ou TedaRNN_Opt_R3.bat: sem argumentos pergunta cada valor; --help lista os argumentos)
Dados: Dataset\\DSIR1n\\B<b>.csv (DSI) e Dataset\\HI1\\Bearing<b>.csv (HI, coluna PC1), b = 1_1, 1_2, ...
Resultados: Optimization\\TedaRNN_Opt_R3_<instancia>_<B1_1-B1_2>_restart[-<state>]_<feedback>_G<wtaG>P<wtaP>\\
            (optuna.db, mono\\Optimization.csv) e IMAGES\\<mesmo nome>_<i>_<rolamento>.png/.pdf, IMAGES\\<...>_resumo.csv
'''
import os
import argparse
import matplotlib
matplotlib.use('Agg')                       # gráficos salvos em arquivo, sem janela
import matplotlib.pyplot as plt
import pandas as pd
from Functions.TedaOptimize_V4_6 import OptimizeSeq, SeqTable, BestParams, SimSeq
from Functions.TedaGraphs import PlotSimulation


SAMPLERS = ['random', 'tpe', 'gp', 'nsgaii', 'auto', 'none']

parser = argparse.ArgumentParser(description='Otimização do AutoCloud_V4_6 em uma sequência de rolamentos (restart).')
parser.add_argument("--bearings", nargs='+', default=['1_1', '1_2'], metavar='B',
                    help="rolamentos em ordem, ex.: 1_1 1_2 (padrão: 1_1 1_2)")
parser.add_argument("--state", default="keep", choices=['keep', 'reset', 'fixed'],
                    help="estado das RNNs no início de cada rolamento seguinte (padrão: keep)")
parser.add_argument("--agg", default="mean", choices=['mean', 'sum'],
                    help="objetivo: média ou soma de WAPE_RUL + WAPE_HI dos rolamentos (padrão: mean)")
parser.add_argument("--stage", default="all", choices=['all', 'model', 'table', 'plot'], help="etapa(s) executada(s) (padrão: all)")
parser.add_argument("--table", default=1, type=int, choices=[0, 1],
                    help="etapa model: 1 = grava o Optimization.csv ao fim; 0 = não grava (execuções em paralelo) (padrão: 1)")
parser.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
parser.add_argument("--sampler", default="tpe", type=str.lower, choices=SAMPLERS, help="OptSampler (padrão: tpe)")
parser.add_argument("--seed", default=1, type=int, help="OptSeed: semente base das sessões (padrão: 1)")
parser.add_argument("--prune", default=1, type=int, choices=[0, 1], help="OptPrune: 1 = poda trials ruins (padrão: 1)")
parser.add_argument("--n_study", default=1, type=int, help="sessões por nº de camadas (padrão: 1)")
parser.add_argument("--n_trials", default=1000, type=int, help="trials novos por sessão e por nº de camadas (padrão: 1000)")
parser.add_argument("--patience", default=500, type=int, help="parada antecipada: trials sem melhora (padrão: 500)")
parser.add_argument("--timeout", default=3600, type=int, help="tempo máximo por sessão, em s (padrão: 3600)")
parser.add_argument("--mS", nargs=2, type=float, default=[1.75, 2.05], metavar=('MIN', 'MAX'), help="faixa de m (padrão: 1.75 2.05)")
parser.add_argument("--nLS", nargs=2, type=int, default=[1, 3], metavar=('MIN', 'MAX'), help="faixa do nº de camadas (padrão: 1 3)")
parser.add_argument("--nIS", nargs=2, type=int, default=[2, 40], metavar=('MIN', 'MAX'), help="faixa de nI (padrão: 2 40)")
parser.add_argument("--nRS", nargs=2, type=int, default=[1, 60], metavar=('MIN', 'MAX'), help="faixa de neurônios por camada (padrão: 1 60)")
parser.add_argument("--nOS", nargs=2, type=int, default=[1, 40], metavar=('MIN', 'MAX'), help="faixa de nO (padrão: 1 40)")
parser.add_argument("--mOS", nargs=2, type=int, default=[0, 40], metavar=('MIN', 'MAX'), help="faixa de mO (padrão: 0 40)")
parser.add_argument("--rho", default=0.1, type=float, help="rho estático da simulação final, etapa plot (padrão: 0.1)")
parser.add_argument("--intr_state", default="carry", choices=['carry', 'reset'],
                    help="estado intervalar da RUL intervalar, etapa plot (padrão: carry)")
args = parser.parse_args()

def check(name, r, low):
    if not low <= r[0] <= r[1]:
        parser.error(f'{name}: requer {low} <= MIN <= MAX (recebido {r[0]} {r[1]})')
check('mS', args.mS, 0.025)
check('nLS', args.nLS, 1)
check('nIS', args.nIS, 1)
check('nRS', args.nRS, 1)
check('nOS', args.nOS, 1)
check('mOS', args.mOS, 0)
if args.n_trials < 1 or args.n_study < 1: parser.error('n_trials e n_study devem ser >= 1')
if args.rho < 0: parser.error('rho deve ser >= 0')
if len(args.bearings) < 2: parser.error('--bearings: informe pelo menos dois rolamentos')
if len(set(args.bearings)) != len(args.bearings): parser.error('--bearings: rolamentos repetidos')

FileName = os.path.splitext(os.path.basename(__file__))[0] + "_" + args.instancia

# Configuração do experimento
FEEDBACK = 'symmetric'      # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = False, False   # configuração principal: soma ponderada dos grânulos (eq. 36)
EOL = 0.3                   # limiar de fim de vida (EoL) do HI
W = 1                       # janela do DSI em cada X_k (nº de amostras do RS; PrepareData/WindowDSI)
FOLDER = 'Optimization'
IMAGES = 'IMAGES'


def LoadData(bearings):
    data = []
    for b in bearings:
        rs_path, hi_path = os.path.join('Dataset', 'DSIR1n', f'B{b}.csv'), os.path.join('Dataset', 'HI1', f'Bearing{b}.csv')
        for p in (rs_path, hi_path):
            if not os.path.exists(p): raise FileNotFoundError(f'{p} não encontrado (rolamento {b}).')
        data.append((f'B{b}', pd.read_csv(rs_path), pd.read_csv(hi_path)['PC1']))
    return data


def Tag(data):
    # Mesmo nome de pasta que OptimizeSeq gera
    tag_rs = 'restart' if args.state == 'keep' else f'restart-{args.state}'
    return f'{FileName}_{"-".join(d[0] for d in data)}_{tag_rs}_{FEEDBACK}_G{int(WTAG)}P{int(WTAP)}'


def StageModel(data):
    OptimizeSeq(data=data, FileName=FileName, OptSampler=args.sampler, OptSeed=args.seed, OptPrune=bool(args.prune),
                n_study=args.n_study, n_trials=args.n_trials, patience=args.patience, timeout=args.timeout,
                mS=args.mS, nLS=args.nLS, nIS=args.nIS, nRS=args.nRS, nOS=args.nOS, mOS=args.mOS,
                feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, eol=EOL, w=W, agg=args.agg, state=args.state,
                OptReset=False, folder=FOLDER, write_table=bool(args.table))


def StageTable(data):
    db = os.path.abspath(os.path.join(FOLDER, Tag(data), 'optuna.db'))
    if not os.path.exists(db): raise FileNotFoundError(f'{db} não encontrado: a etapa model ainda não foi executada.')
    SeqTable(db, [d[0] for d in data], agg=args.agg)


def StagePlot(data):
    tag = Tag(data)
    csv = os.path.join(FOLDER, tag, 'mono', 'Optimization.csv')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'{csv} não encontrado: a etapa model ainda não foi executada ou nenhum trial foi '
                                f'concluído (todos podados: aumente n_trials ou use --prune 0).')
    params = BestParams(csv)
    print(f'Simulação final | params = {params} | rho = {args.rho} | intr_state = {args.intr_state} | state = {args.state}')
    os.makedirs(IMAGES, exist_ok=True)

    def on_end(t, name, i):                 # fim de um rolamento: gráficos deste rolamento
        for ext in ('png', 'pdf'):
            PlotSimulation(teda=t, mode='plt', out=os.path.join(IMAGES, f'{tag}_{i}_{name}.{ext}'), xi=0, ftrs=5, dsi_mode='last3')
            plt.close('all')
        print(f'  {name}: WAPE_RUL={t.wape_RUL:.4f} | WAPE_HI={t.wape_HI:.4f} | V_HI={t.V:.4f} | grânulos={len(t.c)} | '
              f'gráficos em {IMAGES}\\{tag}_{i}_{name}.png/.pdf')

    _, rows = SimSeq(params, data, state=args.state, mode_rul='interval', rho=args.rho, intr_state=args.intr_state,
                     eol=EOL, w=W, feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, on_end=on_end)
    out_csv = os.path.join(IMAGES, f'{tag}_resumo.csv')
    pd.DataFrame(rows).assign(state=args.state, rho=args.rho, intr_state=args.intr_state).to_csv(out_csv, index=False)
    print(f'Resumo: {out_csv}')


if __name__ == '__main__':
    data = LoadData(args.bearings)
    print(f'{FileName}: rolamentos={" -> ".join(d[0] for d in data)} | state={args.state} | agg={args.agg} | '
          f'stage={args.stage} | sampler={args.sampler} | seed={args.seed} | prune={bool(args.prune)} | '
          f'n_study={args.n_study} | n_trials={args.n_trials} | patience={args.patience} | timeout={args.timeout} | '
          f'mS={args.mS} | nLS={args.nLS} | nIS={args.nIS} | nRS={args.nRS} | nOS={args.nOS} | mOS={args.mOS}')
    if args.stage in ('all', 'model'): StageModel(data)
    if args.stage == 'table': StageTable(data)
    if args.stage in ('all', 'plot'): StagePlot(data)
