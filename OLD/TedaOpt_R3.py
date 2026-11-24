'''
Otimização do modelo AutoCloud_V4_6 (rho estático) em uma SEQUÊNCIA de rolamentos (ex.: Bearing1_1 -> Bearing1_2):
o mesmo modelo assimila o primeiro rolamento até o EoL; antes de cada rolamento seguinte, AutoCloud.restart zera
RUL/HI/WAPE e mantém grânulos, pesos das RNNs e modelo do erro (procedimento de Teda_R03 / TedaOptimize_V4_6.OptimizeSeq).

Arquivo renomeado de TedaRNN_Opt_R3 para TedaOpt_R3: pastas de resultados Optimization\\TedaOpt_R3_<...>\\ (para retomar
uma instância criada com o nome antigo, renomeie a pasta Optimization\\TedaRNN_Opt_R3_<...> para TedaOpt_R3_<...>).
Todos os argumentos de OptimizeSeq podem ser definidos pelo usuário (antes fixos no script ou nos padrões da função), em
três grupos (--help): execução (--instancia, --stage, --agg, --bearings), Optuna (--sampler, --seed, --prune, --n_study,
--n_trials, --patience, --timeout, --reset, --warm, --n_best) e modelo TEDA-RNN (--state, --feedback, --wtaG, --wtaP,
--eol, --mS, --nLS, --nRS, --nIS, --nOS, --mOS, --N1S, --N2S, --N3S, --tauS, --actS, --w, --rho, --intr_state).
Os padrões reproduzem a configuração anterior. --actS é uma faixa [MIN, MAX] de ativações (0 = tanh, 1 = relu,
2 = sigmoid); o padrão 0 1 equivale ao padrão actS = [0, 1, 2] de OptimizeSeq, que usa apenas actS[0] e actS[1].
--reset 1 APAGA os estudos da faixa --nLS (todas as sessões --n_study) antes de otimizar.
Nesta revisão (TedaOptimize_V4_6), retomar uma instância com o mesmo --seed repete as sugestões do sampler da execução
anterior; R5/R6 (TedaOptimize_V4_8/V4_9) não têm essa limitação.

Etapas (--stage):
    model : OptimizeSeq -> hiperparâmetros [m, nI, nR, nO, mO, N1, N2, N3, τ, act]; para CADA nº de camadas da faixa
            --nLS, n_study sessões de n_trials trials. Objetivo = média (--agg mean) ou soma (--agg sum), sobre os
            rolamentos, de WAPE_RUL + WAPE_HI final de cada rolamento (predições pontuais).
    plot  : simulação final com o melhor modelo (SimSeq, RUL intervalar com --rho e --intr_state) e gráficos de
            PlotSimulation de cada rolamento em IMAGES/, mais um resumo .csv.
    table : apenas reconstrói o mono\\Optimization.csv a partir do optuna.db (SeqTable), sem executar trials; usado ao fim
            das execuções em paralelo (uma janela por faixa de nº de camadas, mesmo banco, com --table 0).
    all   : model e plot, em ordem (padrão).
Estado recorrente das RNNs no início de cada rolamento seguinte (--state): keep (padrão), reset ou fixed (AutoCloud.restart).

Uso (na pasta principal):
    .venv\\Scripts\\python.exe TedaOpt_R3.py --bearings 1_1 1_2 --instancia teste1 --sampler tpe
                                               --n_trials 1000 --nLS 1 3 --mS 1.75 2.05
    (ou TedaOpt_R3.bat: sem argumentos pergunta cada valor; --help lista os argumentos)
Dados: Dataset\\DSIR1n\\B<b>.csv (DSI) e Dataset\\HI1\\Bearing<b>.csv (HI, coluna PC1), b = 1_1, 1_2, ...
Resultados: Optimization\\TedaOpt_R3_<instancia>_<B1_1-B1_2>_restart[-<state>]_<feedback>_G<wtaG>P<wtaP>\\
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
R2 = dict(nargs=2, metavar=('MIN', 'MAX'))

parser = argparse.ArgumentParser(description='Otimização do AutoCloud_V4_6 em uma sequência de rolamentos (restart).')
# --- Execução
g = parser.add_argument_group('execução')
g.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
g.add_argument("--stage", default="all", choices=['all', 'model', 'table', 'plot'], help="etapa(s) executada(s) (padrão: all)")
g.add_argument("--agg", default="mean", choices=['mean', 'sum'],
               help="objetivo: média ou soma de WAPE_RUL + WAPE_HI dos rolamentos (padrão: mean)")
g.add_argument("--bearings", nargs='+', default=['1_1', '1_2'], metavar='B',
               help="rolamentos em ordem, ex.: 1_1 1_2 (padrão: 1_1 1_2)")
g.add_argument("--table", default=1, type=int, choices=[0, 1],
               help="etapa model: 1 = grava o Optimization.csv ao fim; 0 = não grava (execuções em paralelo) (padrão: 1)")
# --- Optuna (OptimizeSeq)
g = parser.add_argument_group('Optuna (OptimizeSeq)')
g.add_argument("--sampler", default="tpe", type=str.lower, choices=SAMPLERS, help="OptSampler (padrão: tpe)")
g.add_argument("--seed", default=1, type=int, help="OptSeed: semente base das sessões (padrão: 1)")
g.add_argument("--prune", default=1, type=int, choices=[0, 1], help="OptPrune: 1 = poda trials ruins (padrão: 1)")
g.add_argument("--n_study", default=1, type=int, help="sessões por nº de camadas (padrão: 1)")
g.add_argument("--n_trials", default=1000, type=int, help="trials novos por sessão e por nº de camadas (padrão: 1000)")
g.add_argument("--patience", default=500, type=int, help="parada antecipada: trials sem melhora (padrão: 500)")
g.add_argument("--timeout", default=3600, type=int, help="tempo máximo por sessão, em s (padrão: 3600)")
g.add_argument("--reset", default=0, type=int, choices=[0, 1],
               help="OptReset: 1 = APAGA os estudos desta instância (faixa --nLS e --n_study) antes de otimizar (padrão: 0)")
g.add_argument("--warm", default=0, type=int, choices=[0, 1],
               help="OptWarm: 1 = começa pelo melhor trial das outras sessões do mesmo nº de camadas (padrão: 0)")
g.add_argument("--n_best", default=5, type=int, help="linhas por nº de camadas no Optimization.csv (padrão: 5)")
# --- Modelo (AutoCloud / eRNN): configuração e faixas de busca
g = parser.add_argument_group('modelo TEDA-RNN (configuração e faixas de busca)')
g.add_argument("--state", default="keep", choices=['keep', 'reset', 'fixed'],
               help="estado das RNNs no início de cada rolamento seguinte (padrão: keep)")
g.add_argument("--feedback", default="symmetric", choices=['symmetric', 'random'],
               help="sinal de erro das RNNs: symmetric (gradiente) ou random (ablação: RFLO + DFA) (padrão: symmetric)")
g.add_argument("--wtaG", default=0, type=int, choices=[0, 1],
               help="1 = winner-takes-all na saída dos grânulos (0: soma ponderada, eq. 36) (padrão: 0)")
g.add_argument("--wtaP", default=0, type=int, choices=[0, 1],
               help="1 = winner-takes-all na predição da RUL (padrão: 0)")
g.add_argument("--eol", default=0.3, type=float, help="limiar de fim de vida (EoL) do HI (padrão: 0.3)")
g.add_argument("--mS", type=float, default=[1.75, 2.05], **R2, help="faixa de m, limiar do TEDA (padrão: 1.75 2.05)")
g.add_argument("--nLS", type=int, default=[1, 3], **R2, help="faixa do nº de camadas (padrão: 1 3)")
g.add_argument("--nRS", type=int, default=[1, 60], **R2, help="faixa de neurônios por camada (padrão: 1 60)")
g.add_argument("--nIS", type=int, default=[2, 40], **R2, help="faixa de nI, valores passados do HI na entrada (padrão: 2 40)")
g.add_argument("--nOS", type=int, default=[1, 40], **R2, help="faixa de nO, valores futuros de Z_k (padrão: 1 40)")
g.add_argument("--mOS", type=int, default=[0, 40], **R2, help="faixa de mO, valores passados de Z_k (padrão: 0 40)")
g.add_argument("--N1S", type=int, default=[1, 9], **R2, help="faixa de N1, expoente da taxa de aprendizado 1 (padrão: 1 9)")
g.add_argument("--N2S", type=int, default=[1, 9], **R2, help="faixa de N2, expoente da taxa de aprendizado 2 (padrão: 1 9)")
g.add_argument("--N3S", type=int, default=[1, 9], **R2, help="faixa de N3, expoente da taxa de aprendizado 3 (padrão: 1 9)")
g.add_argument("--tauS", type=int, default=[1, 25], **R2, help="faixa de τ, constante de tempo das RNNs (padrão: 1 25)")
g.add_argument("--actS", type=int, default=[0, 1], **R2,
               help="faixa da ativação: 0 = tanh, 1 = relu, 2 = sigmoid (padrão: 0 1)")
g.add_argument("--w", default=1, type=int, help="janela w do DSI em X_k, fixa (padrão: 1)")
g.add_argument("--rho", default=0.1, type=float, help="rho estático da simulação final, etapa plot (padrão: 0.1)")
g.add_argument("--intr_state", default="carry", choices=['carry', 'reset'],
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
check('N1S', args.N1S, 0)
check('N2S', args.N2S, 0)
check('N3S', args.N3S, 0)
check('tauS', args.tauS, 1)
check('actS', args.actS, 0)
if args.actS[1] > 2: parser.error('actS: ativações disponíveis 0 (tanh), 1 (relu) e 2 (sigmoid)')
if args.w < 1: parser.error('w deve ser >= 1')
if not 0 < args.eol < 1: parser.error('eol deve estar entre 0 e 1')
if args.n_trials < 1 or args.n_study < 1: parser.error('n_trials e n_study devem ser >= 1')
if args.n_best < 1: parser.error('n_best deve ser >= 1')
if args.rho < 0: parser.error('rho deve ser >= 0')
if len(args.bearings) < 2: parser.error('--bearings: informe pelo menos dois rolamentos')
if len(set(args.bearings)) != len(args.bearings): parser.error('--bearings: rolamentos repetidos')

FileName = os.path.splitext(os.path.basename(__file__))[0] + "_" + args.instancia   # 'TedaOpt_R3_<instancia>'

# Configuração do experimento (argumentos do grupo 'modelo')
FEEDBACK = args.feedback    # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = bool(args.wtaG), bool(args.wtaP)   # configuração principal: 0 0, soma ponderada dos grânulos (eq. 36)
EOL = args.eol              # limiar de fim de vida (EoL) do HI
W = args.w                  # janela do DSI em cada X_k (nº de amostras do RS; PrepareData/WindowDSI)
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
                N1S=args.N1S, N2S=args.N2S, N3S=args.N3S, τS=args.tauS, actS=args.actS,
                feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, eol=EOL, w=W, agg=args.agg, state=args.state,
                OptReset=bool(args.reset), OptWarm=bool(args.warm), n_best=args.n_best,
                folder=FOLDER, write_table=bool(args.table))


def StageTable(data):
    db = os.path.abspath(os.path.join(FOLDER, Tag(data), 'optuna.db'))
    if not os.path.exists(db): raise FileNotFoundError(f'{db} não encontrado: a etapa model ainda não foi executada.')
    SeqTable(db, [d[0] for d in data], agg=args.agg, n_best=args.n_best)


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
    print(f'{FileName}: stage={args.stage} | agg={args.agg} | rolamentos={" -> ".join(d[0] for d in data)}\n'
          f'  Optuna: sampler={args.sampler} | seed={args.seed} | prune={bool(args.prune)} | n_study={args.n_study} | '
          f'n_trials={args.n_trials} | patience={args.patience} | timeout={args.timeout} | reset={bool(args.reset)} | '
          f'warm={bool(args.warm)} | n_best={args.n_best}\n'
          f'  Modelo: state={args.state} | feedback={FEEDBACK} | wtaG={WTAG} | wtaP={WTAP} | eol={EOL} | '
          f'mS={args.mS} | nLS={args.nLS} | nRS={args.nRS} | nIS={args.nIS} | nOS={args.nOS} | mOS={args.mOS} | '
          f'N1S={args.N1S} | N2S={args.N2S} | N3S={args.N3S} | tauS={args.tauS} | actS={args.actS} | w={W} | '
          f'rho={args.rho} | intr_state={args.intr_state}')
    if args.stage in ('all', 'model'): StageModel(data)
    if args.stage == 'table': StageTable(data)
    if args.stage in ('all', 'plot'): StagePlot(data)
