'''
Otimização do modelo AutoCloud_V5 em uma SEQUÊNCIA de um ou mais rolamentos (ex.: Bearing1_1 -> Bearing1_2):
o mesmo modelo assimila o primeiro rolamento até o EoL; antes de cada rolamento seguinte, AutoCloud.restart zera
RUL/HI/WAPE e mantém grânulos, pesos das RNNs e modelo do erro (procedimento de Teda_R04 / TedaOptimize_V5.OptimizeSeq).

Mudanças em relação ao R6:
    - usa TedaOptimize_V5 / AutoCloud_V5. OptimizeSeq é o mesmo da V4_8 (as predições pontuais não dependem do rho):
      mesmos estudos, objetivos e tabelas; a etapa plot (SimSeq) usa o rho estático --rho (rho_mode='static').
      A otimização do rho (OptimizeRho, TedaOptimize_V5) ainda não está neste script;
    - --bearings aceita um único rolamento (otimização de um rolamento, sem restart);
    - o .bat fica em TedaOpt\TedaOpt_R7.bat (roda da pasta principal) e executa apenas a etapa model (OptimizeSeq);
    - gráficos em IMAGES\TedaOpt_R7\ (pasta com o nome deste script);
    - --par 1: uma janela por nº de camadas (antes, por par de nº de camadas), como no R6 atualizado.

Mudanças do R6 em relação ao R5:
    - --mode single|multi (OptimizeSeq(mode=...) da TedaOptimize_V4_8):
        single (padrão): um objetivo: média (--agg mean) ou soma (--agg sum), sobre os rolamentos, de
                         WAPE_RUL*WAPE_HI de cada rolamento (sem normalização);
        multi : otimização multiobjetivo, dois objetivos minimizados separadamente:
                F_RUL = agg_b(WAPE_RUL_b) e F_HI = agg_b(WAPE_HI_b) (frente de Pareto). Sem a poda do Hyperband
                ao fim de cada rolamento (o Optuna não a permite com vários objetivos); as podas por limiares continuam.
                --patience e o melhor trial durante a otimização usam a soma F_RUL + F_HI. Optimization.csv:
                SCORE_<agg> = F_RUL = agg_b(WAPE_RUL_b) (1ª coluna; média, ou soma, do WAPE_RUL final dos rolamentos),
                parâmetros, w e WAPE_RUL_<b>, WAPE_HI_<b> de cada rolamento; valores não inteiros com 3 casas decimais.
                Linhas = --n_best menores SCORE por nº de camadas + toda a frente de Pareto. A etapa plot usa o menor
                SCORE (menor F_RUL, um extremo da frente) e grava também <...>_pareto.png.
                Estudos multi usam o prefixo mseq<agg>_ e convivem com os single (seq<agg>_) na mesma instância.
                Samplers multiobjetivo: tpe, nsgaii, random;
    - gráficos em IMAGES\\TedaOpt_R6\\ (pasta com o nome deste script).

Mudanças do R5 em relação ao R4:
    - usa TedaOptimize_V4_8: a janela do DSI em X_k (w, PrepareData) pode ser otimizada com --wS MIN MAX
      (inteiros >= 1), como as demais faixas. Sem --wS, w = 1 fixo (como no R4). O mesmo w vale para todos os
      rolamentos do trial; o Optimization.csv ganha a coluna w e a etapa plot usa o w da melhor linha;
    - arquivo renomeado de TedaRNN_Opt_R5 para TedaOpt_R5: pastas de resultados Optimization\\TedaOpt_R5_<...>\\ e
      gráficos em IMAGES\\TedaOpt_R5\\ (pasta com o nome deste script). Para retomar uma instância criada com o nome
      antigo, renomeie a pasta Optimization\\TedaRNN_Opt_R5_<...> para Optimization\\TedaOpt_R5_<...>;
    - todos os argumentos de OptimizeSeq podem ser definidos pelo usuário (antes fixos no script ou nos padrões da função),
      em três grupos (--help): execução (--instancia, --stage, --agg, --bearings), Optuna (--sampler, --seed, --prune,
      --n_study, --n_trials, --patience, --timeout, --reset, --warm, --n_best, --par, --workers) e modelo TEDA-RNN
      (--state, --feedback, --wtaG, --wtaP, --eol, --mS, --nLS, --nRS, --nIS, --nOS, --mOS, --N1S, --N2S, --N3S,
      --tauS, --actS, --w, --wS, --rho, --intr_state). Os padrões reproduzem a configuração anterior.
      --actS é uma faixa [MIN, MAX] de ativações (0 = tanh, 1 = relu, 2 = sigmoid); o padrão 0 1 equivale ao
      padrão actS = [0, 1, 2] de OptimizeSeq, que usa apenas actS[0] e actS[1] (a sigmoid nunca era testada).
      --reset 1 APAGA os estudos da faixa --nLS (todas as sessões --n_study) antes de otimizar; em paralelo, o
      controle apaga uma única vez, antes de abrir as janelas;
    - --workers N: N processos (janelas) por nº de camadas, todos nos MESMOS estudos do optuna.db
      (OptimizeSeq com worker/n_workers): o Optuna distribui os trials entre eles. Ex.: --nLS 1 6 --workers 4 ->
      24 janelas (4 para cada nL de 1 a 6). --n_trials continua sendo o total por sessão e por nº de
      camadas nesta execução: cada worker recebe ceil(n_trials / N). --timeout e --patience valem por worker
      (patience: trials do worker sem melhora do estudo, considerando os trials de todos os workers).
      Cada worker usa 1 thread; use no máximo ~1 processo por núcleo (os.cpu_count()). Com mais de 4 janelas,
      elas abrem minimizadas na barra de tarefas. Janela concluída fecha sozinha; com erro, espera Enter.
      --workers só vale com --par 1.

Mudanças do R4 em relação ao R3:
    - usa TedaOptimize_V4_7: a semente do sampler depende também dos trials já armazenados no estudo, de modo que
      retomar uma instância com o mesmo --seed não repete as sugestões da execução anterior;
    - combinação de parâmetros já testada no estudo (inclusive em execuções anteriores) não é simulada de novo;
    - gráficos da etapa plot em IMAGES\\TedaRNN_Opt_R4\\ (pasta com o nome deste script), apenas .png
      (sem .pdf e sem o resumo .csv; as métricas de cada rolamento são exibidas no terminal);
    - --par 1 (padrão): etapa model em paralelo, uma janela por nº de camadas da faixa --nLS (até o R7 de 25/11/2026:
      uma janela por PAR, [1, 2], [3, 4], ...). Ex.: --nLS 1 6 -> 6 janelas, nL = 1, 2, ..., 6; todas no mesmo
      optuna.db (estudos distintos por nº de camadas). Esta janela (controle) espera todas terminarem, gera o Optimization.csv
      uma única vez (etapa table) e, com --stage all, executa a etapa plot com o melhor modelo entre todas as faixas.
      Faixa com um único nº de camadas e --workers 1 (ex.: --nLS 2 2) ou --par 0: execução nesta janela, sem abrir outras.
      A janela de uma faixa pode ser fechada; o controle avisa e segue com as demais (os trials concluídos ficam no banco).

Retomada (mesma --instancia, mesmos rolamentos, --state, --mode, --agg e faixas de busca): os estudos continuam do banco.
Pode usar o mesmo --seed: a semente efetiva de cada estudo muda com o nº de trials já armazenados (TedaOptimize_V5._Seed),
e combinações já testadas são reaproveitadas em vez de simuladas de novo. A semente efetiva é exibida em cada estudo.

Etapas (--stage):
    model : OptimizeSeq -> hiperparâmetros [m, nI, nR, nO, mO, N1, N2, N3, τ, act] e w; para CADA nº de camadas da faixa
            --nLS, n_study sessões de n_trials trials. Objetivo (--mode single) = média (--agg mean) ou soma
            (--agg sum), sobre os rolamentos, de WAPE_RUL*WAPE_HI final de cada rolamento (predições
            pontuais); --mode multi: F_RUL e F_HI separados (ver acima).
    plot  : simulação final com o melhor modelo (SimSeq, RUL intervalar com --rho e --intr_state) e gráficos de
            PlotSimulation de cada rolamento em IMAGES\\TedaOpt_R7\\ (.png); --mode multi: também a frente de Pareto.
    table : apenas reconstrói o mono\\Optimization.csv a partir do optuna.db (SeqTable), sem executar trials; usado ao fim
            das execuções em paralelo (uma janela por faixa de nº de camadas, mesmo banco, com --table 0).
    test  : avaliação em rolamentos que a otimização nunca viu (--test_bearings, padrão 1_3 1_4 1_5), com os
            --test_top melhores conjuntos do Optimization.csv (escolhidos só com --bearings) FIXOS; o modelo continua
            aprendendo online (causal). --test_warm 1: percorre antes os rolamentos de --bearings. Compara com
            referências causais nos mesmos passos e pesos (persistência para o HI; tendência linear de --ref_win
            amostras para a RUL). Resultados em <instância>\\test\\Test_<rolamentos de teste>.csv e gráficos do modelo
            escolhido em IMAGES\\TedaOpt_R7\\<instância>_test_<i>_<rolamento>.png. Não faz parte de all.
            (ou TedaOpt\\TedaOpt_R7_Test.bat)
    all   : model e plot, em ordem (padrão).
Estado recorrente das RNNs no início de cada rolamento seguinte (--state): keep (padrão), reset ou fixed (AutoCloud.restart).

Uso (na pasta principal):
    .venv\\Scripts\\python.exe -X utf8 TedaOpt_R7.py --bearings 1_1 1_2 --instancia teste1 --sampler tpe
                                               --n_trials 1000 --nLS 1 3 --mS 1.75 2.05 --wS 1 5 --mode multi
    (ou TedaOpt\\TedaOpt_R7.bat, apenas etapa model: sem argumentos pergunta cada valor; --help lista os argumentos)
Dados: Dataset\\DSIR1n\\B<b>.csv (DSI) e Dataset\\HI1\\Bearing<b>.csv (HI, coluna PC1), b = 1_1, 1_2, ...
Resultados: Optimization\\TedaOpt_R7_<instancia>_<B1_1-B1_2>_restart[-<state>]_<feedback>_G<wtaG>P<wtaP>\\
            (optuna.db, mono\\Optimization.csv) e IMAGES\\TedaOpt_R7\\<mesmo nome>_<i>_<rolamento>.png
            (--mode multi: também <mesmo nome>_pareto.png)
'''
import os
import ast
import sys
import time
import shutil
import tempfile
import argparse
import traceback
import subprocess
import matplotlib
matplotlib.use('Agg')                       # gráficos salvos em arquivo, sem janela
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Functions.TedaOptimize_V5 import OptimizeSeq, SeqTable, ParetoTable, BestParamsW, SimSeq, _Parse10
from Functions.TedaOptimize_V4_9 import RefWAPE      # referências (persistência / tendência linear) da etapa test
from Functions.AutoCloud_V5 import RefEoL
from Functions.Utils import PrepareData
from Functions.Utils_OPT_R2 import ClearRuntime
from Functions.TedaGraphs import PlotSimulation


SAMPLERS = ['random', 'tpe', 'gp', 'nsgaii', 'auto', 'none']
R2 = dict(nargs=2, metavar=('MIN', 'MAX'))

parser = argparse.ArgumentParser(description='Otimização do AutoCloud_V4_6 em uma sequência de rolamentos (restart).')
# --- Execução
g = parser.add_argument_group('execução')
g.add_argument("--instancia", default="principal", help="sufixo do nome da pasta de resultados (padrão: principal)")
g.add_argument("--stage", default="all", choices=['all', 'model', 'table', 'plot', 'test'],
               help="etapa(s) executada(s); test = avaliação nos rolamentos --test_bearings (padrão: all)")
g.add_argument("--mode", default="single", choices=['single', 'multi'],
               help="single = um objetivo, agg de WAPE_RUL*WAPE_HI; multi = dois objetivos, "
                    "F_RUL = agg(WAPE_RUL) e F_HI = agg(WAPE_HI), frente de Pareto (padrão: single)")
g.add_argument("--agg", default="mean", choices=['mean', 'sum'],
               help="agregação sobre os rolamentos: média ou soma (padrão: mean)")
g.add_argument("--bearings", nargs='+', default=['1_1', '1_2'], metavar='B',
               help="rolamentos em ordem, ex.: 1_1 1_2 (padrão: 1_1 1_2)")
g.add_argument("--table", default=1, type=int, choices=[0, 1],
               help="etapa model: 1 = grava o Optimization.csv ao fim; 0 = não grava (execuções em paralelo) (padrão: 1)")
g.add_argument("--test_bearings", nargs='+', default=['1_3', '1_4', '1_5'], metavar='B',
               help="etapa test: rolamentos de teste, em ordem, nunca usados na otimização (padrão: 1_3 1_4 1_5)")
g.add_argument("--test_top", default=5, type=int,
               help="etapa test: avalia os K melhores conjuntos de parâmetros do Optimization.csv, na ordem do erro de "
                    "otimização (o 1º é o modelo escolhido; os demais medem a sensibilidade da escolha) (padrão: 5)")
g.add_argument("--test_warm", default=1, type=int, choices=[0, 1],
               help="etapa test: 1 = o modelo percorre antes os rolamentos de --bearings e chega aos de teste já treinado "
                    "(online, com restart); 0 = começa do zero no 1º rolamento de teste (padrão: 1)")
g.add_argument("--ref_win", default=10, type=int,
               help="etapa test: amostras da tendência linear da RUL de referência (0 = todo o histórico) (padrão: 10)")
# --- Optuna (OptimizeSeq)
g = parser.add_argument_group('Optuna (OptimizeSeq)')
g.add_argument("--sampler", default="random", type=str.lower, choices=SAMPLERS, help="OptSampler (padrão: random)")
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
g.add_argument("--n_best", default=10, type=int, help="linhas por nº de camadas no Optimization.csv (padrão: 10)")
g.add_argument("--par", default=1, type=int, choices=[0, 1],
               help="etapa model: 1 = uma janela por nº de camadas de --nLS, em paralelo; 0 = tudo nesta janela (padrão: 1)")
g.add_argument("--workers", default=1, type=int,
               help="etapa model em paralelo: processos (janelas) por nº de camadas, todos nos mesmos estudos; "
                    "--n_trials é dividido entre eles (padrão: 1)")
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
g.add_argument("--tauS", type=int, default=[1, 40], **R2, help="faixa de τ, constante de tempo das RNNs (padrão: 1 40)")
g.add_argument("--actS", type=int, default=[0, 1], **R2,
               help="faixa da ativação: 0 = tanh, 1 = relu, 2 = sigmoid (padrão: 0 1)")
g.add_argument("--w", default=1, type=int, help="janela w do DSI em X_k quando --wS não é informado (padrão: 1)")
g.add_argument("--wS", type=int, default=None, **R2, help="faixa da janela w do DSI em X_k (padrão: w fixo = --w)")
g.add_argument("--rho", default=0.1, type=float, help="rho estático da simulação final, etapa plot (padrão: 0.1)")
g.add_argument("--intr_state", default="carry", choices=['carry', 'reset'],
               help="estado intervalar da RUL intervalar, etapa plot (padrão: carry)")
parser.add_argument("--done", default=None, help=argparse.SUPPRESS)   # uso interno: janela de uma faixa (arquivo de término)
parser.add_argument("--worker", default=0, type=int, help=argparse.SUPPRESS)      # uso interno: nº do worker (0, 1, ...)
parser.add_argument("--n_workers", default=1, type=int, help=argparse.SUPPRESS)   # uso interno: workers por estudo
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
if args.wS is not None: check('wS', args.wS, 1)
if args.w < 1: parser.error('w deve ser >= 1')
if not 0 < args.eol < 1: parser.error('eol deve estar entre 0 e 1')
if args.n_trials < 1 or args.n_study < 1: parser.error('n_trials e n_study devem ser >= 1')
if args.n_best < 1: parser.error('n_best deve ser >= 1')
if args.workers < 1: parser.error('workers deve ser >= 1')
if args.rho < 0: parser.error('rho deve ser >= 0')
if len(args.bearings) < 1: parser.error('--bearings: informe pelo menos um rolamento')
if len(set(args.bearings)) != len(args.bearings): parser.error('--bearings: rolamentos repetidos')
if args.stage == 'test':
    if len(set(args.test_bearings)) != len(args.test_bearings): parser.error('--test_bearings: rolamentos repetidos')
    if set(args.test_bearings) & set(args.bearings):
        parser.error('--test_bearings: rolamentos de teste não podem estar em --bearings (usados na otimização)')
if args.test_top < 1: parser.error('test_top deve ser >= 1')
if args.ref_win < 0: parser.error('ref_win deve ser >= 0')

ScriptName = os.path.splitext(os.path.basename(__file__))[0]   # 'TedaOpt_R7'
FileName = ScriptName + "_" + args.instancia

# Configuração do experimento (argumentos do grupo 'modelo')
FEEDBACK = args.feedback    # 'symmetric' (gradiente no passo atual) ou 'random' (ablação: RFLO + DFA)
WTAG, WTAP = bool(args.wtaG), bool(args.wtaP)   # configuração principal: 0 0, soma ponderada dos grânulos (eq. 36)
EOL = args.eol              # limiar de fim de vida (EoL) do HI
W = args.w                  # janela do DSI em cada X_k sem --wS (nº de amostras do RS; PrepareData/WindowDSI)
FOLDER = 'Optimization'
IMAGES = os.path.join('IMAGES', ScriptName)   # gráficos em IMAGES\<nome deste script>\ (apenas .png)


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


def StageModel(data, reset=None):
    # reset: OptReset; None = --reset. Nas janelas em paralelo é sempre False (o controle apaga os estudos antes, ResetStudies)
    OptimizeSeq(data=data, FileName=FileName, OptSampler=args.sampler, OptSeed=args.seed, OptPrune=bool(args.prune),
                n_study=args.n_study, n_trials=args.n_trials, patience=args.patience, timeout=args.timeout,
                mS=args.mS, nLS=args.nLS, nIS=args.nIS, nRS=args.nRS, nOS=args.nOS, mOS=args.mOS,
                N1S=args.N1S, N2S=args.N2S, N3S=args.N3S, τS=args.tauS, actS=args.actS,
                feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, eol=EOL, w=W, wS=args.wS, agg=args.agg, state=args.state,
                OptReset=bool(args.reset) if reset is None else reset, OptWarm=bool(args.warm), n_best=args.n_best,
                folder=FOLDER, write_table=bool(args.table), worker=args.worker, n_workers=args.n_workers,
                mode=args.mode)


def ResetStudies(data):
    '''--reset 1 em paralelo: apaga UMA vez, antes de abrir as janelas, os estudos da faixa --nLS e das --n_study sessões
    (se cada janela apagasse, um worker apagaria os trials que outro acabou de gravar).'''
    import optuna
    db = os.path.abspath(os.path.join(FOLDER, Tag(data), 'optuna.db'))
    if not os.path.exists(db): return
    storage = optuna.storages.RDBStorage('sqlite:///' + db.replace(os.sep, '/'))
    names = set(optuna.get_all_study_names(storage))
    for nL in range(args.nLS[0], args.nLS[1] + 1):
        for i in range(args.n_study):
            name = f'{"seq" if args.mode == "single" else "mseq"}{args.agg}_L{nL}-{nL}_s{i + 1}'   # _SeqPrefix
            if name in names:
                optuna.delete_study(study_name=name, storage=storage)
                print(f'  {name}: estudo anterior apagado (--reset 1).')
    ClearRuntime(os.path.dirname(db))    # zera também o tempo acumulado da instância (runtime.json)


def Faixas(lo, hi):
    # Um nº de camadas por janela: [1, 6] -> [[1, 1], [2, 2], ..., [6, 6]] (antes: pares [1, 2], [3, 4], [5, 6])
    return [[a, a] for a in range(lo, hi + 1)]


def Lbl(nl):
    # Rótulo da faixa de camadas: '3' (um nº de camadas) ou '1-2'
    return f'{nl[0]}' if nl[0] == nl[1] else f'{nl[0]}-{nl[1]}'


def WorkerArgv(nl, done, k, n_trials):
    # Mesmos argumentos desta execução; a janela roda só a etapa model da sua faixa (worker k de args.workers),
    # sem gravar o Optimization.csv
    a = args
    s = lambda v: [str(x) for x in v]
    return [sys.executable, '-X', 'utf8', os.path.abspath(__file__),
            '--instancia', a.instancia, '--stage', 'model', '--mode', a.mode, '--agg', a.agg, '--bearings', *a.bearings, '--table', '0',
            '--sampler', a.sampler, '--seed', str(a.seed), '--prune', str(a.prune), '--n_study', str(a.n_study),
            '--n_trials', str(n_trials), '--patience', str(a.patience), '--timeout', str(a.timeout),
            '--reset', '0', '--warm', str(a.warm), '--n_best', str(a.n_best), '--par', '0',
            '--state', a.state, '--feedback', a.feedback, '--wtaG', str(a.wtaG), '--wtaP', str(a.wtaP), '--eol', str(a.eol),
            '--mS', *s(a.mS), '--nLS', *s(nl), '--nRS', *s(a.nRS), '--nIS', *s(a.nIS), '--nOS', *s(a.nOS),
            '--mOS', *s(a.mOS), '--N1S', *s(a.N1S), '--N2S', *s(a.N2S), '--N3S', *s(a.N3S), '--tauS', *s(a.tauS),
            '--actS', *s(a.actS), '--w', str(a.w), *(['--wS', *s(a.wS)] if a.wS is not None else []),
            '--worker', str(k), '--n_workers', str(a.workers), '--done', done]


def StageModelPar(data):
    '''Etapa model com args.workers janelas por nº de camadas; espera todas e gera o Optimization.csv (StageTable).'''
    faixas = Faixas(*args.nLS)
    nW = args.workers
    if len(faixas) * nW == 1 or os.name != 'nt':
        if os.name != 'nt': print('Janelas em paralelo disponíveis apenas no Windows: execução nesta janela.')
        StageModel(data)
        return
    n_jobs = len(faixas) * nW
    n_tr = -(-args.n_trials // nW)                                     # n_trials dividido entre os workers (arredonda para cima)
    if n_jobs > (os.cpu_count() or 1):
        print(f'Aviso: {n_jobs} processos para {os.cpu_count()} núcleos lógicos: os processos vão disputar a CPU.')
    if args.reset: ResetStudies(data)                                  # uma única vez, antes das janelas
    ctrl = tempfile.mkdtemp(prefix=f'{ScriptName}_Par_')              # arquivos <faixa>_w<k>.done (fim de cada janela)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')   # 1 thread por janela
    si = None
    if n_jobs > 4:                                                     # muitas janelas: abertas minimizadas (barra de tarefas)
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 7                                             # SW_SHOWMINNOACTIVE
    jobs = []
    print(f'\nEtapa model em paralelo: {n_jobs} janelas = {len(faixas)} faixa(s) x {nW} worker(s) | '
          + ' | '.join(f'nL {Lbl(nl)}' for nl in faixas)
          + (f' | {n_tr} trials por worker, por sessão e por nº de camadas' if nW > 1 else ''))
    for nl in faixas:
        for k in range(nW):
            done = os.path.join(ctrl, f'L{nl[0]}-{nl[1]}_w{k + 1}.done')
            p = subprocess.Popen(WorkerArgv(nl, done, k, n_tr), cwd=os.getcwd(), env=env,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)
            jobs.append((nl, k, p, done))
            print(f'  janela aberta: nL {Lbl(nl)}' + (f', worker {k + 1}/{nW}' if nW > 1 else ''))
            time.sleep(5 if len(jobs) == 1 else 1)   # a 1ª janela cria o optuna.db; as demais o encontram pronto
    print('Aguardando as janelas (verificação a cada 10 s; Ctrl+C encerra apenas esta espera, as janelas continuam)...')
    status = {}
    while len(status) < len(jobs):
        time.sleep(10)
        for nl, k, p, done in jobs:
            key = f'nL {Lbl(nl)}' + (f' worker {k + 1}' if nW > 1 else '')
            if key in status: continue
            if os.path.exists(done):        # gravado pela janela antes da pausa final
                with open(done) as f: rc = f.read().strip()
                status[key] = 'concluída' if rc == '0' else f'erro (código {rc})'
            elif p.poll() is not None:      # janela fechada antes de terminar
                status[key] = 'interrompida (janela fechada)'
            else: continue
            print(f'  {key}: {status[key]}')
    shutil.rmtree(ctrl, ignore_errors=True)
    print('\nTodas as janelas terminaram. Gerando o Optimization.csv a partir do banco comum...')
    StageTable(data)


def Worker(data):
    '''Janela de uma faixa: etapa model; grava o arquivo de término. Fecha sozinha ao concluir; em caso de erro,
    espera Enter (para ler a mensagem).'''
    title = f'{ScriptName} {args.instancia} - nL {Lbl(args.nLS)}' + \
            (f' - worker {args.worker + 1}/{args.n_workers}' if args.n_workers > 1 else '')
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception: pass
    print(title + '\n')
    rc = 0
    try:
        StageModel(data)
    except BaseException:                  # inclui Ctrl+C: a janela de controle é avisada e segue
        traceback.print_exc()
        rc = 1
    with open(args.done, 'w') as f: f.write(str(rc))
    print(f'\nFim desta faixa (código {rc}). O Optimization.csv é gerado pela janela de controle.')
    if rc != 0:
        try: input('Pressione Enter para fechar esta janela...')
        except (EOFError, KeyboardInterrupt): pass
    sys.exit(rc)


def StageTable(data):
    db = os.path.abspath(os.path.join(FOLDER, Tag(data), 'optuna.db'))
    if not os.path.exists(db): raise FileNotFoundError(f'{db} não encontrado: a etapa model ainda não foi executada.')
    SeqTable(db, [d[0] for d in data], agg=args.agg, n_best=args.n_best, w=W, mode=args.mode)


def PlotPareto(data, tag):
    '''--mode multi: todos os trials concluídos (cinza, lidos do optuna.db por ParetoTable), frente de Pareto por nº de
    camadas e modelo escolhido (menor SCORE = menor F_RUL).'''
    db = os.path.abspath(os.path.join(FOLDER, tag, 'optuna.db'))
    df = ParetoTable(db, [d[0] for d in data], agg=args.agg, w=W)
    if df.empty: return
    xr, yh, sc = f'OBJ_RUL_{args.agg}', f'OBJ_HI_{args.agg}', f'SCORE_{args.agg}'
    fr = df[df['pareto'] == 1].sort_values(xr)
    sel = df.iloc[0]                                          # menor SCORE = menor F_RUL (ParetoTable ordena por SCORE)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(df[xr], df[yh], s=10, c='0.75', label=f'trials concluídos ({len(df)})', zorder=1)
    ax.step(fr[xr], fr[yh], where='post', c='0.35', lw=0.8, zorder=2)
    cmap = plt.get_cmap('tab10')
    for i, (nL, g) in enumerate(fr.groupby('nL')):
        ax.scatter(g[xr], g[yh], s=28, color=cmap(i % 10), edgecolor='k', lw=0.4, label=f'frente, {nL} camada(s)', zorder=3)
    ax.scatter([sel[xr]], [sel[yh]], s=180, marker='*', c='gold', edgecolor='k', lw=0.8, zorder=4,
               label=f'escolhido: F_RUL = {sel[sc]:.3f}, F_HI = {sel[yh]:.3f} ({int(sel["nL"])} camada(s))')
    ax.set_xlabel(f'F_RUL: {args.agg} de WAPE_RUL sobre os rolamentos')
    ax.set_ylabel(f'F_HI: {args.agg} de WAPE_HI sobre os rolamentos')
    ax.set_title(f'Frente de Pareto | {len(fr)} pontos não dominados')
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.legend(fontsize=8, loc='best')
    out = os.path.join(IMAGES, f'{tag}_pareto.png')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'Frente de Pareto: {out}')


def StagePlot(data):
    tag = Tag(data)
    csv = os.path.join(FOLDER, tag, 'mono', 'Optimization.csv')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'{csv} não encontrado: a etapa model ainda não foi executada ou nenhum trial foi '
                                f'concluído (todos podados: aumente n_trials ou use --prune 0).')
    first = str(pd.read_csv(csv, nrows=0).columns[0])
    if first.startswith('SCORE_') != (args.mode == 'multi'):
        raise ValueError(f'{csv} foi gerado com outro --mode (1ª coluna {first}): rode --stage table com o --mode desejado.')
    params, w = BestParamsW(csv, w=W)
    print(f'Simulação final | params = {params} | w = {w} | rho = {args.rho} | intr_state = {args.intr_state} | state = {args.state}')
    os.makedirs(IMAGES, exist_ok=True)
    if args.mode == 'multi': PlotPareto(data, tag)

    def on_end(t, name, i):                 # fim de um rolamento: gráficos deste rolamento
        out = os.path.join(IMAGES, f'{tag}_{i}_{name}.png')
        PlotSimulation(teda=t, mode='plt', out=out, xi=0, ftrs=5, dsi_mode='last3')
        plt.close('all')
        print(f'  {name}: WAPE_RUL={t.wape_RUL:.4f} | WAPE_HI={t.wape_HI:.4f} | V_HI={t.V:.4f} | grânulos={len(t.c)} | '
              f'gráfico em {out}')

    SimSeq(params, data, state=args.state, mode_rul='interval', rho=args.rho, intr_state=args.intr_state,
           eol=EOL, w=w, feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, on_end=on_end)


def TopParams(csv, k):
    '''Os k melhores conjuntos de parâmetros distintos do Optimization.csv, em ordem crescente da 1ª coluna (erro de
    otimização: WAPE_<agg> ou SCORE_<agg>). Retorna [(posição, erro, params, w)], params no formato de SimSeq
    [m, nI, nR, nO, mO, N1, N2, N3, τ, act] (mesma leitura de BestParamsW).'''
    df = pd.read_csv(csv)
    prm = list(df.columns[1:11]) + (['w'] if 'w' in df.columns else [])
    df = df.sort_values(df.columns[0], ascending=True).drop_duplicates(subset=prm).head(k)
    out = []
    for r, (_, row) in enumerate(df.iterrows()):
        w = int(row['w']) if 'w' in df.columns else W
        out.append((r + 1, float(row.iloc[0]), _Parse10(row.iloc[1:11].values, w), w))
    return out


def RefTest(nm, RS_b, HI_b, params, w):
    '''WAPE das referências causais (TedaOptimize_V4_9.RefWAPE) nos MESMOS passos e pesos (k - k_seg)^2 do modelo:
    persistência para o HI (h_{k+1} = h_k) e tendência linear das --ref_win últimas amostras para a RUL.'''
    _, nI, _, nO, mO = params[:5]
    sig = np.asarray(HI_b, dtype=float).ravel()
    _, _, _, kX = PrepareData(RS_b, HI_b, w, nI, nO, mO)
    _, J, L = RefEoL(sig, EOL, kX)
    return RefWAPE(sig, EOL, kX, J, L, args.ref_win)


def StageTest(data):
    '''
    Avaliação em rolamentos que a otimização nunca viu (--test_bearings). Os hiperparâmetros vêm do Optimization.csv
    desta instância (escolhidos apenas com --bearings) e ficam FIXOS; o modelo continua aprendendo online (causal), como
    em toda simulação. Com --test_warm 1, percorre antes os rolamentos de --bearings (restart entre rolamentos) e chega
    aos de teste já treinado; com 0, começa do zero no 1º rolamento de teste. As métricas de teste são as dos
    rolamentos de --test_bearings; as dos rolamentos de --bearings (otimização) aparecem para comparação.
    São avaliados os --test_top melhores conjuntos (o 1º é o modelo escolhido). Referências (REF_RUL, REF_HI) nos
    mesmos passos e pesos. Resultados: Optimization\\<instância>\\test\\Test_<rolamentos de teste>[_frio].csv e
    gráficos do modelo escolhido em IMAGES\\TedaOpt_R7\\<instância>_test_<i>_<rolamento>.png.
    '''
    tag = Tag(data)
    csv = os.path.join(FOLDER, tag, 'mono', 'Optimization.csv')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'{csv} não encontrado: rode antes a etapa model (ou table) desta instância.')
    first = str(pd.read_csv(csv, nrows=0).columns[0])
    if first.startswith('SCORE_') != (args.mode == 'multi'):
        raise ValueError(f'{csv} foi gerado com outro --mode (1ª coluna {first}): use o --mode da otimização.')
    test = LoadData(args.test_bearings)
    seq = (data if args.test_warm else []) + test
    role = {d[0]: 'otimização' for d in data}
    role.update({d[0]: 'teste' for d in test})
    tops = TopParams(csv, args.test_top)
    os.makedirs(IMAGES, exist_ok=True)
    print(f'\nEtapa test | otimização: {" -> ".join(d[0] for d in data)} | teste: {" -> ".join(d[0] for d in test)} | '
          f'sequência simulada: {" -> ".join(d[0] for d in seq)} ({"com" if args.test_warm else "sem"} os rolamentos '
          f'de otimização antes) | {len(tops)} conjunto(s) de parâmetros | {first} = erro de otimização')
    rows = []
    for pos, err, params, w in tops:
        chosen = pos == 1

        def on_end(t, name, i, chosen=chosen):   # gráficos apenas do modelo escolhido, nos rolamentos de teste
            if chosen and role[name] == 'teste':
                out = os.path.join(IMAGES, f'{tag}_test_{i}_{name}.png')
                PlotSimulation(teda=t, mode='plt', out=out, xi=0, ftrs=5, dsi_mode='last3')
                plt.close('all')

        t0 = time.time()
        _, rr = SimSeq(params, seq, state=args.state, mode_rul='interval' if chosen else 'single', rho=args.rho,
                       intr_state=args.intr_state, eol=EOL, w=w, feedback=FEEDBACK, wtaG=WTAG, wtaP=WTAP, on_end=on_end)
        for r, (nm, RS_b, HI_b) in zip(rr, seq):
            refR, refH = RefTest(nm, RS_b, HI_b, params, w)
            if not chosen: r.update(V_HI=np.nan, cov_HI=np.nan, sp_HI=np.nan)   # intervalos só no modelo escolhido
            rows.append(dict(posicao=pos, erro_otimizacao=err, papel=role[nm], rolamento=nm,
                             WAPE_RUL=r['WAPE_RUL'], REF_RUL=refR, WAPE_HI=r['WAPE_HI'], REF_HI=refH,
                             V_HI=r['V_HI'], cov_HI=r['cov_HI'], sp_HI=r['sp_HI'], granulos=r['granulos'], passos=r['passos'],
                             m=params[0], nI=params[1], nR=str(params[2]), nO=params[3], mO=params[4], N1=params[5],
                             N2=params[6], N3=params[7], TAU=params[8], activation=params[9], w=w))
        te = [x for x in rows if x['posicao'] == pos and x['papel'] == 'teste']
        print(f'  #{pos} ({first} = {err:.4f}, {time.time() - t0:.0f} s) | teste: WAPE_RUL = '
              f'{np.mean([x["WAPE_RUL"] for x in te]):.4f} (ref {np.mean([x["REF_RUL"] for x in te]):.4f}) | WAPE_HI = '
              f'{np.mean([x["WAPE_HI"] for x in te]):.4f} (ref {np.mean([x["REF_HI"] for x in te]):.4f})')
    df = pd.DataFrame(rows)
    out_dir = os.path.join(FOLDER, tag, 'test')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'Test_{"-".join(d[0] for d in test)}{"" if args.test_warm else "_frio"}.csv')
    df.to_csv(out, index=False)

    # Resumo: modelo escolhido, rolamento a rolamento; e média de teste de cada conjunto
    pd.set_option('display.width', 200)
    best = df[df['posicao'] == 1][['papel', 'rolamento', 'WAPE_RUL', 'REF_RUL', 'WAPE_HI', 'REF_HI', 'V_HI', 'granulos']]
    print(f'\nModelo escolhido (#1), por rolamento (REF_* = referências nos mesmos passos):\n{best.round(4).to_string(index=False)}')
    agg = (df[df['papel'] == 'teste'].groupby('posicao')
           .agg(erro_otimizacao=('erro_otimizacao', 'first'), WAPE_RUL=('WAPE_RUL', 'mean'), REF_RUL=('REF_RUL', 'mean'),
                WAPE_HI=('WAPE_HI', 'mean'), REF_HI=('REF_HI', 'mean')))
    agg['RUL/ref'] = agg['WAPE_RUL'] / agg['REF_RUL']
    agg['HI/ref'] = agg['WAPE_HI'] / agg['REF_HI']
    print(f'\nMédia nos rolamentos de teste, por conjunto de parâmetros (ordem da otimização):\n{agg.round(4).to_string()}')
    print(f'\nResultados: {out}')
    return df


if __name__ == '__main__':
    data = LoadData(args.bearings)
    print(f'{FileName}: stage={args.stage} | mode={args.mode} | agg={args.agg} | rolamentos={" -> ".join(d[0] for d in data)}\n'
          f'  Optuna: sampler={args.sampler} | seed={args.seed} | prune={bool(args.prune)} | n_study={args.n_study} | '
          f'n_trials={args.n_trials} | patience={args.patience} | timeout={args.timeout} | reset={bool(args.reset)} | '
          f'warm={bool(args.warm)} | n_best={args.n_best} | par={args.par} | workers={args.workers}\n'
          f'  Modelo: state={args.state} | feedback={FEEDBACK} | wtaG={WTAG} | wtaP={WTAP} | eol={EOL} | '
          f'mS={args.mS} | nLS={args.nLS} | nRS={args.nRS} | nIS={args.nIS} | nOS={args.nOS} | mOS={args.mOS} | '
          f'N1S={args.N1S} | N2S={args.N2S} | N3S={args.N3S} | tauS={args.tauS} | actS={args.actS} | '
          f'wS={args.wS if args.wS is not None else f"fixo (w = {W})"} | rho={args.rho} | intr_state={args.intr_state}')
    if args.done: Worker(data)              # janela aberta por StageModelPar
    if args.stage in ('all', 'model'): StageModelPar(data) if args.par else StageModel(data)
    if args.stage == 'table': StageTable(data)
    if args.stage in ('all', 'plot'): StagePlot(data)
    if args.stage == 'test': StageTest(data)
