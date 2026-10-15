import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from Functions.Utils import *
from scipy.signal import stft
from scipy.signal import hilbert

def PlotPLY(x=None,y=None,w=400,h=400,title='r'):
    if x is None: x = [i for i in range(len(y))]
    fig=make_subplots(rows=1,cols=1)
    trace = go.Scatter(x=x, y= y)
    fig.add_trace(trace, row=1, col=1)

    fig.update_layout(width = w, height = h, title = title)
    fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
    fig.show()

def PlotMPL(x=None, y=None, w=5, h=5, title='r'):
    if x is None: x = [i for i in range(len(y))]
    plt.figure(figsize=(w, h))  # Define o tamanho do gráfico
    plt.plot(x, y, linestyle='-', color='b', label="Sinal")  # Plota os dados
    plt.xlabel("Tempo")  # Nome do eixo X
    plt.ylabel("Amplitude")  # Nome do eixo Y
    plt.title(title)  # Define o título do gráfico
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.legend()  # Exibe legenda
    plt.show()  # Mostra o gráfico

def PlotSingle(x,y,mode='plt',w=5,h=3,title='r'):
    if mode=='plt':
        PlotMPL(x,y,w,h,title)
    elif mode=='ply':
        w=w*100
        h=h*100
        PlotPLY(x,y,w,h,title)

def PlotSeriesPLY(xSeries=None,ySeries=None, names=None, title='Séries Temporais',
                  markers=None, w=600, h=400):
    
    if xSeries==None: 
        xSeries = [[i for i in range(len(ySeries[j]))] for j in range(len(ySeries))]
    x=xSeries
    y=ySeries
    line_modes = ['lines', 'markers']
    fig = make_subplots(rows=1, cols=1)
    
    if names is None:
        names = [f'Série {i+1}' for i in range(len(y))]
    if markers is None:
        markers = [0] * len(y) # Padrão para todos como 'lines'

    for x, y, name, m_idx in zip(x, y, names, markers):
        mode = line_modes[m_idx] if m_idx < len(line_modes) else 'lines'
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode=mode),row=1, col=1)

    fig.update_layout(width=w, height=h, title=title,template='plotly_white')
    fig.update_yaxes(title_text='Amplitude', row=1, col=1,)
    fig.update_xaxes(title_text='Tempo / Frequência', row=1, col=1,)
    fig.show()

def PlotSeriesPLT(xSeries=None,ySeries=None, names=None, title='Séries Temporais',
                            w=6, h=4):
    
    if xSeries is None:
        xSeries = [np.arange(len(y)) for y in ySeries]
    
    fig, ax = plt.subplots(figsize=(w, h))
    
    if names is None:
        names = [f'Série {i+1}' for i in range(len(ySeries))]
        

    for x, y, name in zip(xSeries, ySeries, names):
    
        ax.plot(x, y, label=name, linestyle='-', linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel('Tempo / Frequência')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, fontsize='small')
    plt.tight_layout()
    plt.show()

def PlotSeries(xSeries=None,ySeries=None,mode='plt',w=5,h=3,title='r'):
    if mode=='plt':
        PlotSeriesPLT(xSeries=xSeries,ySeries=ySeries,w=w,h=h,title=title)
    elif mode=='ply':

        PlotSeriesPLY(xSeries=xSeries,ySeries=ySeries,w=w*100,h=h*100,title=title)
     

def PlotTwoScales(y1,y2,x1=None,x2=None,w=5,h=3,y1_name=None,y2_name=None):
    if y1_name is None: y1_name = 'y1'
    if y2_name is None: y2_name = 'y2'
    if x1 is None: x1 = np.arange(len(y1))
    if x2 is None: x2 = np.arange(len(y2))
    fig, ax1 = plt.subplots(figsize=(w, h))

    ax1.set_xlabel('x')
    ax1.set_ylabel(y1_name, color='blue')
    ax1.plot(x1, y1, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    ax2.set_ylabel(y2_name, color='red')  # we already handled the x-label with ax1
    ax2.plot(x2, y2, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def PlotTwoScalesPLY(y1, y2,x1=None, x2=None, w=500, h=300, y1_name='y1', y2_name='y2',x_name='Cycle'):
    # Criar subplots com um eixo Y secundário
    if x1==None:
        x1 = np.arange(len(y1))
    if x2==None:
        x2 = np.arange(len(y2))
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Adicionar a primeira série (Eixo Principal - Esquerda)
    fig.add_trace(
        go.Scatter(x=x1, y=y1, name=y1_name, line=dict(color='blue')),
        secondary_y=False,
    )

    # Adicionar a segunda série (Eixo Secundário - Direita)
    fig.add_trace(
        go.Scatter(x=x2, y=y2, name=y2_name, line=dict(color='red')),
        secondary_y=True,
    )

    # Configurar títulos e cores dos eixos
    fig.update_layout(
        width=w, height=h,
        xaxis_title=x_name,
        template="plotly_white", # Fundo limpo como o tight_layout
        legend=dict(orientation="v"
                    #, yanchor="bottom"
                    , y=1.0
                    #, xanchor="right"
                    , x=1.2
                    )
    )

    # Estilizar o eixo Y principal (Azul)
    fig.update_yaxes(
        title_text=f"<b>{y1_name}</b>", 
        title_font=dict(color="blue"), 
        tickfont=dict(color="blue"), 
        secondary_y=False
    )

    # Estilizar o eixo Y secundário (Vermelho)
    fig.update_yaxes(
        title_text=f"<b>{y2_name}</b>", 
        title_font=dict(color="red"), 
        tickfont=dict(color="red"), 
        secondary_y=True
    )

    fig.show()

def PlotMultiScalesPLY(yS, xS=None, names=None, colors=None, w=500, h=300, x_name='Cycle'):
    n_series = len(yS)
    if n_series == 0:
        raise ValueError("A lista 'yS' deve conter pelo menos uma série temporal.")

    # 1. Tratamento do eixo X
    if xS is None:
        xS = [np.arange(len(y)) for y in yS]

    # 2. Tratamento dos Nomes
    if names is None:
        names = [f'y{i+1}' for i in range(n_series)]

    # 3. Cores dinâmicas
    if colors is None:
        palette = px.colors.qualitative.Plotly
        colors = [palette[i % len(palette)] for i in range(n_series)]

    # 4. Cálculo do Espaçamento dos Eixos Laterais
    n_right_axes = n_series // 2
    n_left_axes = (n_series - 1) - n_right_axes

    step = 0.045 if n_series > 8 else 0.06

    left_margin = step * n_left_axes
    right_margin = step * n_right_axes

    left_domain = left_margin
    right_domain = 1.0 - right_margin - 0.02 

    fig = go.Figure()

    layout_update = dict(
        width=w, 
        height=h,
        xaxis=dict(title=x_name, domain=[left_domain, right_domain]),
        template="plotly_white",
        legend=dict(
            orientation="v", 
            y=1.0, 
            x=1.00, 
            xanchor="left"
        ),
        margin=dict(l=20, r=20, t=30, b=80)
    )

    # 5. Construção dos Eixos Y e Traces
    for i in range(n_series):
        axis_name = f'yaxis{i+1}' if i > 0 else 'yaxis'
        
        fig.add_trace(
            go.Scatter(
                x=xS[i], 
                y=yS[i], 
                name=names[i], 
                line=dict(color=colors[i]),
                yaxis=f'y{i+1}' if i > 0 else 'y'
            )
        )

        # Título configurado com HTML/CSS para rotação horizontal
        title_html = f"<span style='writing-mode: horizontal-tb; display: inline-block;'><b>{names[i]}</b></span>"

        # Configuração base do eixo Y
        axis_config = dict(
            title=dict(
                text=title_html, 
                font=dict(color=colors[i]),
                standoff=10
            ),
            tickfont=dict(color=colors[i]),
            showgrid=(i == 0)
        )

        if i > 0:
            axis_config['overlaying'] = 'y'
            axis_config['anchor'] = 'free'
            
            # Posições horizontais calculadas corretamente no nível do eixo
            if i % 2 == 1: # Lado Direito
                pos = right_domain + step * ((i - 1) // 2 + 1)
                axis_config['side'] = 'right'
                axis_config['position'] = min(pos, 0.98)
            else: # Lado Esquerdo
                pos = left_domain - step * (i // 2)
                axis_config['side'] = 'left'
                axis_config['position'] = max(pos, 0.0)

        layout_update[axis_name] = axis_config

    fig.update_layout(**layout_update)
    fig.show()

def PlotPredError(rtlo,w=9,h=3):
    s = len(rtlo.yWAPE)
    t = rtlo.t
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(w, h))
    axes = axes.flatten()
    ax1,ax2,ax3,ax4 = axes[0], axes[1], axes[2], axes[3]

    ax1.plot(t, rtlo.yR, color='black',label='Y-Real', linestyle='-')
    ax1.plot(t, rtlo.yP, color='blue',label='Y-Pred', linestyle='-')
    ax1.plot(t, rtlo.yL, color='blue', linestyle='--')
    ax1.plot(t, rtlo.yU, color='blue', linestyle='--')
    
    ax1.set_title('Y - Real x Prediction')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y', color='black')
    ax1.legend()

    ax2.plot(t, rtlo.eR, color='black',label='e-Real', linestyle='-')
    ax2.plot(t, rtlo.eP, color='blue',label='e-Pred', linestyle='-')
    ax2.set_title('Error - Real x Prediction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Prediction Error', color='black') 
    ax2.legend()
    
    ax3.plot(t[-s:], rtlo.yWAPE, color='blue',label='WAPE', linestyle='-')
    ax3.set_title('Prediction WAPE')
    ax3.set_xlabel('X')
    ax3.set_ylabel('WAPE', color='black') 

    '''ax4.plot(t[-s:], rtlo.rWAPE, color='blue',label='WAPE', linestyle='-')
    ax4.set_title('RUL Prediction WAPE')
    ax4.set_xlabel('X')
    ax4.set_ylabel('WAPE', color='black') '''

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def PlotPredErrorPLY(rtlo, w=800, h=300):
    # s: tamanho do vetor WAPE (caso comece depois do início)
    t = rtlo.t
    tol = 25
    # Criando o layout de 1 linha e 4 colunas
    fig = make_subplots(
        rows=1, cols=3, 
        shared_xaxes=True,
        subplot_titles=('Y - Real x Prediction', 'RUL - Real x Pred', 'Error Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}]]
    )

    # --- Subplot 1: Y Real x Pred (com Intervalos) ---
    fig.add_trace(go.Scatter(x=t, y=rtlo.yR, name='Deg-R', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=rtlo.yP, name='Deg-P', line=dict(color='blue')), row=1, col=1)
    # Intervalos (Dashed)
   
    fig.add_trace(go.Scatter(x=t, y=rtlo.yU, name='y-U', line=dict(color='blue', dash='dash')
                             , showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=t, y=rtlo.yL, name=f"Deg-U", fill='tonexty',
                            fillcolor='rgba(149, 100, 237, 0.25)', line=dict(width=0)
                            , showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=rtlo.yL, line=dict(color='blue', dash='dash')
                             , showlegend=False), row=1, col=1)


    fig.add_trace(go.Scatter(x=t, y=rtlo.rulR, name='RUL-R', line=dict(color='black')
                             , showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=rtlo.rulP, name='RUL-P', line=dict(color='blue')
                             , showlegend=True), row=1, col=2)
    
    fig.add_trace(go.Scatter( x=t, y=(1 + tol / 100) * rtlo.rulR,mode='lines',
        line=dict(width=0),showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter( x=t, y=(1 - tol / 100) * rtlo.rulR, fill='tonexty', 
        fillcolor='rgba(128, 128, 128, 0.25)',line=dict(width=0),name=f"T-{tol}%",
        mode='lines'), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=rtlo.rulU, mode='lines', line=dict(color='blue', dash='dash')
                             , showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=rtlo.rulL, name=f"RUL-U", fill='tonexty',
                            fillcolor='rgba(100, 149, 237, 0.25)', line=dict(width=0)
                            , showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=rtlo.rulL, mode='lines', line=dict(color='blue', dash='dash')
                             , showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=t, y=rtlo.εR_hist, name='RUL wMAPE', line=dict(color='black')),
        row=1, col=3, 
        secondary_y=False  # Fica FORA do go.Scatter
    )

    fig.add_trace(
        go.Scatter(x=t, y=rtlo.εM_hist, name='Deg wMAPE', line=dict(color='blue')),
        row=1, col=3, 
        secondary_y=True   # Fica FORA do go.Scatter
    )
    # Atualizando Layout e Eixos
    fig.update_layout(
        width=w, height=h,
        title_text=f"RTLO Model Performance Analysis",
        template='plotly_white',
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_yaxes(
        title_text="<b>RUL wMAPE</b>", 
        title_font=dict(color="black"), 
        tickfont=dict(color="black"), 
        secondary_y=False,
        row=1, col=3
    )

    fig.update_yaxes(
        title_text="<b>RUL wMAPE</b>", 
        title_font=dict(color="blue"), 
        tickfont=dict(color="blue"), 
        secondary_y=True,
        row=1, col=3
    )

    # Labels dos eixos (opcional, já que os títulos ajudam)
    fig.update_xaxes(title_text="Time / Index")
    fig.update_yaxes(title_text="Amplitude", col=1)
    fig.update_yaxes(title_text="Error", col=2)

    fig.show()

def PlotPredErrorPLT(rtlo, save=False, w=12, h=4):
    t = rtlo.t
    tol = 25
    
    # Criando a figura com 3 subplots lado a lado
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w, h), sharex=True)
    fig.suptitle("RTLO Model Performance Analysis", fontsize=14, fontweight='bold')

    # --- Subplot 1: Y Real x Prediction ---
    ax1.plot(t, rtlo.yR, color='black', label='Deg-R')
    ax1.plot(t, rtlo.yP, color='blue', label='Deg-P')
    # Faixa de Incerteza Deg
    ax1.fill_between(t, rtlo.yL, rtlo.yU, color='cornflowerblue', alpha=0.25, label='Deg-U')
    ax1.plot(t, rtlo.yL, color='blue', linestyle='--', linewidth=0.5)
    ax1.plot(t, rtlo.yU, color='blue', linestyle='--', linewidth=0.5)
    
    ax1.set_title('Y - Real x Prediction')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 2: RUL - Real x Pred ---
    ax2.plot(t, rtlo.rulR, color='black', label='RUL-R')
    ax2.plot(t, rtlo.rulP, color='blue', label='RUL-P')
    
    # Faixa de Tolerância (T-25%)
    upper_tol = (1 + tol / 100) * rtlo.rulR
    lower_tol = (1 - tol / 100) * rtlo.rulR
    ax2.fill_between(t, lower_tol, upper_tol, color='gray', alpha=0.25, label=f'T-{tol}%')
    
    # Faixa de Incerteza RUL
    ax2.fill_between(t, rtlo.rulL, rtlo.rulU, color='cornflowerblue', alpha=0.25, label='RUL-U')
    ax2.plot(t, rtlo.rulL, color='blue', linestyle='--', linewidth=0.5)
    ax2.plot(t, rtlo.rulU, color='blue', linestyle='--', linewidth=0.5)
    
    ax2.set_title('RUL - Real x Pred')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 3: Error Analysis (Dois Eixos Y) ---
    # Eixo Esquerdo: RUL wMAPE
    ax3.plot(t, rtlo.εR_hist, color='black', label='RUL wMAPE')
    ax3.set_ylabel('RUL wMAPE', color='black', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='black')
    
    # Criando o Eixo Direito (Secundário) para Deg wMAPE
    ax3_sec = ax3.twinx()
    ax3_sec.plot(t, rtlo.εM_hist, color='blue', label='Deg wMAPE')
    ax3_sec.set_ylabel('Deg wMAPE', color='blue', fontweight='bold')
    ax3_sec.tick_params(axis='y', labelcolor='blue')
    
    ax3.set_title('Error Analysis')
    ax3.set_xlabel('Time / Index')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Ajuste de layout para não sobrepor títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save is not False:
        fig.savefig(save, dpi=500, transparent=False)
    plt.show()

def PlotSphere3D(points, centroid, R, title=None, w=600, h=600):
    """
    Plots a set of 3D points, a centroid, and a sphere of radius R centered at the centroid.
    
    Parameters:
    - points: A list or numpy array of 3D points, shape (N, 3).
    - centroid: A tuple or list representing the 3D centroid (x, y, z).
    - R: A float representing the radius of the sphere.
    """
    if title == None: 
        title="3D Points, Centroid, and Sphere"

    points = np.array(points)
    cx, cy, cz = centroid
    
    # 1. Generate parametric coordinates for the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    x_sphere = cx + R * np.outer(np.cos(u), np.sin(v))
    y_sphere = cy + R * np.outer(np.sin(u), np.sin(v))
    z_sphere = cz + R * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # 2. Add the sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, 
        y=y_sphere, 
        z=z_sphere,
        opacity=0.5, # Semi-transparent to see the points inside
        colorscale='Blues',
        showscale=False,
        name='Sphere',
        legendgroup='Sphere',
        showlegend=True
    ))

    # 3. Add the centroid point
    fig.add_trace(go.Scatter3d(
        x=[cx], 
        y=[cy], 
        z=[cz],
        mode='markers',
        marker=dict(size=6, symbol='x', color='red'),
        name='Centroid'
    ))

    # 4. Add the 3D points
    if points.size > 0:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], 
            y=points[:, 1], 
            z=points[:, 2],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.8),
            name='Points'
        ))

    # 5. Update layout to keep the proportions equal
    fig.update_layout(
        width=w, height=h,
        title=title,
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
            aspectmode='data' # Ensures the sphere looks perfectly round, not stretched
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def plot_single_df(df,j=0):
  fig=make_subplots(rows=1,cols=1)
  trace = go.Scatter(x=[i for i in range(len(df))], y = df.iloc[:,j])
  fig.add_trace(trace, row=1, col=1)
  fig.update_layout(width = 400, height = 400, title = df.columns[j])
  fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
  fig.show()

def PlotSeries3D(xSeries,zSeries,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
  x = [[g  for g in range(len(x))] for x in xSeries]
  y = []
  for i in range(len(xSeries)):
      y.append([i for j in range(len(xSeries[i]))])
  z = [z for z in zSeries]
  traces = []

  for i in range(len(x)):
      trace = go.Scatter3d(
          x=x[i],
          y=y[i],
          z=z[i],
          mode='lines',
          name=f'{i}',
          line=dict(color=z[i], colorscale='Viridis', width=2),
      )
      traces.append(trace)

  layout = go.Layout(
      title=title,
      scene=dict(
          xaxis=dict(title=xlabel),
          yaxis=dict(title=ylabel),
          zaxis=dict(title=zlabel),
          camera=dict(
              eye=dict(x=-1.25, y=-1.25, z=0.75)
          )
      ),
      width=width, height = height
  )

  fig = go.Figure(data=traces, layout=layout)
  fig.show()
  return traces

def plot_3d(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
  x = [[g  for g in range(len(df))] for i in range(len(df.columns))]
  y = []
  for i in range(len(df.columns)):
      y.append([i for j in range(len(df))])
  z = [df.iloc[:,i] for i in range(len(df.columns))]
  traces = []
  for i in range(len(x)):
      trace = go.Scatter3d(
          x=x[i],
          y=y[i],
          z=z[i],
          mode='lines',
          name=df.columns[i],
          line=dict(color=z[i], colorscale='Viridis', width=2),
      )
      traces.append(trace)

  layout = go.Layout(
      title=title,
      scene=dict(
          xaxis=dict(title=xlabel),
          yaxis=dict(title=ylabel),
          zaxis=dict(title=zlabel),
          camera=dict(
              eye=dict(x=-1.25, y=-1.25, z=0.75)
          )
      ),
      width=width, height = height
  )

  fig = go.Figure(data=traces, layout=layout)
  fig.show()
  return traces


def plot_3d_fft_2(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
    x = [df.iloc[:, 0] for i in range(1, len(df.columns))]
    y = []
    for i in range(1, len(df.columns)):
        y.append([i for j in range(len(df))])
    z = [df.iloc[:, i] for i in range(1, len(df.columns))]
    
    traces = []
    for i in range(len(x)):
        trace = go.Scatter3d(
            x=x[i],
            y=y[i],
            z=z[i],
            mode='lines',
            name=df.columns[i + 1],
            line=dict(color=z[i], colorscale='Viridis', width=2),
        )
        traces.append(trace)
    
    return traces
def plot_3d_inline(fft_r,env_r,title1 = 'df1',title2 = 'df2'):

    fig = make_subplots( rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=(title1, title2),)

    fft_traces = plot_3d_fft_2(fft_r, 500, 500, 'fft_r', 'r', 'r', 'r')
    for trace in fft_traces:
        fig.add_trace(trace, row=1, col=1)

    env_traces = plot_3d_fft_2(env_r, 500, 500, 'env_r', 'r', 'r', 'r')
    for trace in env_traces:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout( height=600, width=1200, title_text="Comparison of FFT_R and ENV_R in 3D")
    fig.show()


def plot_3d_fft(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
    x = [df.iloc[:,0] for i in range(1,len(df.columns))]
    y = []
    for i in range(1,len(df.columns)):
        y.append([i for j in range(len(df))])
    z = [df.iloc[:,i] for i in range(1,len(df.columns))]
    traces = []
    for i in range(len(x)):
        trace = go.Scatter3d(
            x=x[i],
            y=y[i],
            z=z[i],
            mode='lines',
            name=df.columns[i+1],
            line=dict(color=z[i], colorscale='Viridis', width=2),
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            zaxis=dict(title=zlabel),
            camera=dict(
                eye=dict(x=-1.25, y=-1.25, z=0.75)
            )
        ),
        width=width, height = height
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_dataframe2(dataframes,w=12,h=6):
    # Number of columns and rows for subplots
    num_cols = len(df.columns)
    num_plots_per_row = 4
    num_rows = (num_cols + num_plots_per_row - 1) // num_plots_per_row  # Ceiling division

    # Create subplots with the given number of rows and columns (4 per row)
    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(w,h))
    fig.suptitle("DataFrame Columns Plotted", fontsize=16)

    # Flatten axes array to easily iterate through (if needed for 2D grid of axes)
    axes = axes.flatten()

    # Generate x values (length of the DataFrame)
    x_values = list(range(len(df)))

    # Plot each column in a subplot
    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(x_values, df[col], label=label, color='blue')
        ax.plot(x_values, df2[col], label=label1, color='orange')
        ax.plot(x_values, df3[col], label=label2, color='green')
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)

    # Hide any unused subplots if the number of columns is not a multiple of 4
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()


def plot_features(df, cols_qtd=4, brng='brng', show=True, w=12, h=2.5,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-df.shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil

    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=column)
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)
        axes[i].legend()
        if i< nr-1: 
            axes[i].set_xlabel('')
            axes[i].tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)
            

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()

def plot_features2(df,df2, cols_qtd=4, brng='brng', show=True, w=12, h=6,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-df.shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    if show_title: fig.suptitle(f'Features - {brng}', fontsize=10)  # Título principal
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil

    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=df.columns[i])
        axes[i].plot(df2.index, df2[df2.columns[i]], label=df2.columns[i])
        axes[i].set_title(column, fontsize=10)
        axes[i].set_ylim(ylim[0],ylim[1])
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()

def plot_featuresN(dfs, cols_qtd=4, brng='brng', show=True, w=12, h=2.5,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-dfs[0].shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    if show_title: fig.suptitle(f'Features - {brng}', fontsize=10)  # Título principal
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil
    names = ['X','Y','Z']
    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(dfs[0].columns):
        for df,name in zip(dfs,names):
            axes[i].plot(df.index, df[df.columns[i]], label=name)

        axes[i].set_title(column, fontsize=10)
        axes[i].set_ylim(ylim[0],ylim[1])
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)
        axes[i].legend(fontsize=5)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()


def plot_metrics(df,out, cols_qtd, brng, show=True, w=12, h=6):
    """
    Plota métricas de Monotonicidade, Trendabilidade e Correlação no Matplotlib.

    Parâmetros:
    - df: DataFrame com os dados.
    - cols_qtd: Número de colunas de subplots.
    - brng: Nome do bearing (usado no título).
    - show: Se True, exibe o gráfico.
    - w, h: Dimensões da figura.
    
    Retorna:
    - df_r: DataFrame filtrado com colunas que atendem critérios (>0.5).
    """

    df_r = pd.DataFrame()
    categories = ['M', 'T', 'C']
    categories2 = ['Monotonicidade', 'Trendabilidade', 'Correlação']
    bar_colors = ['blue', 'orange', 'green']

    # 🔹 Determinar número de linhas e colunas
    cols_qtd = min(cols_qtd, df.shape[1])  # Limitar colunas ao máximo existente
    rows = -(-df.shape[1] // cols_qtd)  # Cálculo correto do número de linhas (ceil)

    # 🔹 Criar figura e eixos para subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols_qtd, figsize=(w, h))
    fig.suptitle(f'Rolamento{brng[7:-4]}: Métricas de Avaliação das Características',
                  fontsize=9, y =0.95)

    # 🔹 Achatar matriz de eixos para facilitar a iteração
    axes = np.array(axes).flatten()

    # 🔹 Loop sobre cada coluna do DataFrame
    for i, column in enumerate(df.columns):
        metrics = np.array([
            calculate_monotonicity(df[column].values),
            calculate_trendability(df[column].values),
            calculate_correlation(df[column].values)
        ])
        metrics = np.abs(metrics)

        # 🔹 Filtrar colunas com métricas > 0.5
        if metrics[0] > 0.5 and metrics[1] > 0.5:
            df_r[column] = df[column]

        # 🔹 Criar gráfico de barras
        axes[i].bar(categories, metrics, color=bar_colors, width=0.4,label = categories2)
        axes[i].set_ylim(0, 1)
        yticks = list(np.arange(0, 1.25, .25))
        axes[i].set_yticks(sorted(yticks))
        axes[i].set_yticklabels(sorted(yticks), color='black') 
        axes[i].set_title(column, fontsize=8)
        axes[i].tick_params(axis="x", labelrotation=0, labelsize=6)
        axes[i].tick_params(axis="y", labelrotation=0, labelsize=6)
        axes[i].grid(True, alpha=0.3, linestyle='--',linewidth=0.75,color='black')
        handles, labels = axes[i].get_legend_handles_labels() 
        if i == 12:
            axes[i].legend(handles[:1], labels[:1], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))
        if i == 13:
            axes[i].legend(handles[1:2], labels[1:2], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))
        if i == 14:
            axes[i].legend(handles[2:], labels[2:], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))

    # 🔹 Remover subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 🔹 Ajustar layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 🔹 Exibir gráfico
    if show:
        plt.savefig(out+f'{brng[:-4]}_Metrics.eps', dpi=500)
        plt.show()
        plt.close(fig)  # Liberar memória

    return df_r

def plot_multiple_features(dfs,out, cols_qtd, brngs, labels, show=True, w=12, h=6):
    names = [brngs[i][7:-4] for i in range(len(brngs))]
    """
    Plota múltiplas séries temporais de até três DataFrames em um conjunto de subplots.

    Parâmetros:
    - dfs: Lista contendo os DataFrames a serem comparados (exatamente 3).
    - cols_qtd: Número de colunas de subplots.
    - brngs: Lista com os nomes dos bearings (usado no título).
    - labels: Lista com rótulos para os DataFrames na legenda.
    - show: Se True, exibe o gráfico.
    - w: Largura da figura.
    - h: Altura da figura.

    Retorna:
    - None (exibe o gráfico).
    """

    # Pegar as colunas do primeiro DataFrame (assume que os 3 têm as mesmas colunas)
    columns = dfs[0].columns
    nc = cols_qtd  # Número de colunas de subplots
    nr = -(-len(columns) // nc)  # Cálculo correto do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    fig.suptitle(f'Comparação de Características - Rolamentos{", ".join(names[:-1])} e {names[-1]} '
                 , fontsize=9, y =0.95)


    # Garantir que `axes` seja uma matriz 2D
    axes = np.array(axes).flatten()

    # 🔹 Loop sobre cada coluna para plotar os três DataFrames no mesmo subplot
    for i, column in enumerate(columns):
        for df, label in zip(dfs, labels):
            #print(label)
            axes[i].plot(df.index, df[column], label=label)
        #print(column)
        axes[i].set_title(column, fontsize=8)
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=10)
        axes[i].legend(fontsize=5,loc="center right")
        axes[i].set_xlabel("Ciclo", fontsize=5) 
        axes[i].set_ylabel("Magnitude", fontsize=5) 
        #axes[i].set_xlim(0, 123)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    # Exibir o gráfico se necessário
    if show:
        plt.savefig(out+'Features.eps', dpi=500)
        plt.savefig(out+'Features.png', dpi=500)
        plt.show()

'''def PlotDataframes(dataframes, cols_qtd=4,w=12,h=8):

    cmap = mpl.colormaps.get_cmap('tab20').resampled(len(dataframes))
    colors = (np.array([cmap(i) for i in range(1+len(dataframes))]))

    nc = cols_qtd  # Número de colunas
    nr = -(-dataframes[0].shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    wspace=0.001, hspace=0.001)
    fig.add_gridspec(wspace=0.25, hspace=0.15)
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten() 
    # Plot each column in a subplot
    for i, col in enumerate(dataframes[0].columns):
        ax = axes[i]
        for j,df in enumerate(dataframes):
            ax.plot(df.index, df[col], label=f'{j+1}', color=colors[j])
        #ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8,ncol=len(dataframes))
        ax.grid(True)
        if i< nr*nc-cols_qtd: 
            ax.set_xlabel('')
            ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)

    # Hide any unused subplots if the number of columns is not a multiple of 4
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()'''

def PlotFramesPLT(dataframes,names=None,cols_qtd=4,w=11,h=1.5):
    cmap = mpl.colormaps.get_cmap('tab20').resampled(len(dataframes))
    colors = (np.array([cmap(i) for i in range(1+len(dataframes))]))
    if names is None:
        names = [f'S{i+1}' for i in range(len(dataframes))]
    nc = cols_qtd  # Número de colunas
    nr = -(-dataframes[0].shape[1] // nc)  # Cálculo do número de linhas (ceil)
    h = h*nr
    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    #wspace=0.5, hspace=0.001
                    )
    fig.add_gridspec(wspace=0.05, hspace=0.05)
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten() 
    # Plot each column in a subplot
    for i, col in enumerate(dataframes[0].columns):
        ax = axes[i]
        for j,df in enumerate(dataframes):
            ax.plot(df.index, df[col], label=names[j], color=colors[j])
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=7,ncol=len(dataframes))
        ax.grid(True)
        if i< nr*nc-cols_qtd: 
            ax.set_xlabel('')
            ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

def PlotFramesPLY(dataframes, names=None, cols_qtd=4, w=1100, h=200):
    """
    Plota colunas de múltiplos DataFrames em uma grade de subplots com Plotly.
    
    Parameters:
    -----------
    dataframes : list[pd.DataFrame]
        Lista de DataFrames a serem plotados.
    names : list[str], optional
        Nomes das séries para a legenda.
    cols_qtd : int
        Número de colunas no layout de subplots.
    w : int
        Largura total da figura em pixels.
    h : int
        Altura base por LINHA em pixels.
    """
    if names is None:
        names = [f'S{i+1}' for i in range(len(dataframes))]
        
    # Gera a paleta de cores proporcional ao número de DataFrames
    palette = px.colors.qualitative.Dark24
    colors = [palette[j % len(palette)] for j in range(len(dataframes))]
    
    # Número de variáveis/colunas a plotar
    num_vars = dataframes[0].shape[1]
    
    # Cálculo do número de linhas necessárias
    nc = cols_qtd
    nr = int(np.ceil(num_vars / nc))
    
    # Títulos para cada subplot com base nos nomes das colunas
    subplot_titles = [str(col) for col in dataframes[0].columns]
    
    # Cria a estrutura de subplots
    fig = make_subplots(
        rows=nr, 
        cols=nc, 
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Itera sobre cada coluna do DataFrame
    for i, col in enumerate(dataframes[0].columns):
        # Determina a linha e coluna (índices 1-based para o Plotly)
        row = (i // nc) + 1
        col_idx = (i % nc) + 1
        
        # Plota a linha correspondente de CADA DataFrame no subplot atual
        for j, df in enumerate(dataframes):
            # A legenda só deve ser registrada no primeiro subplot (i == 0) para não duplicar
            show_in_legend = (i == 0)
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=names[j],
                    legendgroup=names[j],  # Agrupa as séries de mesmo nome entre subplots
                    showlegend=show_in_legend,
                    line=dict(color=colors[j], width=1.5)
                ),
                row=row, 
                col=col_idx
            )
            
    # Configurações do layout geral
    fig.update_layout(
        width=w,
        height=h * nr,  # Ajusta a altura dinâmica proporcionalmente às linhas
        template="plotly_white",
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ajusta os títulos dos subplots para um tamanho menor (similar ao fontsize=8)
    fig.for_each_annotation(lambda a: a.update(font=dict(size=10)))
    
    # Adiciona linhas de grade suaves a todos os eixos
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')

    return fig

def PlotSTFT(x,fs,nperseg=1024,noverlap=None):
    
    t = np.arange(len(x))*(1/fs)

    x_env = hilbert(x)
    x_env = np.abs(x_env)
    f, t_segment, Zxx = stft(x, fs=fs, nperseg=nperseg,noverlap=noverlap)
    f2, t_segment2, Zxx2 = stft(x_env, fs=fs, nperseg=nperseg,noverlap=noverlap)
    Zxx = np.abs(Zxx)
    Zxx2 = np.abs(Zxx2)
    # 2. Criação da Estrutura com Subplots Lado a Lado
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True,
        subplot_titles=('Espectrograma STFT (Sinal Bruto)', 'Espectrograma STFT (Envelope)'),
        horizontal_spacing=0.08
    )

    # 3. Adiciona o primeiro Espectrograma (Sinal Original)
    fig.add_trace(
        go.Heatmap(
            z=Zxx,
            x=t_segment,
            y=f,
            colorscale='Viridis',
            zsmooth='fast',  # Equivalente ao shading='gouraud' do Matplotlib
            showscale=False  # Mantém a Colorbar apenas no segundo subplot para não poluir
        ),
        row=1, col=1
    )

    # 4. Adiciona o segundo Espectrograma (Envelope)
    fig.add_trace(
        go.Heatmap(
            z=Zxx2,
            x=t_segment2,
            y=f2,
            colorscale='Viridis',
            zsmooth='fast',
            colorbar=dict(
                title='Magnitude',
                len=1.0  # Ajusta a altura da barra lateral de cores
            )
        ),
        row=1, col=2
    )

    # 5. Configuração Geral do Layout e Rótulos dos Eixos
    fig.update_layout(
        title_text="Análise de Espectrogramas STFT",
        template="plotly_white",
        width=1200,
        height=500,
        margin=dict(t=80, b=50, l=60, r=60)
    )

    # Rótulos dos Eixos X e Y para ambos os subplots
    fig.update_xaxes(title_text="Tempo (Segundos)", row=1, col=1)
    fig.update_xaxes(title_text="Tempo (Segundos)", row=1, col=2)

    fig.update_yaxes(title_text="Frequência (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Frequência (Hz)", row=1, col=2)

    fig.show()
    