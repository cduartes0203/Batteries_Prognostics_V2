
import numpy as np
import pandas as pd
import pickle

from scipy.stats import entropy
from scipy.signal import hilbert
import os
import re
import scipy.stats as stats
import math
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, median_filter
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fs=32000

def NormalizeSeries(x,lB=0,uB=1):
    xNorm = ((uB-lB)*((x-np.min(x))/(np.max(x)-np.min(x))))+lB
    return xNorm

def signal_entropy(signal, bins=50):
    """Calcula a entropia de um sinal transformando-o em distribuição de probabilidades."""
    
    # Criar histograma normalizado (estimativa de distribuição de probabilidades)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    
    # Remover valores zero para evitar log(0)
    hist = hist[hist > 0]

    # Calcular entropia de Shannon
    return entropy(hist, base=3)  # Base 2 para entropia em bits (opcional)

def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

# filters ###############################################################################################################################

def moving_average_df(df, window_size):

    return df.apply(lambda col: np.convolve(col, np.ones(window_size) / window_size, mode='same'))

def exponential_moving_average(data, alpha=0.3):
    ema = [data.iloc[0]]  # Start with the first value
    for i in range(1, len(data)):
        ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
    return pd.Series(ema, index=data.index)

def exponential_moving_average_df(df, alpha=0.3):
    ema_df = df.copy()
    for col in df.columns:
        ema_df[col] = exponential_moving_average(df[col], alpha)
    return ema_df

# features ###############################################################################################################################

def calculate_entropy(vector):
    contagem = Counter(vector)
    total = len(vector)
    # Calcular a probabilidade de cada elemento
    probabilidades = np.array([freq / total for freq in contagem.values()])
    # Aplicar a fórmula da entropia de Shannon
    entropia = -np.sum(probabilidades * np.log10(probabilidades))
    
    return entropia

def calculate_crest_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    peak = np.max(np.abs(vector))
    return peak / rms

def calculate_wave_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    mean = np.mean(np.abs(vector))
    return rms / mean

def calculate_impulse_factor(vector):
    peak = np.max(np.abs(vector))
    mean = np.mean(np.abs(vector))
    return peak / mean

def calculate_margin_factor(vector):
    peak = np.max(np.abs(vector))
    rms = np.sqrt(np.mean(np.square(vector)))
    return peak / (rms ** 2)

def time_features(vec):
    vec =vec - np.mean(vec)
    mn = (np.mean(vec))
    mn_abs = (np.mean(np.abs(vec)))
    var = (np.var(vec))
    std = math.sqrt(var)
    skw = (stats.skew(vec))
    krt = (stats.kurtosis(vec))
    ntrpy = signal_entropy(vec)
    rms = (np.sqrt(np.mean(np.square(np.array(vec)))))
    max = np.max(vec)
    min = np.min(vec)
    ptp = (np.ptp(vec))

    features = [mn,mn_abs,var,std,skw,krt,ntrpy,rms,max,min,ptp]
    column_names =['Média','Média Absoluta','Variância','Desvio Parão',
                   'Assimetria','Kurtose','Entropia','RMS','Máximo','Mínimo','Pico-a-pico']
    return features, column_names
def freq_features(vec,rate=fs):
    vec = vec - np.mean(vec)
    freq, fft1 = envelope(vec,rate)
    fft = fft1[(freq>=1) & (freq<=10000)].copy()
    freq = freq[(freq>=1) & (freq<=10000)].copy()
    energy = np.sum(fft**2)
    mn_energy = (np.mean(fft**2))
    var_energy = (np.var(fft**2))
    std_energy = math.sqrt(var_energy)
    skw_energy= (stats.skew(fft**2))
    ntrpy = signal_entropy(fft**2)
    krt_energy = (stats.kurtosis(fft**2))
    energy_density = np.sum((fft)**2)/np.sum(fft1**2)
    rms_energy = (np.sqrt(np.mean(np.square(np.array(fft**2)))))

    features = [energy,mn_energy,var_energy,std_energy,
                skw_energy,ntrpy,krt_energy,energy_density,rms_energy]
    column_names =['Energia','Energia Média','Variância da Energia','Desvio Padrão da energia'
                   ,'Assimeria da Energia','Entropia da Energia','Kurtose da Energia','Densidade da Energia','RMS da Energia']
    return features, column_names


# fft calculation ########################################################################################################################
def fft_calc(signal,rate):
  fft_shft_abs = np.fft.fftshift(2.0 * np.abs(np.fft.fft(signal) / np.size(np.fft.fft(signal))))
  freq_shft = np.fft.fftshift(np.fft.fftfreq(np.size(fft_shft_abs), 1/rate))
  return freq_shft[(freq_shft>=0) & (freq_shft<=rate/2)], fft_shft_abs[(freq_shft>=0) & (freq_shft<=rate/2)]

def df_fft(df,rate):
    fft = df.copy()
    for i in range(len(fft.columns)):
        freq_shft, fft_shft_abs = fft_calc(fft.iloc[:,i], rate)
        fft.iloc[:,i] = fft_shft_abs

    fft.insert(0, 'FREQ',freq_shft)
    return fft

# envelope ########################################################################################################################

def envelope(signal, rate):
  amplitude_envelope = np.abs(hilbert(signal))
  amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
  fft_hilbert = np.fft.fftshift(2.0 * np.abs(np.fft.fft(amplitude_envelope) / np.size(np.fft.fft(amplitude_envelope))))
  freq_hilbert = np.fft.fftshift(np.fft.fftfreq(np.size(fft_hilbert), 1/rate))
  return freq_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)], fft_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)]  


def df_env(df,rate):
    env = df.copy()
    for i in range(len(env.columns)):
        freq_shft, fft_shft_abs = envelope(env.iloc[:,i], rate)
        env.iloc[:,i] = fft_shft_abs

    env.insert(0, 'FREQ',freq_shft)
  
    return env

# FILTER ########################################################################################################################

def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """
    Aplica um filtro passa-banda em um sinal.

    Parâmetros:
    - signal_data: np.array -> Sinal de entrada
    - lowcut: float -> Frequência de corte inferior (Hz)
    - highcut: float -> Frequência de corte superior (Hz)
    - fs: float -> Frequência de amostragem (Hz)
    - order: int -> Ordem do filtro (padrão: 4)

    Retorna:
    - Sinal filtrado
    """

    nyquist = 0.5 * fs  # Frequência de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist

    # Criando filtro Butterworth passa-banda
    b, a = signal.butter(order, [low, high], btype='band')

    # Aplicando filtro no sinal
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal

# metrics ########################################################################################################################

def join_path(dirs):
    path = os.path.join(dirs[0], dirs[1])
    if len(dirs)==2:
        return path
    elif len(dirs)>2:
        for i in range(2,len(dirs)):
            path = os.path.join(path, dirs[i])
        return path

def sort_files(path):
    samples = os.listdir(path)
    samples.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return samples

def signal_entropy(signal, bins=50):
    """Calcula a entropia de um sinal transformando-o em distribuição de probabilidades."""
    
    # Criar histograma normalizado (estimativa de distribuição de probabilidades)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    
    # Remover valores zero para evitar log(0)
    hist = hist[hist > 0]

    # Calcular entropia de Shannon
    return entropy(hist, base=3)  # Base 2 para entropia em bits (opcional)

def pad_vector(*vetores):

    max_len = max(len(v) for v in vetores)
    vetores_preenchidos = [np.pad(v, (0, max_len - len(v)), mode='constant') for v in vetores]
    return np.sum(vetores_preenchidos, axis=0)

def pad_matrix(matrix, max_rows, max_cols):
    return np.pad(matrix, ((0, max_rows - matrix.shape[0]), (0, max_cols - matrix.shape[1])), mode='constant')

import numpy as np

def sum_matrix(m1, m2):
    """
    Soma duas matrizes NumPy de tamanhos diferentes coordenada por coordenada.

    Parâmetros:
    - m1: np.array, primeira matriz.
    - m2: np.array, segunda matriz.

    Retorna:
    - np.array: Matriz resultante da soma.
    """
    # Encontrar o maior número de linhas e colunas
    max_rows = max(m1.shape[0], m2.shape[0])
    max_cols = max(m1.shape[1], m2.shape[1])

    # Preencher as matrizes menores com zeros para igualar os tamanhos
    m1_padded = np.pad(m1, ((0, max_rows - m1.shape[0]), (0, max_cols - m1.shape[1])), mode='constant')
    m2_padded = np.pad(m2, ((0, max_rows - m2.shape[0]), (0, max_cols - m2.shape[1])), mode='constant')

    # Somar coordenada por coordenada
    return m1_padded + m2_padded

def normalize_cstm(sig,max,min):
    sig = 2*(sig - min)/(max - min)-1
    return sig

def calculate_correlation(column):
    f = np.array(column)
    t = np.array([(i+1) for i in range(len(f))])
    num = sum([
        abs((f[i] - f.mean()) * (t[i] - t.mean())) for i in range(len(f))
    ])
    den = math.sqrt(sum([(f[i] - f.mean())**2 for i in range(len(f))]) * sum([(t[i] - t.mean())**2 for i in range(len(t))]))

    if den == 0:
        return 0
    if num == 0:
        return 0
    else:
        return num/den


def calculate_monotonicity(column):
    # Calculate the difference between consecutive elements
    diffs = np.diff(column)

    # Count positive, negative, and zero differences
    positive_diffs = np.sum(diffs > 0)
    negative_diffs = np.sum(diffs < 0)
    total_diffs = len(diffs)
    
    if total_diffs == 0:
        return 0  # Handle case of single element or empty column
    
    # Calculate monotonicity score as a normalized value between -1 and 1
    monotonicity_score = abs((positive_diffs - negative_diffs) / total_diffs)
    
    return monotonicity_score

def rank_sequence(vector):
    """
    Returns the rank sequence of a vector.

    Parameters:
    - vector (list or numpy array): The input vector.

    Returns:
    - numpy array: An array representing the rank of each element in the vector.
    """
    # Convert the vector to a numpy array if it isn't already
    vector = np.array(vector)
    
    # Get the sorted indices
    sorted_indices = np.argsort(vector)
    
    # Create an array to hold the ranks
    ranks = np.empty_like(sorted_indices)
    
    # Assign ranks based on sorted indices
    ranks[sorted_indices] = np.arange(len(vector))
    
    return ranks + 1

def calculate_trendability(x):
    t = np.array([i+1 for i in range(len(x))])

    K = len(x)
    s_xt = np.sum(x*t)
    s_x = np.sum(x)
    s_x2 = np.sum(x**2)
    s_t = np.sum(t)
    s_t2 = np.sum(t**2)
    
    n = (K*s_xt) -(s_x*s_t)
    if n == 0:
        return 0
    
    d =math.sqrt(((K*s_x2)-(s_x**2))*((K*s_t2)-(s_t**2)))
    
    if d == 0:
        return 0
    
    return (n/d)

def calculate_robustness(column, window_size=3):
    # Step 1: Compute the smoothed values (mean trend) using a rolling window (moving average)
    smoothed_column = column.rolling(window=window_size, center=True, min_periods=1).mean()

    # Step 2: Initialize variables for robustness calculation
    K = len(column)
    robustness_sum = 0
    
    # Step 3: Calculate robustness using the given formula
    for xk, xTk in zip(column, smoothed_column):
        if xk != 0:  # Avoid division by zero
            robustness_sum += np.exp(-(abs(xk - xTk) / abs(xk)))
    
    # Step 4: Compute the final robustness value
    robustness = robustness_sum / K
    
    return robustness

def calculate_criteria(column):
    cri = (calculate_correlation(column) + calculate_monotonicity(column))*0.5
    return cri

# dataframe ###############################################################################################################################

def sqr_df(df):
    aux = df.copy()
    for i in range(len(aux.columns)):
        if i!=0:
            aux[aux.columns[i]] = aux[aux.columns[i]] * aux[aux.columns[i]]
    return aux

def normalize_df(df, lB=0, uB=1):
    for col in df.columns:
        df[col] = NormalizeSeries(df[col], lB, uB)
    return df

# filters ###############################################################################################################################

def moving_average_df(df, window_size):

    return df.apply(lambda col: np.convolve(col, np.ones(window_size) / window_size, mode='same'))

def exponential_moving_average(data, alpha=0.3):
    ema = [data.iloc[0]]  # Start with the first value
    for i in range(1, len(data)):
        ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])
    return pd.Series(ema, index=data.index)

def exponential_moving_average_df(df, alpha=0.3):
    ema_df = df.copy()
    for col in df.columns:
        ema_df[col] = exponential_moving_average(df[col], alpha)
    return ema_df

def gaussian_smoothing(data, sigma=2):
    return pd.Series(gaussian_filter1d(data, sigma=sigma))

def savitzky_golay_smoothing(data, window_size=5, poly_order=2):
    return pd.Series(savgol_filter(data, window_length=window_size, polyorder=poly_order))

def median_smoothing(data, window_size=3):
    return pd.Series(median_filter(data, size=window_size))

# Apply smoothing to all columns
def apply_smoothing(df, smoothing_function, **kwargs):
    smoothed_df = df.copy()
    for col in df.columns:
        smoothed_df[col] = smoothing_function(df[col], **kwargs)
    return smoothed_df

# features ###############################################################################################################################

def calculate_entropy(vector):
    contagem = Counter(vector)
    total = len(vector)
    # Calcular a probabilidade de cada elemento
    probabilidades = np.array([freq / total for freq in contagem.values()])
    # Aplicar a fórmula da entropia de Shannon
    entropia = -np.sum(probabilidades * np.log10(probabilidades))
    
    return entropia

def calculate_crest_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    peak = np.max(np.abs(vector))
    return peak / rms

def calculate_wave_factor(vector):
    rms = np.sqrt(np.mean(np.square(vector)))
    mean = np.mean(np.abs(vector))
    return rms / mean

def calculate_impulse_factor(vector):
    peak = np.max(np.abs(vector))
    mean = np.mean(np.abs(vector))
    return peak / mean

def calculate_margin_factor(vector):
    peak = np.max(np.abs(vector))
    rms = np.sqrt(np.mean(np.square(vector)))
    return peak / (rms ** 2)

def time_features(vec):
    mn = (np.mean(vec))
    mn_abs = (np.mean(np.abs(vec)))
    var = (np.var(vec))
    std = math.sqrt(var)
    skw = (stats.skew(vec))
    krt = (stats.kurtosis(vec))
    ntrpy = signal_entropy(vec)
    rms = (np.sqrt(np.mean(np.square(np.array(vec)))))
    max = np.max(vec)
    min = np.min(vec)
    ptp = (np.ptp(vec))
    crest = np.max(vec) / rms
    zcr = np.count_nonzero(np.diff(np.sign(vec)))
    ff = rms / np.mean(np.abs(vec))

    features = [mn,mn_abs,var,std,skw,krt,ntrpy,rms,max,min,ptp,crest,zcr,ff]
    column_names = ['Média', 'Média Absoluta', 'Variância', 'Desvio Padrão', 
                    'Assimetria', 'Kurtose', 'Entropia', 'RMS', 'Máximo','Mínimo', 'Pico-a-pico',
                    'Fator de Cresta', 'Taxa de cruzamento por zero', 'Fator de Forma']
    return features, column_names
def freq_features(vec,rate=fs):
    freq, fft1 = envelope(vec,rate)
    fft = fft1[(freq>=1) & (freq<=1500)].copy()
    freq = freq[(freq>=1) & (freq<=1500)].copy()
    energy = np.sum(fft**2)
    mn_energy = (np.mean(fft**2))
    var_energy = (np.var(fft**2))
    std_energy = math.sqrt(var_energy)
    skw_energy= (stats.skew(fft**2))
    ntrpy = signal_entropy(fft**2)
    krt_energy = (stats.kurtosis(fft**2))
    energy_density = np.sum((fft)**2)/np.sum(fft1**2)
    rms_energy = (np.sqrt(np.mean(np.square(np.array(fft**2)))))
    peak_frequency = freq[np.argmax(fft**2)]
    spectral_centroid = np.sum(freq * fft**2) / np.sum(fft**2)
    spectral_bandwidth = np.sqrt(np.sum((freq - spectral_centroid)**2 * fft**2) / np.sum(fft**2))
    spectral_flatness = np.sum(fft**2) / np.sum(np.abs(fft)**2)
    spectral_rolloff = freq[np.where(np.cumsum(fft**2) >= 0.85 * np.sum(fft**2))[0][0]]
    peak_to_rms = np.max(fft) / rms_energy



    features = [energy, mn_energy, var_energy, std_energy, skw_energy, ntrpy, krt_energy, energy_density, rms_energy,
                peak_frequency, spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff, peak_to_rms]
    
    column_names = ['Energia', 'Energia Média', 'Variância da Energia', 'Desvio Padrão da Energia', 'Assimetria da Energia', 
                    'Entropia da Energia', 'Kurtose da Energia', 'Densidade da Energia', 'RMS da Energia', 
                    'Frequência de Pico', 'Centroide Espectral', 'Largura de Banda Espectral', 'Planicidade Espectral', 
                    'Corte Espectral', 'Razão Pico para RMS']
    return features, column_names

def calculate_rs(df1, df2,j):
    num, aux1, aux2 = 0, 0, 0
    for i in range(len(df1)):
        num = num + ((df1.iloc[i,j] - df1.iloc[:,j].mean()) * (df2.iloc[i,j]- df2.iloc[:,j].mean()))
        aux1 = aux1 + (df1.iloc[i,j] - df1.iloc[:,j].mean())**2
        aux2 = aux2 + (df2.iloc[i,j] - df2.iloc[:,j].mean())**2
    rs = abs(num)/math.sqrt(aux1*aux2)
    #print(rs)
    return rs

'''def related_similarity(x, y):
    num, aux1, aux2 = 0, 0, 0
    for i in range(len(x)):
        num = num + (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        aux1 = aux1 + (x[i] - np.mean(x)) **2 
        aux2 = aux2 + (y[i] - np.mean(y)) **2
    return abs(num)/math.sqrt(aux1*aux2)'''

def related_similarity(x, y):
    """
    Calcula a similaridade entre dois vetores baseada no coeficiente de correlação de Pearson.

    Retorna:
    - O valor absoluto do coeficiente de correlação de Pearson entre x e y.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))  # Covariância
    aux1 = np.sum((x - x_mean) ** 2)  # Variância de x
    aux2 = np.sum((y - y_mean) ** 2)  # Variância de y

    # Evitar divisão por zero
    if aux1 == 0 or aux2 == 0:
        return 0  

    return abs(num) / math.sqrt(aux1 * aux2)  # Retorna o valor absoluto

# fft calculation ########################################################################################################################
def fft_calc(signal,rate):
  fft_shft_abs = np.fft.fftshift(2.0 * np.abs(np.fft.fft(signal) / np.size(np.fft.fft(signal))))
  freq_shft = np.fft.fftshift(np.fft.fftfreq(np.size(fft_shft_abs), 1/rate))
  return freq_shft[(freq_shft>=0) & (freq_shft<=rate/2)], fft_shft_abs[(freq_shft>=0) & (freq_shft<=rate/2)]
  #return freq_shft, fft_shft_abs

def df_fft(df,rate):
    fft = pd.DataFrame()
    for i in range(len(df.columns)):
        freq_shft, fft_shft_abs = fft_calc(df.iloc[:,i], rate)
        fft.insert(i, f'Mode_{i}',fft_shft_abs)

    fft.insert(0, 'FREQ',freq_shft)
    return fft

# envelope ########################################################################################################################

def envelope(signal, rate):
  amplitude_envelope = np.abs(hilbert(signal))
  amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
  fft_hilbert = np.fft.fftshift(2.0 * np.abs(np.fft.fft(amplitude_envelope) / np.size(np.fft.fft(amplitude_envelope))))
  freq_hilbert = np.fft.fftshift(np.fft.fftfreq(np.size(fft_hilbert), 1/rate))
  #return freq_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)], fft_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)]  
  return freq_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)], fft_hilbert[(freq_hilbert>=0) & (freq_hilbert<=rate/2)]  


def df_env(df,rate):
    env = pd.DataFrame()
    for i in range(len(df.columns)):
        freq_shft, fft_shft_abs = envelope(df.iloc[:,i], rate)
        env.insert(i, f'Mode_{i}',fft_shft_abs)

    env.insert(0, 'FREQ',freq_shft)
  
    return env


# FILTER ########################################################################################################################

def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """
    Aplica um filtro passa-banda em um sinal.

    Parâmetros:
    - signal_data: np.array -> Sinal de entrada
    - lowcut: float -> Frequência de corte inferior (Hz)
    - highcut: float -> Frequência de corte superior (Hz)
    - fs: float -> Frequência de amostragem (Hz)
    - order: int -> Ordem do filtro (padrão: 4)

    Retorna:
    - Sinal filtrado
    """

    nyquist = 0.5 * fs  # Frequência de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist

    # Criando filtro Butterworth passa-banda
    b, a = signal.butter(order, [low, high], btype='band')

    # Aplicando filtro no sinal
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal

# metrics ########################################################################################################################

def sum_non_zero_intervals(vector):
    # Initialize variables
    sums = []
    current_sum = 0
    in_interval = False

    # Iterate through the vector
    for value in vector:
        if value != 0:
            current_sum += value
            
            in_interval = True
        elif in_interval:
            sums.append(current_sum)
            current_sum = 0
            in_interval = False

    # Add the last interval if it ends at the end of the vector
    if in_interval:
        sums.append(current_sum)

    return sums

def harmonics_area_sum(df):
    df_copy = df.copy()
    df_r = pd.DataFrame()
    columns_name = df_copy.columns
    del df_copy[columns_name[0]]
    
    for i in range(len(df_copy.columns)):
       df_r[df_copy.columns[i]] = ((sum_non_zero_intervals(df_copy.iloc[:,i])))
    
    return df_r

# PCA ########################################################################################################################

def df_PCA(dados, n_componentes=2):
    """
    Aplica PCA em um conjunto de dados e funde as features reduzidas.
    
    Parâmetros:
    - dados (pd.DataFrame ou np.array): Matriz de dados (amostras x features).
    - n_componentes (int): Número de componentes principais a manter.
    
    Retorno:
    - df_reduzido (pd.DataFrame): Novo DataFrame com as features fundidas.
    - pca (PCA): Objeto PCA treinado.
    """
    
    # Normalizando os dados
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)

    # Aplicando PCA
    pca = PCA(n_components=n_componentes)
    dados_pca = pca.fit_transform(dados_normalizados)

    # Criando um DataFrame com as features reduzidas
    colunas_pca = [f"PC{i+1}" for i in range(n_componentes)]
    df_reduzido = pd.DataFrame(dados_pca, columns=colunas_pca)

    return df_reduzido, pca

def df_DropCols(df,cols=[None]):
    for df_col in df.columns:
        for col in cols:
            if df_col.endswith(col):
                df=df.drop(columns=df_col)
    return df

def process_RS(df_RS,nRS):
    df_RS = df_RS.copy()
    indx,vec = [],[]
    df_RS['i'] = df_RS.index.values
    df = df_RS.iloc[:,:-1].values
    for i in range(len(df)-nRS):
        vec.append(df[i:nRS+i].flatten())
        indx.append(df_RS.index.values[i:nRS+i][-1])
    vec.append(df[-nRS:].flatten())
    indx.append(df_RS.index.values[-nRS:][-1])
    return np.array(vec)

def process_HI(df_HI,nHI,init):
    hi = df_HI.iloc[:,0].values
    HI = []
    for i in range(len(hi)-init):
        x = hi[i:init+i]
        HI.append(x[-nHI:])
    return np.array(HI)

def process_input(df_RS,df_HI,nRS,nHI=None):
    if nHI is None: nHI = nRS
    RS = process_RS(df_RS,nRS)
    hi = df_HI.iloc[:,0].values
    HI = []
    for i in range(len(hi)-nRS):
        HI.append(hi[i:nRS+i])
    HI.append(hi[-nRS:])
    HI = np.array(HI).T
    HI = HI[-nHI:].T
    return RS, HI

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def PrepareData(df_RS, df_HI, nX, nY, mY=0, nZ=1, mode='past'):

    if mode == 1:
        mode = 'ahead'
    elif mode == 0:
        mode = 'past'
    """
    Prepara e alinha as janelas temporais para os modelos de predição (X, Y) e clusterização (Z).
    
    Parâmetros:
    -----------
    df_RS : pd.DataFrame ou np.ndarray (K x 3)
        Série temporal multidimensional para clusterização.
    df_HI : pd.DataFrame, pd.Series ou np.ndarray (K x 1)
        Série temporal monodimensional para predição autoregressiva.
    nX : int
        Tamanho da janela de dados passados para o input X.
    nY : int
        Tamanho da janela de dados futuros para o output Y (modo 'ahead') ou tamanho total de Y (modo 'past').
    mY : int, default=0
        Quantidade de dados PASSADOS adicionados à janela Y no modo 'ahead'. Ignorado no modo 'past'.
    nZ : int, default=1
        Tamanho da janela de dados passados para Z.
    mode : str, 'past' ou 'ahead'
        Estratégia de construção da janela de saída Y.
        
    Retorna:
    --------
    Zs, Xs, Ys : np.ndarray
        Arrays contendo as janelas alinhadas.
    """
    ST = np.array(df_HI).ravel()
    RS = np.array(df_RS)
    K = len(ST)

    # 1. Determina o menor k possível com base nas janelas passadas (nX, nZ e mY se 'ahead')
    if mode == 'ahead':
        k_inicio = max(nX, nZ, mY) - 1
        k_fim = K - nY - 1  # Garante que k + nY não ultrapasse K - 1
    elif mode == 'past':
        if nY > nX:
            raise ValueError("Na abordagem 'past', nY deve ser menor ou igual a nX.")
        k_inicio = max(nX, nZ) - 1
        k_fim = K - 2       # Garante que k + 1 não ultrapasse K - 1
    else:
        raise ValueError("Modo inválido. Escolha 'past' ou 'ahead'.")

    # Verifica se há dados suficientes para gerar ao menos uma janela
    if k_inicio > k_fim:
        raise ValueError(f"Série temporal muito curta ({K} amostras) para os parâmetros fornecidos.")

    Xs, Ys, Zs = [], [], []

    # 2. Construção e Alinhamento das Janelas
    for k in range(k_inicio, k_fim + 1):
        # Input X_k: dados de ST no intervalo [k - nX + 1, k]
        X_k = ST[k - nX + 1 : k + 1]
        
        # Output Y_k
        if mode == 'past':
            # Y_k vai de (k - nY + 2) até (k + 1) inclusive
            Y_k = ST[k - nY + 2 : k + 2]
        else:  # mode == 'ahead'
            # Se mY > 0: pega de [k - mY + 1] até [k + nY]
            # Se mY = 0: cai no comportamento original [k + 1] até [k + nY]
            Y_k = ST[k - mY + 1 : k + 1 + nY]

        # Input Z_k para Clusterização
        if nZ == 1:
            Z_k = RS[k]
        else:
            Z_k = RS[k - nZ + 1 : k + 1]

        Xs.append(X_k)
        Ys.append(Y_k)
        Zs.append(Z_k.flatten())

    # Preservando o retorno original (retirando a última amostra como no seu código)
    return np.array(Zs)[:-1], np.array(Xs)[:-1], np.array(Ys)[:-1]