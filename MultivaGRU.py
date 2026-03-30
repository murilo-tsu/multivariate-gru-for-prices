# =============================================================================
# CONFIGURAÇÕES DE AMBIENTE 
# =============================================================================
import os
SEED = 1
os.environ['PYTHONHASHSEED']        = str(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
os.environ['TF_DETERMINISTIC_OPS']  = '1'

# =============================================================================
# BIBLIOTECAS CIENTÍFICAS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import scipy.stats as stats
from scipy import signal
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =============================================================================
# BIBLIOTECAS PADRÃO
# =============================================================================
import random
import warnings
import logging
warnings.filterwarnings('ignore')
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# TENSORFLOW 
# =============================================================================
import tensorflow as tf
tf.random.set_seed(SEED)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, BatchNormalization

# =============================================================================
# CARREGAMENTO DA BASE DE DADOS | DataFrame fornecido por Market Intelligence
# =============================================================================

path = r"C:\\Users\\murilo.ribeiro\\OneDrive - EUROCHEM FERTILIZANTES TOCANTINS\\01 - VS Code Repository\\Fertilizer Price Forecasting\\"
file = "Historico CFR Brasil.xlsx"
df = pd.read_excel(path+file)
df = df.pivot(index = 'Date', columns = 'Product', values='Valor')

# df.columns.name = None
# df.index.name = None
# df = df.reset_index()
# df = df.rename(columns={'index':'Date'})

# =============================================================================
# DEFINIÇÃO DE FUNÇÕES DE APOIO
# =============================================================================

# Definição de métrica de erro = (S)ymmetric (M)ean (A)bsolute (P)ercentage (E)rror
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def feature_engineering(df, rawMaterial, momento = [2,10], rolling = 12, preditor=None):
    
    arr = df[rawMaterial]
    # df[f'{rawMaterial}_roc_{momento[0]}'] = arr.pct_change(2)*100
    #df[f'{rawMaterial}_roc_{momento[1]}'] = arr.pct_change(2)*100

    #df[f'{rawMaterial}_momentum_{momento[0]}'] = arr - arr.shift(2)
    #df[f'{rawMaterial}_momentum_{momento[1]}'] = arr - arr.shift(10)

    df[f'{rawMaterial}_diff_1'] = arr.diff(1)
    # df[f'{rawMaterial}_accel'] = df[f'{rawMaterial}_diff_1'].diff(1)
    
    
    media = arr.rolling(rolling).mean()
    desvio = arr.rolling(rolling).std()
    minimo = arr.rolling(rolling).min()
    maximo = arr.rolling(rolling).max()
    
    #df[f'{rawMaterial}_dist_media_{rolling}'] = ((arr - media)/media)*100
    df[f'{rawMaterial}_zscore_{rolling}'] = (arr - media)/desvio
    # df[f'{rawMaterial}_pos_range_{rolling}'] = (arr - minimo)/(maximo - minimo)
    df['is_above_trend'] = (df[rawMaterial] > df[rawMaterial].rolling(rolling).mean()*1.15).astype(int)
    
    if preditor is not None:
        arr_p = df[preditor]
        df[f'{preditor}_mov_avg'] = arr_p.rolling(rolling).mean()

def recalcular_features(historico_precos,valor_preditor, rawMaterial, momento=[2,10], rolling=12, preditor=None):

    if preditor is not None:
        df_temp = pd.DataFrame({rawMaterial: historico_precos, preditor:valor_preditor})
    else:
        df_temp = pd.DataFrame({rawMaterial: historico_precos})
    feature_engineering(df_temp, rawMaterial, momento, rolling, preditor=preditor)
    df_temp = df_temp.dropna()
    
    # Retorna a última linha como array (preço + features)
    return df_temp.iloc[-1].values

def apply_forecast_decay(forecast, hist_mean, decay_rate=0.15):
    """
    Aplica mean-reversion progressiva ao forecast.
    Quanto mais distante o horizonte, mais o valor é puxado para a média histórica.
    
    decay_rate: intensidade do decay (0.10 = suave, 0.25 = agressivo)
    Peso do modelo no step i = (1 - decay_rate)^i
    """
    decayed = np.copy(forecast)
    for i in range(len(forecast)):
        peso_modelo = (1 - decay_rate) ** i  
        decayed[i] = peso_modelo * forecast[i] + (1 - peso_modelo) * hist_mean
    return decayed

# =============================================================================
# MAIN FUNCTION: Treinamento e Previsão usando RNN
# =============================================================================

def forecast_cfr_prices(rawMaterial, dadosHistoricos = df, herdar = True, horizonte = 12, ordem_n = False, preditor = None):
    
    # Criar uma pasta para guardar as informações do matéria-prima
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, rawMaterial)
    os.makedirs(folder_path, exist_ok = True)
    
    # Definir um escalador para os valores 
    # scaler = RobustScaler()
    scaler = MinMaxScaler()
    dados_originais = dadosHistoricos[rawMaterial].values.copy()
    
    #  Ajuste de outliers para MOP e Urea
    if rawMaterial in ['MOP','Urea']:
        dados_transformados = winsorize(dados_originais, limits = [0, 0.01])
    else:
        dados_transformados = dados_originais
    
    if preditor is not None:
        df_temp = pd.DataFrame({rawMaterial:dados_transformados, preditor:dadosHistoricos[preditor]}, index = dadosHistoricos.index)
    else:
        df_temp = pd.DataFrame({rawMaterial:dados_transformados}, index = dadosHistoricos.index)
    
    # Aplicar a engenharia de features para auxiliar no treinamento do modelo
    # // O ajuste desses features pode ser feito diretamente na função feature_engineering
    if ordem_n:
        feature_engineering(df_temp, rawMaterial=rawMaterial, rolling = horizonte, preditor=preditor)
        df_temp = df_temp.dropna()
    else:
        df_temp = df_temp[[rawMaterial]].dropna()
    x_train_raw, x_test_raw, _, _ = train_test_split(
    df_temp.values, df_temp.values[:,0], test_size=0.2, random_state=123, shuffle=False)

    # Aplicar escala nos dados
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    features = scaler.transform(df_temp.values)
    target = features[:,0]
    y_train = x_train[:,0]
    y_test = x_test[:,0]
    
    # 1) Definição de Parâmetros do Modelo --------------------------------------------------------------------
    
    # Principais hiperparâmetros do modelo
    num_input = 4
    num_features = x_train.shape[1]
    batch_size = 48
    
    # Geração de séries temporais
    train_generator = TimeseriesGenerator(x_train,y_train, length = num_input,sampling_rate=1,batch_size = batch_size)
    test_generator = TimeseriesGenerator(x_test,y_test, length = num_input, sampling_rate=1, batch_size = batch_size)
    data_generator = TimeseriesGenerator(features,target, length = num_input, sampling_rate=1, batch_size = batch_size)
    
    # Rede Neural
    modelo = Sequential(name=f'LSTM_{rawMaterial}')
    modelo.add(GRU(8, activation = 'leaky_relu', input_shape = (num_input,num_features), return_sequences=True))
    modelo.add(Bidirectional(GRU(4, activation='tanh', return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))))
    modelo.add(tf.keras.layers.Dropout(0.2))
    modelo.add(Dense(1))
    
    # Compilar e treinar o modelo de redes neurais
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode = 'min', restore_best_weights=True, min_delta = 1e-5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience=10, mode = 'min', min_lr = 1e-3)
    modelo.compile(loss = tf.losses.MeanSquaredError(),
                # optimizer = tf.optimizers.Adam(),
                # adicionando uma taxa de aprendizado para prevenir overfitting
                optimizer = tf.optimizers.Adam(learning_rate=0.001),
                metrics = [tf.metrics.RootMeanSquaredError()])    

    # 2) Treinamento e Teste do Modelo ---------------------------------------------------------------------
    
    print(num_features)
    # Fit da base de treino
    history = modelo.fit(train_generator, epochs = 300,
                        validation_data = test_generator,
                        shuffle = False,
                        # Inativando o early_stopping para visualizar o treino completo
                        #callbacks = [early_stopping, reduce_lr],
                        #callbacks = [early_stopping],
                        verbose = 0)
    
    # Gráfico de VAL_LOSS & LOSS em função das épocas de treinamento

    loss_per_epoch = modelo.history.history['loss']
    loss_per_epoch = history.history['loss']
    val_loss_per_epoch = history.history['val_loss']
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_per_epoch)), loss_per_epoch, label='Train Loss')
    plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label='Val Loss')

    # Marcar o epoch onde o early stopping salvou os pesos
    best_epoch = np.argmin(val_loss_per_epoch)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.title(f'{rawMaterial} - Loss & Val_loss por Época')
    plt.legend()
    # Gravar o arquivo
    filepath = os.path.join(folder_path, f'{rawMaterial} - Loss & Val_loss por Época.png')
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')

    
    # Prever e comparar a base de testes
    predictions = modelo.predict(test_generator)
    if num_features == 1:
        # Univariado: inversão direta
        rev_trans = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(features[predictions.shape[0]*-1:])
        df_forecast = pd.DataFrame(columns=['Actual','Predicted'])
        df_forecast.loc[:,'Predicted'] = rev_trans[:,0]
        df_forecast.loc[:,'Actual'] = actual[:,0]
    else:
        # Multivariado: reconstruir o array completo para inverter escala
        df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][num_input:])], axis=1)
        rev_trans = scaler.inverse_transform(df_pred)
        actual = features[predictions.shape[0]*-1:]
        actual = scaler.inverse_transform(actual)
        df_forecast = pd.DataFrame(columns=['Actual','Predicted'])
        df_forecast.loc[:,'Predicted'] = rev_trans[:,0]
        df_forecast.loc[:,'Actual'] = actual[:,0]
    smape_ = smape(df_forecast['Actual'], df_forecast['Predicted'])
    
    ax = df_forecast[['Actual','Predicted']][-100:].plot(figsize=(8,4))
    plt.text(0.02, 0.98, f'sMAPE: {round(smape_,2)}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    plt.title(f'{rawMaterial} - Modelo LSTM (Ordem = {num_features}) | Antes de ajuste de Viés e Lag de Previsão')
    # Gravar o arquivo
    filepath = os.path.join(folder_path, f'{rawMaterial} - Modelo LSTM (Ordem = {num_features}) - Antes de ajuste de Viés e Lag de Previsão.png')
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')

    # Checar bias das medidas de previsão
    bias = np.mean(df_forecast['Predicted'] - df_forecast['Actual'])
    print(f"Bias: {bias:.2f}")  
    corrected_predictions = df_forecast['Predicted'] - bias  
    
    # Checar novo Bias
    new_bias = np.mean(corrected_predictions - df_forecast['Actual'])
    print(f"New Bias: {new_bias:.2f}")
    
    # Novo sMAPE
    new_smape = smape(df_forecast['Actual'], corrected_predictions)
    print(f"New SMAPE: {new_smape:.2f}%")
    df_forecast['Predicted'] = df_forecast['Predicted'] - bias
    
    # Correlação cruzada para determinar o shift nos períodos
    actual = df_forecast['Actual'].values
    predicted = df_forecast['Predicted'].values
    correlation = signal.correlate(actual - actual.mean(), 
                                predicted - predicted.mean(), 
                                mode='full')
    lags = signal.correlation_lags(len(actual), len(predicted))
    optimal_lag = lags[np.argmax(correlation)]
    print(f"Optimal lag = {optimal_lag} periods")

    # Forçar o tipo float (segurança)
    actual = df_forecast['Actual'].values.astype(float)
    predicted = df_forecast['Predicted'].values.astype(float)

    # Fazer um ajuste do lag para deslocar a curva de previsão
    lag = abs(optimal_lag) 
    df_forecast['Predicted_lag_corrected'] = df_forecast['Predicted'].shift(-lag)

    # Drop NaN 
    df_forecast_clean = df_forecast.dropna()

    # Comparar ambas as previsões: Antes & Depois (com shift)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5))

    # Antes da correção
    df_forecast_clean[['Actual', 'Predicted']][-60:].plot(ax=axes[0])
    axes[0].set_title(f'Antes da Correção (Lag = {optimal_lag} periods)')
    # Depois da correção
    df_forecast_clean[['Actual', 'Predicted_lag_corrected']][-60:].plot(ax=axes[1])
    axes[1].set_title(f'Após Correção (Shift = {lag} periods)')
    plt.tight_layout()
    # plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')

    # Plot Final: base final após treinamento e ajustes das bases de dados
    smape_ = smape(df_forecast_clean['Actual'], df_forecast_clean['Predicted_lag_corrected'])
    ax = df_forecast_clean[['Actual','Predicted_lag_corrected']][-100:].plot(figsize=(8,4))
    plt.text(0.02, 0.98, f'sMAPE: {round(smape_,2)}%', 
            transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    plt.title(f'{rawMaterial} - Modelo LSTM (Ordem = {num_features}) | Ajustado pelo Viés e Lag de Previsão')  
    
    # Gravar o arquivo
    filepath = os.path.join(folder_path, f'{rawMaterial} - Modelo LSTM (Ordem = {num_features}) - Ajustado pelo Viés e Lag de Previsão.png')
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')

    # 3) Forecasting dentro do horizonte estipulado --------------------------------------------------------

    lastdate = dadosHistoricos.index[-1]
    futuredates = pd.date_range(start = lastdate + pd.DateOffset(weeks=1), periods = horizonte, freq = 'W')
    
    # ---- DEPRECADO ----
    # DF_PROXY será usado para o plot final do gráfico
    df_proxy = pd.DataFrame(columns = [rawMaterial], index = futuredates[:])
    # proxy_scaled = scaler.transform(df_proxy)
    # df_scaled = scaler.transform(pd.DataFrame(dadosHistoricos[rawMaterial]))

    df_scaled = scaler.transform(df_temp.values)

    forecast = []
    current_batch = df_scaled[-num_input:]
    current_batch = current_batch.reshape(1, num_input, num_features)

    # Histórico de preços reais para recalcular features
    historico_precos = list(dadosHistoricos[rawMaterial].values)

    if preditor is not None:
        ultimo_valor_predito = dadosHistoricos[preditor].iloc[-horizonte].mean()
    else:
        ultimo_valor_predito = None

    print(num_features)
    if num_features == 1:
        buffer = list(df_scaled[-3:, 0])  # últimos 3 valores reais escalados

        for i in range(horizonte):
            # Previsão do próximo valor
            current_pred = modelo.predict(current_batch, verbose = 0)[0][0]
            
            # Suavizar: média móvel para eliminar zig-zag e reduzir drift
            buffer.append(current_pred)
            smoothed_pred = np.mean(buffer[-3:])
            
            # Inverter escala para obter o preço real
            preco_real = scaler.inverse_transform([[smoothed_pred]])[0, 0]
            forecast = np.append(forecast, preco_real)
            
            # Atualizar o dado atual, remover antigo e substituir pela nova predição
            current_batch = np.append(current_batch[:, 1:, :], [[[smoothed_pred]]], axis=1)

    else:
        for i in range(horizonte):   
            # Prever o preço
            current_pred_scaled = modelo.predict(current_batch, verbose=0)[0, 0]

            # Inverter escala para obter o preço real
            dummy = np.zeros((1, num_features))
            dummy[0, 0] = current_pred_scaled
            preco_real = scaler.inverse_transform(dummy)[0, 0]

            historico_precos.append(preco_real)
            features_recalculadas = recalcular_features(historico_precos, ultimo_valor_predito, rawMaterial, preditor=preditor)
            features_scaled = scaler.transform(features_recalculadas.reshape(1, -1))
            current_batch = np.append(
                current_batch[:, 1:, :],
                features_scaled.reshape(1, 1, num_features),
                axis=1
            )

            # Guardar o preço previsto
            forecast = np.append(forecast, preco_real)
    
    forecast_real = np.array(forecast)
    df_proxy.index.name = 'Date'

    # CORREÇÃO 1: Removendo o viés de forecast
    forecast_corrected_1 = forecast_real - bias
    df_proxy[f'{rawMaterial}_predicted'] = forecast_corrected_1
    
    # CORREÇÃO 2: Ajustando o shift da previsão
    forecast_corrected_2 = np.roll(forecast_real, -optimal_lag) - bias
    df_proxy[f'{rawMaterial}_predicted_lagged'] = forecast_corrected_2
    df_proxy[f'{rawMaterial}_predicted_lagged'].iloc[:-optimal_lag] = df_proxy[f'{rawMaterial}_predicted'].iloc[:-optimal_lag]

    # Combinar os dados
    df_combined = pd.concat([df[[rawMaterial]], df_proxy])
    df_combined['LowerBound'] = np.minimum(df_proxy[f'{rawMaterial}_predicted'],df_proxy[f'{rawMaterial}_predicted_lagged'])
    df_combined['UpperBound'] = np.maximum(df_proxy[f'{rawMaterial}_predicted'],df_proxy[f'{rawMaterial}_predicted_lagged'])
    df_combined['Prediction'] = (df_combined['LowerBound'] + df_combined['UpperBound'])/2
    df_combined['RawMaterial'] = rawMaterial
    df_combined = df_combined.rename(columns={rawMaterial:'HistoricalValues'})
    df_combined = df_combined[['RawMaterial','HistoricalValues','LowerBound','Prediction','UpperBound']]
    df_combined.index = pd.to_datetime(df_combined.index)

    # =====================================================================
    # CORREÇÃO 3: Decay — mean-reversion progressiva (após correções 1 e 2)
    # =====================================================================
    hist_mean_anchor = dadosHistoricos[rawMaterial].iloc[-52:].mean()
    
    df_proxy[f'{rawMaterial}_predicted'] = apply_forecast_decay(
        df_proxy[f'{rawMaterial}_predicted'].values, hist_mean_anchor, decay_rate=0.05
    )
    df_proxy[f'{rawMaterial}_predicted_lagged'] = apply_forecast_decay(
        df_proxy[f'{rawMaterial}_predicted_lagged'].values, hist_mean_anchor, decay_rate=0.05
    )

    # Combinar os dados
    df_combined = pd.concat([df[[rawMaterial]], df_proxy])
    df_combined['LowerBound'] = np.minimum(df_proxy[f'{rawMaterial}_predicted'], df_proxy[f'{rawMaterial}_predicted_lagged'])
    df_combined['UpperBound'] = np.maximum(df_proxy[f'{rawMaterial}_predicted'], df_proxy[f'{rawMaterial}_predicted_lagged'])
    df_combined['Prediction'] = (df_combined['LowerBound'] + df_combined['UpperBound']) / 2

    # =====================================================================
    # BANDA DE CONFIANÇA — incerteza crescente com o horizonte
    # =====================================================================
    hist_vol = dadosHistoricos[rawMaterial].pct_change().std()
    ultimo_preco = dadosHistoricos[rawMaterial].iloc[-1]
    
    steps = np.arange(1, horizonte + 1)
    banda_width = ultimo_preco * hist_vol * np.sqrt(steps) * 1.00

    prediction_values = df_combined['Prediction'].iloc[-horizonte:].values
    df_combined.loc[df_combined.index[-horizonte:], 'CI_Lower'] = prediction_values - banda_width
    df_combined.loc[df_combined.index[-horizonte:], 'CI_Upper'] = prediction_values + banda_width

    df_combined['RawMaterial'] = rawMaterial
    df_combined = df_combined.rename(columns={rawMaterial: 'HistoricalValues'})
    df_combined = df_combined[['RawMaterial', 'HistoricalValues', 'LowerBound', 'Prediction', 'UpperBound', 'CI_Lower', 'CI_Upper']]
    df_combined.index = pd.to_datetime(df_combined.index)

    # Tamanho dos gráficos
    plt.figure(figsize=(12, 5))

    lag = 160
    historical_data = df_combined['HistoricalValues'].iloc[-lag:].dropna()

    # Dados Históricos min, max & mean
    hist_min = historical_data.min()
    hist_max = historical_data.max()
    hist_mean = historical_data.mean()
    hist_std = historical_data.std()

    # Plotar: DADOS HISTÓRICOS
    plt.plot(historical_data.index, historical_data.values, 
            'b-', linewidth=1.5, label='Historical')

    # Plotar: DADOS PREVISTOS
    plt.plot(df_combined.iloc[-lag:].index, df_combined['Prediction'].iloc[-lag:], 
            'g--', linewidth=1.5, label='Prediction')

    # Plotar: BANDA DE CONFIANÇA (incerteza crescente)
    ci_data = df_combined[['CI_Lower', 'CI_Upper']].dropna()
    if not ci_data.empty:
        plt.fill_between(ci_data.index, ci_data['CI_Lower'], ci_data['CI_Upper'],
                         alpha=0.15, color='green', label='Banda de Confiança 95%')

    # Adicionar referências
    plt.axhline(y=hist_min, color='lightblue', linestyle='--', linewidth=1, 
                alpha=0.7, label=f'Historical Min: {hist_min:.1f}')
    plt.axhline(y=hist_max, color='lightblue', linestyle='--', linewidth=1, 
                alpha=0.7, label=f'Historical Max: {hist_max:.1f}')
    plt.axhline(y=hist_mean, color='navy', linestyle=':', linewidth=1.5, 
                alpha=0.7, label=f'Historical Mean: {hist_mean:.1f}')
    plt.axhspan(hist_mean - hist_std, hist_mean + hist_std, 
                alpha=0.1, color='blue', label=f'±1 Std Dev ({hist_std:.1f})')

    forecast_data = df_combined[df_combined['Prediction'] > 0].iloc[-lag:]

    # Anotações de dados previstos
    first_idx = forecast_data.index[0]
    first_val = forecast_data['Prediction'].iloc[0]
    plt.annotate(f'First: {first_val:.1f}', 
                (first_idx, first_val),
                textcoords="offset points", 
                xytext=(0, 15),
                ha='center', 
                fontsize=9,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor="#ABD49F", edgecolor='green', alpha=0.9))

    last_idx = forecast_data.index[-1]
    last_val = forecast_data['Prediction'].iloc[-1]
    plt.annotate(f'Last: {last_val:.1f}', 
                (last_idx, last_val),
                textcoords="offset points", 
                xytext=(0, -20),
                ha='center', 
                fontsize=9,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor="#ABD49F", edgecolor='green', alpha=0.9))


    plt.axvline(x=first_idx, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{rawMaterial}', fontsize=12)
    plt.title(f'{rawMaterial} - Dados Históricos vs. Previsão | Últimas {lag} semanas', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    # Gravar o arquivo
    filepath = os.path.join(folder_path, f'{rawMaterial} - Dados Históricos vs. Previsão - Últimas {lag} semanas.png')
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')
    
    # ------------------ DEPRECADO ----------------------------
    # Criar uma abstração de DISTRIBUIÇÃO DE PROBABILIDADES 
    # sigma = df_combined['Prediction'].iloc[-horizonte:].std()
    # mu = df_combined['Prediction'].iloc[-horizonte:].mean()
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 10000)
    # y = stats.norm.pdf(x, mu, sigma)
    # fig, ax = plt.subplots()
    # ax.plot(x,y)

    # percentiles = [10, 20, 50, 80, 90]
    # for percentile in percentiles:
    #     x_percentile = stats.norm.ppf(percentile/100, mu, sigma)
    #     y_percentile = stats.norm.pdf(x_percentile, mu, sigma)
    #     ax.axvline(x_percentile, color = 'r', linestyle = '--')
    #     ax.text(x_percentile, y_percentile, f'P{100-percentile}: {round(x_percentile,2)} kton', ha = 'center')
    # plt.title(f'{rawMaterial} - Distribuição Probabilidade para Valor Estimado')
    # filepath = os.path.join(folder_path, f'{rawMaterial} - Distribuição Probabilidade para Valor Estimado.png')
    # plt.savefig(filepath, dpi=300, bbox_inches = 'tight')    
    # ------------------ DEPRECADO ----------------------------
    
    # =====================================================================
    # DISTRIBUIÇÃO DE PROBABILIDADE — Monte Carlo baseado nos limites reais
    # =====================================================================
    
    pred_values = df_combined['Prediction'].iloc[-horizonte:].values
    ci_lower = df_combined['CI_Lower'].iloc[-horizonte:].values
    ci_upper = df_combined['CI_Upper'].iloc[-horizonte:].values
    
    # Monte Carlo: para cada step, gerar N cenários
    n_simulations = 5000
    np.random.seed(SEED)
    all_scenarios = []
    
    for i in range(horizonte):
        mu_step = pred_values[i]
        # Desvio proporcional à largura da banda naquele step
        # CI foi construído com 1.0 (~68%), então banda ≈ 1 sigma
        sigma_step = (ci_upper[i] - ci_lower[i]) / 2
        
        # Amostrar cenários para este step
        cenarios = np.random.normal(mu_step, sigma_step, n_simulations)
        all_scenarios.extend(cenarios)
    
    all_scenarios = np.array(all_scenarios)
    
    # Percentis reais da simulação
    percentis = {
        'P90 (Otimista)': np.percentile(all_scenarios, 90),
        'P50 (Base)': np.percentile(all_scenarios, 50),
        'P10 (Pessimista)': np.percentile(all_scenarios, 10)
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(all_scenarios, bw_method=0.2)
    x = np.linspace(all_scenarios.min(), all_scenarios.max(), 1000)
    y = kde(x)
    ax.fill_between(x, y, alpha=0.15, color='steelblue')
    ax.plot(x, y, color='steelblue', linewidth=2)
    
    # Linhas de percentis
    colors = {'P90 (Otimista)': 'green', 'P50 (Base)': 'navy', 'P10 (Pessimista)': 'red'}
    
    for label, value in percentis.items():
        color = colors[label]
        y_val = kde(value)[0]
        ax.axvline(value, color=color, linestyle='--', alpha=0.8, linewidth=1.2)

        # Ajustar posição para evitar sobreposição
        if 'Pessimista' in label:
            ha, offset_x, offset_y = 'right', -5, max(y)*0.08
        elif 'Otimista' in label:
            ha, offset_x, offset_y = 'left', 5, max(y)*0.08
        else:  # P50
            ha, offset_x, offset_y = 'center', 0, max(y)*0.12
        
        ax.annotate(f'{label}\n${value:.1f}', 
                    xy=(value, y_val),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    ha=ha, fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1))
    
    # Referência histórica
    ax.axvline(hist_mean, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(hist_mean, max(y)*0.95, f'Média Hist: ${hist_mean:.0f}', 
            ha='center', fontsize=8, color='gray')
    
    ax.set_xlabel(f'{rawMaterial} (USD/MT)', fontsize=11)
    ax.set_ylabel('Densidade', fontsize=11)
    ax.set_title(f'{rawMaterial} — Distribuição de Cenários (Monte Carlo) | Horizonte de {horizonte} semanas', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    filepath = os.path.join(folder_path, f'{rawMaterial} - Distribuição de Cenários.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Abrir todos os gráficos
    plt.show()
    

if __name__ == '__main__':
    for i in ['MOP']:
        
        print(f' ================= INICIANDO FORECASTING: RawMaterial = {i} ================= ')
        
        # PROXY --> MP predita : MP preditora
        proxy = {'MOP':'MAP'}
        
        forecast_cfr_prices(
            rawMaterial = i,
            dadosHistoricos = df,
            herdar = False,
            horizonte = 12,
            ordem_n = True,
            preditor = proxy.get(i)
            # preditor = None
        )
        
        print(f' Arquivos gravados na pasta -> {i}')
        print(f' ================= FORECASTING FINALIZADO: RawMaterial = {i} ================= ')