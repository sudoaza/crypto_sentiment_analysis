import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
from sklearn.preprocessing import MinMaxScaler
from talib import SMA

files = [\
        'data/Augmento Data/bitcointalk/bitcoin_24H_total.csv',
        'data/Augmento Data/bitcointalk/ethereum_24H_total.csv',
        'data/Augmento Data/bitcointalk/ripple_24H_total.csv'
    ]

datasets = []

for file in files:
    coin = file.split('/')[-1].split('_')[0]
    df = pd.read_csv(file, parse_dates=True, index_col='date', names=['date', f"volume_{coin}"])
    datasets.append(df)


data = pd.concat(datasets, axis=1).fillna(0)

# scaler = MinMaxScaler()
# norm_data = scaler.fit_transform(data)
# norm_data = pd.DataFrame(norm_data, columns=['volume_bitcoin_norm', 'volume_ethereum_norm', 'volume_ripple_norm'], index=data.index)
# data = pd.concat([data, norm_data], axis=1)

long_period = 20
short_period = 7

data['btc_sma'] = SMA(data['volume_bitcoin'], timeperiod=long_period)
data['eth_sma'] = SMA(data['volume_ethereum'], timeperiod=long_period)
data['xrp_sma'] = SMA(data['volume_ripple'], timeperiod=long_period)

data['btc_norm'] = (data['volume_bitcoin'] - data['btc_sma']) / np.maximum(data['btc_sma'], 1)
data['eth_norm'] = (data['volume_ethereum'] - data['eth_sma']) / np.maximum(data['eth_sma'], 1)
data['xrp_norm'] = (data['volume_ripple'] - data['xrp_sma']) / np.maximum(data['xrp_sma'], 1)

data['btc_norm_sma'] = SMA(data['btc_norm'], timeperiod=short_period)
data['eth_norm_sma'] = SMA(data['eth_norm'], timeperiod=short_period)
data['xrp_norm_sma'] = SMA(data['xrp_norm'], timeperiod=short_period)

print(data.head())
print(data.tail())

print(data.describe())

data[['btc_norm_sma', 'eth_norm_sma']].plot()

plt.show()
