# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv(
    'nifty-50-daily-shares-turnover.csv',
    sep=';',
    header=None,
    index_col=0,
    date_format='%d.%m.%Y'
)
df.index.name = 'date'

# %%
df

# %%
check_nan = df[[1]].isnull().values.any()
check_nan

# %%
df.interpolate(inplace=True)
df

# %%
check_nan = df[[1]].isnull().values.any()
check_nan

# %%
df.sort_index(ascending=True, inplace=True)
df

# %%
df.plot()

# %%
min_max_scaler = MinMaxScaler()
values_scaled = min_max_scaler.fit_transform(df.values)
df_scaled = pd.DataFrame(values_scaled)

# %%
df_scaled

# %%
df_scaled.plot()

# %%
def build_train_with_look_back(data, look_back = 10):
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    for i in range(0, len(data) - look_back):
        train = data.iloc[i : look_back + i]
        test = data.iloc[look_back + i]
        X = pd.concat([X, pd.DataFrame(train.values.T)])
        y = pd.concat([y, pd.Series(test.values[0])])
        
    for k in range(look_back):
        X = X.rename(columns = {k : 't' + str(k)})
        
    return X.reset_index(drop = True), y.values

# %%
#print(tscv)

# %%
def eval_model(model, model_name, look_back, epochs, split_no, losses):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=False)

    predictions = model.predict(X_train)
    predictions = min_max_scaler.inverse_transform(predictions)
    y_train_actual = min_max_scaler.inverse_transform(y_train.reshape(-1, 1))
    
    chart_params_label = 'LookBack=%d, Epochs=%d' % (look_back, epochs) 
    plt.plot(y_train_actual, color='blue', label='Actual Data (Train)')
    plt.plot(predictions, color='red', label='Predicted Data (Train)')
    plt.title(model_name + ' [' + chart_params_label + ']: Time Series Prediction (Train)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    predictions_test = model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test[:,0], predictions_test[:,0]))
    predictions_test = min_max_scaler.inverse_transform(predictions_test)
    y_test_actual = min_max_scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.plot(y_test_actual, color='green', label='Actual Data (Test)')
    plt.plot(predictions_test, color='yellow', label='Predicted Data (Test)')
    plt.title(model_name + ' [' + chart_params_label + ']: Time Series Prediction (Test)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'])
    plt.title(model_name + ' [' + chart_params_label + ']: Model loss (Test)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    losses.append({
        'model_name': model_name,
        'look_back': look_back,
        'epochs': epochs,
        'split_no': split_no,
        'loss': history.history['loss'][-1],
        'mse': test_score
    })

# %%
losses = []

for look_back in [5, 10, 15]:
    X, y = build_train_with_look_back(df_scaled, look_back)

    split_no = 1
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for epochs in [50, 100, 200, 300]: #, 700]:
            model = Sequential()
            model.add(SimpleRNN(units=100, activation='relu', input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            eval_model(model, 'SimpleRNN', look_back, epochs, split_no, losses)
            
            model = Sequential()
            model.add(LSTM(units=100, activation='relu', input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            eval_model(model, 'LSTM', look_back, epochs, split_no, losses)
            
            model = Sequential()
            model.add(GRU(units=100, activation='relu', input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            eval_model(model, 'GRU', look_back, epochs, split_no, losses)
            
        split_no += 1

# %%
df_losses = pd.DataFrame(losses)
df_losses = df_losses.set_index(['model_name', 'look_back', 'epochs'])
df_losses

# %%
pd.set_option('display.max_rows', None)
df_losses

# # %%
# look_back = 10
# X, y = build_train_with_look_back(df_scaled, look_back)
    
# #tscv = TimeSeriesSplit(n_splits=2)
# # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# # y_train, y_test = y[train_index], y[test_index]
# tscv = TimeSeriesSplit(n_splits=2)
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     model = Sequential()
#     model.add(SimpleRNN(units=50, activation='relu', input_shape=(look_back, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
#     print(history.history['loss'][-1])
    
#     predictions_test = model.predict(X_test)
#     predictions_test_denormalized = min_max_scaler.inverse_transform(predictions_test)
#     y_test_actual = min_max_scaler.inverse_transform(y_test.reshape(-1, 1))

#     test_score = np.sqrt(mean_squared_error(y_test[:,0], predictions_test[:,0]))
#     print(test_score)
#     # predictions = model.predict(X_train)
#     # predictions = min_max_scaler.inverse_transform(predictions)
#     # y_train_actual = min_max_scaler.inverse_transform(y_train.reshape(-1, 1))

#     # chart_params_label = ' LookBack=%d Epochs=%d, ' % (look_back, 50) 
#     # plt.plot(y_train_actual, color='blue', label='Actual Data (Train)' + chart_params_label)
#     # plt.plot(predictions, color='red', label='Predicted Data (Train)' + chart_params_label)
#     # plt.title('Time Series Prediction')
#     # plt.xlabel('Time')
#     # plt.ylabel('Value')
#     # plt.legend()
#     # plt.show()

# # %%

# %%
