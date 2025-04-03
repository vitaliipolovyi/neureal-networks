# %%-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# %%
# Клас моделі LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, input_size)  # Перетворення вхідних даних у формат (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# %%
# Завантаження даних
data = pd.read_csv('/CBSHP.csv')

# Конвертація стовпця 'Date' в datetime
data['Date'] = pd.to_datetime(data['Date'])

# Встановлення стовпця 'Date' як індекс
data.set_index('Date', inplace=True)

# Визначення ознак та цільової змінної
X = data.drop(columns=['Close'])
y = data['Close']

# Нормалізація даних
def min_max_scaling(data):
  return (data - data.min()) / (data.max() - data.min())

normalized_data = data.apply(min_max_scaling)

# %%
plt.figure(figsize=(14, 7))
plt.plot(data.index, y, label='Реальна ціна закриття', color='blue')  # Реальні ціни закриття на тестовому наборі даних (денормалізовані)
plt.title('Реальні ціни закриття')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.legend()
plt.show()

# %%
# Визначення тренувального (80%) та тестового (20%) наборів даних
train_size = int(len(data) * 0.8)
train_data, test_data = normalized_data.iloc[:train_size], normalized_data.iloc[train_size:]

# Визначення тренувальних ознак та цільової змінної
X_train, y_train = train_data.drop(columns=['Close']), train_data['Close']
X_test, y_test = test_data.drop(columns=['Close']), test_data['Close']

# %%
# Ініціалізація моделі LSTM
input_size = X_train.shape[1]
hidden_size = 10
num_layers = 1
output_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

# %%
# Втрата та оптимізатор
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

# %%
predictions = []
n_epochs = 40
for epoch in range(n_epochs):
  # walk-forward аналіз
  predictions = []

  for i in range(len(X_train), len(normalized_data)):
      train_inputs = torch.from_numpy(X_train.values[:i]).float().to(device)
      train_targets = torch.from_numpy(y_train.values[:i].reshape(-1, 1)).float().to(device)

      # Forward pass
      lstm_model.train()
      outputs = lstm_model(train_inputs)
      loss = criterion(outputs, train_targets)

      # Backward та оптимізація
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Прогнозування на тестових даних (20%)
      test_inputs = torch.from_numpy(X_test.values[:i - len(X_train) + 1]).float().to(device)
      test_outputs = lstm_model(test_inputs)
      predictions.append(test_outputs[-1].item())

  # Перетворення прогнозів з нормалізованого формату до початкового масштабу
  predicted_prices_denorm = np.array(predictions) * (data['Close'].max() - data['Close'].min()) + data['Close'].min()

  # Денормалізація тестового набору даних (y_test)
  y_test_denorm = y_test * (data['Close'].max() - data['Close'].min()) + data['Close'].min()

  # Розрахунок RMSE для LSTM моделі з денормалізованими даними
  lstm_rmse = np.sqrt(mean_squared_error(y_test_denorm, predicted_prices_denorm))

  if (epoch + 1) % 10 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch + 1, loss.item()))
  print("LSTM RMSE:", lstm_rmse)

# %%
# Побудова графіку прогнозованих та реальних значень цін закриття для LSTM з денормалізованими даними
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test_denorm, label='Реальна ціна закриття', color='blue')  # Реальні ціни закриття на тестовому наборі даних (денормалізовані)
plt.plot(test_data.index, predicted_prices_denorm, label='Прогноз LSTM', color='purple')  # Прогнозовані ціни закриття LSTM моделі (денормалізовані)
plt.title('Прогнозовані vs Реальні ціни закриття')
plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.legend()
plt.show()