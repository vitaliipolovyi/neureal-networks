# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import os

# %%
df = pd.read_csv('car_data.csv')

# %%
df['Gender'] = df['Gender'].astype('category')
df['Gender_Code'] = df['Gender'].cat.codes

# %%
df

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# %%
fig,ax=plt.subplots(2,3,figsize=(25,15))
sns.distplot(df['Age'],ax=ax[0,0])
sns.boxplot(y=df['Age'],ax=ax[0,1])
sns.histplot(data=df,x='Age',ax=ax[0,2],hue='Purchased',kde=True)

sns.distplot(df['AnnualSalary'],ax=ax[1,0])
sns.boxplot(y=df['AnnualSalary'],ax=ax[1,1])
sns.histplot(data=df,x='AnnualSalary',ax=ax[1,2],hue='Purchased',kde=True)
    
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing Continuous Columns",fontsize=30)

# %%
df.drop(['User ID'],axis=1,inplace=True)

# %%
sns.countplot(data=df, x='Gender')

# %%
plt.pie(df['Purchased'].value_counts(),labels=df['Purchased'].value_counts().index,autopct='%.2f',explode=[0,0.1])
plt.title("Class Imbalance")
plt.show()

# %%
sns.pairplot(df,hue='Purchased')
plt.show()

# %%
sns.heatmap(df[['Age', 'AnnualSalary', 'Gender_Code']].corr(), annot=True)

# %%
df_min_max_scaled = df[['Age', 'AnnualSalary', 'Purchased']]
for column in df_min_max_scaled.columns: 
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

# %%
x = df_min_max_scaled[['Age', 'AnnualSalary']] #df.drop(columns = ['Purchased'])
x

# %%
y = df_min_max_scaled[['Purchased']]
y.info()

# %%
x.info()

# %%
ROS = RandomOverSampler()
x, y = ROS.fit_resample(x, y)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 128
# %%
models = [
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dense(2, activation='softmax')
    ]),
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dropout(0.3),
        Dense(2, input_dim=2, activation='softmax')
    ]),
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ]),
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ]),
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dense(150, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ]),
    Sequential([
        Dense(300, input_dim=2, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(150, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ]),
]

for model in models:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=0)
        
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plot_model(model, show_shapes=True)
    plt.show()
    model.summary()

# %%
model = Sequential([
    Dense(200, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

log_dir = os.path.join("logs", "fit", "model")

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test), callbacks=[early_stopping, tensorboard])

plot_model(model, show_shapes=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()
plot_model(model, show_shapes=True)

# %%
