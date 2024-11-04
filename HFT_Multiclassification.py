# Importing modules

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


abs_path = "C:/Sauvegarde/Trading_house/CFM/Data_Challenge/Input"

# Data Fields
df_X_tr = pd.read_csv(abs_path + "/X_train.csv")
df_Y_tr = pd.read_csv(abs_path + "/Y_train.csv") # 24 different categories


# Features engineering
col_std = ['price', 'bid', 'ask', 'flux']

def feature_engineer(df_X, col_std):

    df_X['PxV'] = df_X.price * df_X.flux
    df_X['VWAP'] = df_X['PxV'].expanding(1).sum().fillna(0) / (1 + np.abs(df_X.flux.expanding(1).sum()))
    df_X['Num_tick'] = df_X['bid_size'] + df_X['ask_size']
    df_X['MIDt'] = (df_X.ask + df_X.bid) / 2
    df_X['Eff_spread'] = 2 * np.abs(df_X.price - df_X.MIDt)
    #df_X = pd.get_dummies(df_X, prefix = ['action','side','venue'] , columns = ['action','side','venue'], dtype=float)
    df_X = df_X.drop(['action','side','venue', 'obs_id'],axis = 1)
    print(df_X.isnull().sum())
    
    return df_X.values.reshape(-1, 100, df_X.shape[1])

df_X = feature_engineer(df_X_tr.copy(), col_std)


def pre_processing_Y(df_Y):
    df_Y = df_Y.drop('obs_id', axis = 1)
    return pd.get_dummies(df_Y, prefix = "eqt_code_cat" , columns = ['eqt_code_cat'], dtype=float)

df_Y = pre_processing_Y(df_Y_tr)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.33, random_state=42)

# train & Test tensor transformation
tensor_X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
tensor_X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
tensor_X_train.shape

# Neural Network model

# 1 Bidirectional GRU model

forward_layer = tf.keras.layers.GRU(units = 24, return_sequences=True)
backward_layer = tf.keras.layers.GRU(units = 24, return_sequences=True, activation = 'relu', go_backwards = True)

model = tf.keras.Sequential([
    # Feature extractor
    tf.keras.layers.Input((tensor_X_train.shape[1], tensor_X_train.shape[2])),
    tf.keras.layers.Bidirectional(forward_layer, backward_layer = backward_layer),   

    tf.keras.layers.GRU(units = 48, return_sequences=False, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    
    # Classifier
    tf.keras.layers.Dense(48, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001)),
    tf.keras.layers.Dense(48, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(24, activation = 'softmax'),
    ])

model.summary()

# 2) Convolutional GRU model

model = tf.keras.Sequential([
    tf.keras.layers.Input((tensor_X_train.shape[1], tensor_X_train.shape[2])),
    
    tf.keras.layers.Conv1D(filters = 200, kernel_size = 16, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(filters = 200, kernel_size = 16, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(keepdims = True),
    tf.keras.layers.Conv1D(filters = 200, kernel_size = 16, padding="same", activation="relu"),
    tf.keras.layers.Conv1D(filters = 200, kernel_size = 16, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(keepdims = True),
    
    tf.keras.layers.GRU(units = 200, return_sequences=True, activation = 'relu'),
    tf.keras.layers.GRU(units = 200, return_sequences=True, activation = 'relu'),
    tf.keras.layers.GRU(units = 200, return_sequences=False, activation = 'relu'),
    
    
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation = 'softmax'),
    ])
model.summary() 
  
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Compilation and test 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )

# fitting phase
r = model.fit(tensor_X_train, y_train, epochs=1000, batch_size = 1000, validation_split=0.40, callbacks=[early_stopping])


# On test le modèle
test_loss, test_acc = model.evaluate(tensor_X_test,  y_test, verbose=1)
print('\nTest accuracy:', test_acc)

y_pred = model.predict(tensor_X_test)
df_pred = pd.DataFrame(y_pred)

df_pred.idxmax(axis=1)
y_test.columns = [i for i in range(0,24)]
y_test.idxmax(axis=1)

# Definition of my own accuracy function

def accuracy(y_true, y_pred):
    
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score
    
    """
    
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    
    #returns accuracy
    return correct_predictions / len(y_true)


accuracy(y_test.idxmax(axis=1), df_pred.idxmax(axis=1))




