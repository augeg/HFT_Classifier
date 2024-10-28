import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

abs_path = "C:/Sauvegarde/Trading_house/CFM/Data_Challenge/Input"

df_X_tr = pd.read_csv(abs_path + "/X_train.csv")
df_Y_tr = pd.read_csv(abs_path + "/Y_train.csv") # 24 different categories
df_X_ts = pd.read_csv(abs_path + "/X_test.csv")

df_X_train = df_X_tr.copy()
df_Y_train = df_Y_tr.copy()

# Unbalance data  ? NO 

# # Checking and dealing with outliers ?! 
df_X_tr[['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']].describe()
# df_X_tr.loc[df_X_tr['price'] < -100]

newdf = df_X_tr.select_dtypes(include = ['number'])
newdf_total = df_X_tr.set_index('obs_id').join(df_Y_tr.set_index('obs_id'), how = 'inner')

newdf_total[['venue', 'eqt_code_cat']].groupby(['eqt_code_cat']).sum()


# newdf_total = newdf_total[['price', 'bid', 'ask', 'bid_size', 'ask_size', 'trade', 'flux', 'eqt_code_cat']]
# newdf_total.loc[newdf_total.bid].describe()

# newdf_total.loc[newdf_total.eqt_code_cat == 2].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 0].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 1].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 2].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 3].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 4].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 5].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 6].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 7].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 8].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 9].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 10].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 11].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 12].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 13].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 14].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 15].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 16].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 17].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 18].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 19].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 20].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 21].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 22].describe()
# newdf_total.loc[newdf_total.eqt_code_cat == 23].describe()

# newdf_total.loc[newdf_total.price > 250]['eqt_code_cat'].value_counts() # outlier cat 6 
# newdf_total.loc[newdf_total.price < -1]['eqt_code_cat'].value_counts() # outlier cat 15 et 19 

# newdf_total.loc[newdf_total.bid > 10]['eqt_code_cat'].value_counts() # outlier un peu 13
# newdf_total.loc[newdf_total.bid < -1]['eqt_code_cat'].value_counts() # No negative outlier

# newdf_total.loc[newdf_total.ask > 250]['eqt_code_cat'].value_counts() # outlier cat 6 
# newdf_total.loc[newdf_total.ask < -250]['eqt_code_cat'].value_counts() # outlier surtout 19

# newdf_total.loc[newdf_total.bid_size > 5000]['eqt_code_cat'].value_counts() # outlier 16
# newdf_total.loc[newdf_total.bid_size < 1]['eqt_code_cat'].value_counts() # No negative outlier

# newdf_total.loc[newdf_total.ask_size > 5000]['eqt_code_cat'].value_counts() # outlier 16
# newdf_total.loc[newdf_total.ask_size < 1]['eqt_code_cat'].value_counts() # No negative outlier

# newdf_total.loc[newdf_total.flux > 1000]['eqt_code_cat'].value_counts() # outlier 16 et 8
# newdf_total.loc[newdf_total.flux < -1000]['eqt_code_cat'].value_counts() # outlier 16 et 8

# # IL FAUT VIRER LES OUTLIERS !!!!

# # Checking nan ... 
# df_X_train.isnull().values.sum() # No missing values 

# # Checking data - first glance
# set(df_X_train['venue']) # Market place {0, 1, 2, 3, 4, 5}
# set(df_X_train['action']) # Type {'A', 'D', 'U'}
# set(df_X_train['trade']) # Traded {False, True}
# set(df_X_train['side']) #{'A', 'B'}}

# df_X_train['flux'].describe()

# DataFrame setup by obs_id ==> need to expand dim from X_train to 16080000 ==> 160800 x 100

# Data Engineering
# col_winsorize = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']


# from scipy.stats.mstats import winsorize
# # On winsorize le merdier ==> on enlève les 5% min et les 5% max
# for col in col_winsorize : 
#     df_X_train[col] = winsorize(df_X_train[col], limits = 0.01)

def feature_engineer(df_X):

    df_X['spread'] = df_X['bid'] - df_X['ask']
    df_X['OBI'] = (df_X['bid_size'] - df_X['ask_size']) / (df_X['bid_size'] + df_X['ask_size'])
    df_X['spread_size'] = df_X['bid_size'] - df_X['ask_size']
   
    return df_X



# First we do engineering
df_X_train = feature_engineer(df_X_train)







col = ['venue', 'action', 'side', 'price', 'bid', 'ask', 'bid_size', 'ask_size', 'trade', 'flux', 'spread', 'OBI', 'spread_size'] 
col_std = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux', 'spread', 'OBI','spread_size']

# Normalisation of training data

scaled_data = df_X_train.copy()
data = scaled_data[col_std]
scaler = RobustScaler().fit(data.values)
data = scaler.transform(data.values)
scaled_data[col_std] = data
scaled_data.shape
df_X_train = scaled_data


def pre_processing_X(df_X, col, col_std):

    df_X = df_X[col]
    
    #Here we extract the columns with object datatype as they are the categorical columns
    categorical_columns = df_X.select_dtypes(include=['object']).columns.tolist()


    #Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df_X[categorical_columns])

    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded_train_X = pd.concat([df_X, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded_train_X = df_encoded_train_X.drop(categorical_columns, axis=1)

    # Encoding "venue = place de quotation"
    df_encoded_train_X_2D = pd.get_dummies(df_encoded_train_X, prefix = "venue" , columns = ['venue'], dtype=float)

    # Reshaping X_DataFrame 
    df_train_X_reshaped = df_encoded_train_X_2D.values.reshape(-1, 100, df_encoded_train_X_2D.shape[1])

    return df_train_X_reshaped

# ['obs_id', 'venue', 'order_id', 'action', 'side', 'price', 'bid', 'ask', 'bid_size', 'ask_size', 'trade', 'flux'] 

df_X_train_reshaped = pre_processing_X(df_X_train, col, col_std)
df_X_train_reshaped.shape # format attendu (160800, 100, xxx)

# On fait pareil avec train_Y - mais c'est plus simple
def pre_processing_Y(df_Y):
    return pd.get_dummies(df_Y, prefix = "eqt_code_cat" , columns = ['eqt_code_cat'], dtype=float)


df_train_Y_2D = pre_processing_Y(df_Y_train)


# On prépare le dataframe de train, de val et de test
X_train, X_test, y_train, y_test = train_test_split(df_X_train_reshaped, df_train_Y_2D, test_size=0.33, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# train & Test tensor transformation
tensor_X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
tensor_X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
tensor_X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)

# Pour plus tard LSTM model doit être sous la forme (batch_size, timesteps, features)
# Neural Network model

model_v0 = tf.keras.Sequential([
    tf.keras.layers.Input((tensor_X_train.shape[1], tensor_X_train.shape[2])),
    tf.keras.layers.LSTM(units = 64, return_sequences=True),
    tf.keras.layers.LSTM(units = 128, return_sequences=False),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation = 'softmax'),
    ])

model_v0.summary()

#################

forward_layer = tf.keras.layers.LSTM(units = 64, return_sequences=True)
backward_layer = tf.keras.layers.LSTM(units = 64, return_sequences=True, activation = 'relu', go_backwards = True)

model = tf.keras.Sequential([
    tf.keras.layers.Input((tensor_X_train.shape[1], tensor_X_train.shape[2])),
    tf.keras.layers.Bidirectional(forward_layer, backward_layer = backward_layer),
    tf.keras.layers.GRU(units = 32, return_sequences=False),
    #ê tf.keras.layers.Flatten(), # <== Pour passer de 3 dimensions à 2
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation = 'softmax'),
    ])

model.summary()
    
    
    

# Compilation and test 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )

# fitting phase
r = model.fit(tensor_X_train, y_train.iloc[:,1:], epochs=20, batch_size = 300, validation_data=(tensor_X_val, y_val.iloc[:,1:]),)

# On test le modèle
test_loss, test_acc = model.evaluate(tensor_X_test,  y_test.iloc[:,1:], verbose=2)
print('\nTest accuracy:', test_acc)

# pred = model.predict(tensor_X_test)
# df_pred = pd.DataFrame(pred)
# df_pred = df_pred.idxmax(axis=1)
# df_pred = df_pred.reset_index()
# df_pred.columns = ['obs_id','eqt_code_cat']

# y = tf.argmax(y_test.iloc[:,1:].values, axis = 1)
# y_ = tf.argmax(pred, axis = 1)
# tf.math.confusion_matrix(labels = y, predictions = y_)

# from sklearn.metrics import confusion_matrix
# import numpy as np
import matplotlib.pyplot as plt
# matrix = confusion_matrix(y, y_)

# df_matrix = pd.DataFrame(matrix)
# df_matrix.style.background_gradient(cmap='Blues')
# print(df_matrix)


# plt.pcolor(df_matrix)
# plt.yticks(np.arange(0.5, len(df_matrix.index), 1), df_matrix.index)
# plt.xticks(np.arange(0.5, len(df_matrix.columns), 1), df_matrix.columns)
# plt.show()

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


#### On fait le test :
# First we do engineering
df_X_test = df_X_ts.copy()
df_X_test = feature_engineer(df_X_test)

# Normalization of test data
scaled_data = df_X_test.copy()
data = scaled_data[col_std]
data = scaler.transform(data.values)
scaled_data[col_std] = data 
df_X_test = scaled_data    

df_X_test_reshaped = pre_processing_X(df_X_test, col, col_std)  
df_X_test_reshaped.shape
tensor_X_test_reshaped = tf.convert_to_tensor(df_X_test_reshaped, dtype=tf.float32)
tensor_X_test_reshaped.shape

res = model.predict(tensor_X_test_reshaped)
df_res = pd.DataFrame(res)
df_res.shape

new_res = df_res.idxmax(axis=1)
new_res = new_res.reset_index()

new_res.columns = ['obs_id','eqt_code_cat']
new_res = new_res.astype('str')

new_res.to_csv(abs_path + "/Y_test_final.csv", index = False)



