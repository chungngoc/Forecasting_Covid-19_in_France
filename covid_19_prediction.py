import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime as dt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.preprocessing.sequence import TimeseriesGenerator


path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
data_files = ["time_series_covid19_confirmed_global.csv", "time_series_covid19_deaths_global.csv", "time_series_covid19_recovered_global.csv"]

url_confirmed = os.path.join(path_data, data_files[0])
url_deaths = os.path.join(path_data, data_files[1])
url_recovered = os.path.join(path_data, data_files[2])

df_confirmed = pd.read_csv(url_confirmed)
df_deaths = pd.read_csv(url_deaths)
df_recovered = pd.read_csv(url_recovered)

df_confirmed.describe(include = 'all')


def preprocessing_data(df):
  # La latitude et la longitude ne contribuent pas à notre analyse
  # nous allons donc supprimer ces 2 colonnes.
  df.drop(['Lat', 'Long'], axis=1, inplace = True)

  # nous devrons donc fusionner les données 
  # pour obtenir le infection totale de chaque pays par jour.
  df = df.groupby('Country/Region').sum()

  df = df.transpose().reset_index().rename(columns={'index':'Date'})
  df.rename_axis(None, axis=1,inplace=True)
  df['Date'] = pd.to_datetime(df['Date'])
  return df

df_confirmed = preprocessing_data(df_confirmed)
df_deaths = preprocessing_data(df_deaths)
df_recovered = preprocessing_data(df_recovered)


liste_Pays = ['India', 'Italy', 'France', 'US', 'Spain', 'United Kingdom']
for i in liste_Pays:
  df_confirmed[i].plot()

plt.legend()
plt.title('Infection Cases of COVID-19')
plt.show()

# Choose 1 country for prediction
def choose_country(name_country):
  data_C = df_confirmed[['Date',name_country]]
  data_C.rename(columns={name_country:"Confirmed"}, inplace =True)

  data_D = df_deaths[['Date',name_country]]
  data_D.rename(columns={name_country:"Deaths"}, inplace =True)

  data_R = df_recovered[['Date',name_country]]
  data_R.rename(columns={name_country:"Recovered"}, inplace =True)

  return data_C, data_D, data_R

# C - confirmed; D - death; R - recovered
data_C, data_D, data_R = choose_country("France")  

data_confirmed = data_C['Confirmed'].to_numpy()
data_deaths = data_D['Deaths'].to_numpy()
data_recovered = data_R['Recovered'].to_numpy()

numbers_of_dates_confirmed = data_C.index.values.reshape(-1,1)
numbers_of_dates_deaths = data_D.index.values.reshape(-1,1)
numbers_of_dates_recovered = data_R.index.values.reshape(-1,1)

# Prediction Covid_confirmed cases for the next n days
nb_future_pre = 15  # Prediction for next 15 days
days_start_to_futures = np.array([i for i in range(data_C.shape[0]+nb_future_pre)]).reshape(-1, 1)

# Spit data
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(numbers_of_dates_confirmed,data_confirmed, test_size=0.15, shuffle=True)
X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(numbers_of_dates_deaths,data_deaths, test_size=0.15, shuffle=True)
X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(numbers_of_dates_recovered,data_recovered, test_size=0.3, shuffle=False)

print(X_train_confirmed.shape)
print(y_train_confirmed.shape)
print(X_test_confirmed.shape)
print(y_test_confirmed.shape)

# Linear regression

def linear_regression(X_train, X_test, y_train, y_test):
  # poly features
  pf =PolynomialFeatures(degree=6,include_bias=True)

  # normalize splitted data
  poly_X_train = pf.fit_transform(X_train)
  poly_X_test = pf.fit_transform(X_test)
  poly_future_pre = pf.fit_transform(days_start_to_futures)
  #print(poly_X_train)

  # fit a Linear Regression model
  lin_regr=LinearRegression(normalize=True, fit_intercept=True)
  lin_regr.fit(poly_X_train, y_train)

  coef=lin_regr.coef_
  print("coef: ", coef)

  # Prediction
  y_pred = lin_regr.predict(poly_X_test)
  poly_pred = lin_regr.predict(poly_future_pre)

  # all values of predictions must > 0
  for i in range(len(y_pred)):
    if y_pred[i] < 0:
      y_pred[i] = 0
      
  # Plot prediction on test set
  plt.plot(y_test)
  plt.plot(y_pred)
  plt.legend(['Test Data', 'Predictions'])
  plt.show()

  # Calcul error
  mae=mean_absolute_error(y_pred, y_test)
  mse=mean_squared_error(y_pred, y_test)
  rmse = np.sqrt(mse)
  print("RMSE of ", lin_regr.__class__.__name__, round(rmse,1))
  print("MAE of ", lin_regr.__class__.__name__, round(mae,1), '\n')
  
  # all values of predictions must > 0
  for i in range(len(poly_pred)):
    if poly_pred[i] < 0:
      poly_pred[i] = 0

  return poly_pred

# function to plot prediction
def plot_prediction(name_model, prediction_data, type_data):
  if type_data == "Confirmed":
    data = data_C
  elif type_data == "Deaths":
    data = data_D
  elif type_data == "Recovered":
    data = data_R

  dates_pre = pd.date_range(start = data['Date'][0], periods=data.shape[0] + nb_future_pre)
  plt.figure(figsize= (12,8))
  plt.xlabel("Dates")
  plt.ylabel("France " +type_data +" cases")
  title = "Predicted values of " +type_data + " cases with " + name_model
  plt.title(title)

  plt.plot(data['Date'], data[type_data], linestyle ='--', color= 'blue')
  plt.plot(dates_pre, prediction_data, linestyle ='--', color = 'red')
  plt.legend(['Real_data', 'Predictions'])
  plt.show()

# Confirmed cases data
poly_pred_confirmed = linear_regression(X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed)

plot_prediction("linear regression", poly_pred_confirmed, "Confirmed")

# Deaths cases data

poly_pred_deaths = linear_regression(X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths)

plot_prediction("linear regression", poly_pred_deaths, "Deaths")

# Recovered cases data

poly_pred_recovered = linear_regression(X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered)

plot_prediction("linear regression", poly_pred_recovered, "Recovered")

# Support vector machine regression(SVR)
def svm_model(X_train, y_train, X_test, y_test):
  # SVM Model
  svm_reg = SVR(kernel='poly', C=0.5, gamma=0.01, epsilon=0.01)
  svm_reg.fit(X_train, y_train)

  y_pred = svm_reg.predict(X_test)
  svm_pred = svm_reg.predict(days_start_to_futures)
  # Plot
  plt.plot(y_test)
  plt.plot(y_pred)
  plt.legend(['Test Data', 'SVR Predictions'])

  mae=mean_absolute_error(y_pred, y_test)
  mse=mean_squared_error(y_pred, y_test)
  rmse = np.sqrt(mse)
  print("RMSE of ", svm_reg.__class__.__name__, round(rmse,1))
  print("MAE of ", svm_reg.__class__.__name__, round(mae,1), '\n')

  return svm_pred

# Confirmed cases data
svm_pred_confirmed = svm_model(X_train_confirmed, y_train_confirmed, X_test_confirmed, y_test_confirmed)
plot_prediction("SVR", svm_pred_confirmed, "Confirmed")

# Deaths cases data
svm_pred_deaths = svm_model(X_train_deaths, y_train_deaths, X_test_deaths, y_test_deaths)
plot_prediction("SVR", svm_pred_deaths, "Deaths")

# Recovered cases data
svm_pred_recovered = svm_model(X_train_recovered, y_train_recovered, X_test_recovered, y_test_recovered)
plot_prediction("SVR", svm_pred_recovered, "Recovered")

# Long Short-Term Memory network (LSTM) - univariate
#Choose timesteps, features and number of days in future, ...
nb_steps = 10
nb_features = 1
nb_future_pre = 30
nb_node = 150
percent_split = 2/3

# Split data
normalizing = False
def split_data(data, nb_steps):
  if normalizing:
    # Normalizing data
    normal = MinMaxScaler()
    normal = normal.fit(data.reshape(-1,1))
    normal = normal.transform(data.reshape(-1,1))
  else : normal = data

  data_X, data_y = list(), list()
  #The batches would be [[t_1,...,t_{nb_steps}],predict=tt_{n_steps+1}].
  for i in range(len(normal)):  
    if i + nb_steps >= len(normal):
      break
    seq_x, seq_y = normal[i: i+nb_steps], normal[i+nb_steps]
    data_X.append(seq_x)
    data_y.append(seq_y)
  
  return np.asarray(data_X), np.asarray(data_y)

# Predict n days in future
def predict_future(model, data, nb_future_pre, nb_steps):
  y_preds = model.predict(data)
  for i in range(nb_future_pre):
    X = y_preds[-nb_steps:].reshape(1,nb_steps, nb_features)
    y = model.predict(X)
    #data = np.concatenate((data, y))
    y_preds = np.concatenate((y_preds, y))
  
  return y_preds

# Buil model
def buil_model(nb_node):
  model = Sequential()
  model.add(LSTM(nb_node, activation='relu', input_shape=(nb_steps, nb_features) ))
  model.add(Dense(75, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer="adam",loss="mse")
  model.summary()
  return model

# Create train and test set
train_size = int(percent_split*len(data_confirmed))

train_set_confirmed, test_set_confirmed = data_confirmed[:train_size], data_confirmed[train_size:]
train_set_deaths, test_set_deaths = data_deaths[:train_size], data_deaths[train_size:]
train_set_recovered, test_set_recovered = data_recovered[:train_size], data_recovered[train_size:]

X_train_confirmed, y_train_confirmed = split_data(train_set_confirmed, nb_steps)
X_test_confirmed, y_test_confirmed = split_data(test_set_confirmed, nb_steps)

X_train_deaths, y_train_deaths = split_data(train_set_deaths, nb_steps)
X_test_deaths, y_test_deaths = split_data(test_set_deaths, nb_steps)

X_train_recovered, y_train_recovered = split_data(train_set_recovered, nb_steps)
X_test_recovered, y_test_recovered = split_data(test_set_recovered, nb_steps)

def plot_loss_curve(history):
  # Plot Loss curve
  history_dict = history.history
  print(history_dict.keys())
  loss = history_dict['loss']
  #val_loss = history_dict['val_loss']
  epochs = range(1, len(loss)  + 1)

  plt.plot(epochs, loss, 'r', label='Training loss')
  #plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title("Training loss")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def plot_pre_test_set(model, X_test, y_test):
  # plot prediction on Test set
  y_test_pred_C = model.predict(X_test)
  #print(y_test_pred_C)
  plt.plot(y_test)
  plt.plot(y_test_pred_C)
  plt.legend(['Test Data', 'LSTM prediction'])
  plt.title('Prediction on Test Data')
  plt.show()

def lstm_predict(model, data, type_data):
  data_X, data_y = split_data(data, nb_steps)
  lstm_pred = predict_future(model, data_X, nb_future_pre, nb_steps)

  if normalizing:
    normal = MinMaxScaler()
    normal = normal.fit(data.reshape(-1,1))
    lstm_pred = normal.inverse_transform(lstm_pred)

  # Plot prediction
  plt.xlabel("Dates")
  plt.ylabel("France " +type_data + " cases")
  plt.title("Predicted values of " + type_data + " cases with LSTM-univariate")
  plt.plot(data[nb_steps:], label='origin', linestyle ='--', color = 'red')
  plt.plot(lstm_pred.reshape(-1,1), label='predict', linestyle ='-', color= 'blue')
  plt.legend(['Real_data', 'Predictions'])
  plt.show()

# Model and prediction for Confirmed cases
model_confirmed = buil_model(nb_node)

# fit model
history_confirmed = model_confirmed.fit(X_train_confirmed, y_train_confirmed, epochs=200,verbose=0)

plot_loss_curve(history_confirmed)
plot_pre_test_set(model_confirmed, X_test_confirmed, y_test_confirmed)

lstm_predict(model_confirmed, data_confirmed, "confirmed")

# Model and prediction for Deaths cases
model_deaths = buil_model(nb_node)

# fit model
history_deaths = model_deaths.fit(X_train_deaths, y_train_deaths, epochs=200,verbose=0)

plot_loss_curve(history_deaths)
plot_pre_test_set(model_deaths, X_test_deaths, y_test_deaths)

lstm_predict(model_deaths, data_deaths, "deaths")

# Model and prediction for Recovered cases
model_recovered = buil_model(nb_node)

# fit model
history_recovered = model_recovered.fit(X_train_recovered, y_train_recovered, epochs=200,verbose=0)

plot_loss_curve(history_recovered)
plot_pre_test_set(model_recovered, X_test_recovered, y_test_recovered)

lstm_predict(model_recovered, data_recovered, "recovered")


# LSTM-Multivariate
df_merge = pd.merge(data_C, data_D, how='left', on=['Date'])
df_merge = pd.merge(df_merge, data_R, how='left', on=['Date'])
df_merge['Age'] = 20.8
df_merge.loc[df_merge['Date'].dt.year.isin([2021,2022]), 'Age'] = 21.1
df_merge.set_index("Date", inplace =True)

# Change columns's position to predict Deaths cases
# The first column will be "Deaths"
df_merge = df_merge[['Deaths', 'Confirmed', 'Recovered', 'Age']] 

# Visualization
for i in ['Deaths', 'Recovered']:
  df_merge[i].plot()

plt.legend()
plt.title(' Deaths and Recovered Cases of COVID-19 of US')
plt.show()

# Correlations
sns.heatmap(df_merge.corr(), annot = True)

df_merge.drop(['Recovered'], axis=1, inplace = True)
values = df_merge.values

# normalise data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

#convert data to supervised form
def to_supervised(data,time_step = 2):
    df = pd.DataFrame(data)
    column = []
    column.append(df)
    for i in range(1,time_step+1):
        column.append(df.shift(-i))
    df = pd.concat(column,axis=1)
    df.dropna(inplace = True)
    nb_features = data.shape[1]
    data = df.values
    supervised_data = data[:,:nb_features*time_step]
    supervised_data = np.column_stack( [supervised_data, data[:,nb_features*time_step]])
    return supervised_data

time_step = 2
supervised = to_supervised(scaled,time_step)
#print(supervised)
pd.DataFrame(supervised).head()

# Split data
nb_features = df_merge.shape[1]
percent_split = 0.8
train_size = int(df_merge.shape[0]*percent_split)

X = supervised[:,:nb_features*time_step]
y = supervised[:,nb_features*time_step]
#print(y[0])
X_train = X[:train_size,:]
X_test = X[train_size:,:]
y_train = y[:train_size]
y_test = y[train_size:]

print (X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#convert data to fit for lstm
X_train = X_train.reshape(X_train.shape[0], time_step, nb_features)
X_test = X_test.reshape(X_test.shape[0], time_step, nb_features)
print(X_train)
print (X_train.shape,X_test.shape)

# Buil model
def buil_model(nb_node):
  model = Sequential()
  model.add(LSTM(nb_node, activation='relu', input_shape = ( time_step,nb_features) ))
  model.add(Dense(75, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer="adam",loss="mse")
  model.summary()
  return model

model = buil_model(nb_node = 150)
# fit model
history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=200,batch_size = 72,verbose=0)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()
plt.title("loss during training")
plt.show()

y_pred = model.predict(X_test)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[2]*X_test.shape[1])

#scale back the prediction to orginal scale
temp = np.concatenate( (y_pred, X_test[:,1:nb_features] ) , axis =1)
temp = scaler.inverse_transform(temp)
test_pred = temp[:,0]

y_test = y_test.reshape( len(y_test), 1)

temp = np.concatenate( (y_test, X_test[:,1:nb_features] ) ,axis = 1)
temp = scaler.inverse_transform(temp)
real_data = temp[:,0]

#plot the prediction
plt.plot(test_pred, label = "prediction",c = "b")
plt.plot(real_data,label = "real data",c="r")
plt.title("comparison between prediction and real data")
plt.legend()
plt.show()

# Error
print("Mean absolute error : {}".format(mean_absolute_error(test_pred,real_data)) )
print("Mean squared error : {}".format(mean_squared_error(test_pred,real_data)) )


date_format = "%Y-%m-%d"
lockdown_1_start = dt.strptime('2020-03-17', date_format)
lockdown_1_end = dt.strptime('2020-05-11', date_format)

lockdown_2_start = dt.strptime('2020-10-30', date_format)
lockdown_2_end = dt.strptime('2020-12-15', date_format)

lockdown_3_start = dt.strptime('2021-04-04', date_format)
lockdown_3_end = dt.strptime('2021-05-03', date_format)

lockdown_1 = pd.date_range(start = lockdown_1_start, end = lockdown_1_end)
lockdown_2 = pd.date_range(start = lockdown_2_start, end = lockdown_2_end)
lockdown_3 = pd.date_range(start = lockdown_3_start, end = lockdown_3_end)

lockdown = lockdown_1.union(lockdown_2).union(lockdown_3)

df_merge['Lockdown'] = 
df_merge.loc[df_merge.index.isin(lockdown), 'Lockdown'] = 2
sns.heatmap(df_merge.corr(), annot = True)

values = df_merge.values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

time_step = 2
supervised = to_supervised(scaled,time_step)
#print(supervised)
pd.DataFrame(supervised).head()

# Split data
nb_features = df_merge.shape[1]
percent_split = 0.8
train_size = int(df_merge.shape[0]*percent_split)

X = supervised[:,:nb_features*time_step]
y = supervised[:,nb_features*time_step]
#print(y[0])
X_train = X[:train_size,:]
X_test = X[train_size:,:]
y_train = y[:train_size]
y_test = y[train_size:]

print (X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#convert data to fit for lstm
X_train = X_train.reshape(X_train.shape[0], time_step, nb_features)
X_test = X_test.reshape(X_test.shape[0], time_step, nb_features)

print (X_train.shape,X_test.shape)

model = buil_model(nb_node = 150)
# fit model
history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=200,batch_size = 72,verbose=0)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()
plt.title("loss during training")
plt.show()

y_pred = model.predict(X_test)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[2]*X_test.shape[1])

#scale back the prediction to orginal scale
temp = np.concatenate( (y_pred, X_test[:,1:nb_features] ) , axis =1)
temp = scaler.inverse_transform(temp)
test_pred = temp[:,0]

y_test = y_test.reshape( len(y_test), 1)

temp = np.concatenate( (y_test, X_test[:,1:nb_features] ) ,axis = 1)
temp = scaler.inverse_transform(temp)
real_data = temp[:,0]

#plot the prediction
plt.plot(test_pred, label = "prediction",c = "b")
plt.plot(real_data,label = "real data",c="r")
plt.title("comparison between prediction and real data")
plt.legend()
plt.show()

# Error
print("Mean absolute error : {}".format(mean_absolute_error(test_pred,real_data)) )
print("Mean squared error : {}".format(mean_squared_error(test_pred,real_data)) )
