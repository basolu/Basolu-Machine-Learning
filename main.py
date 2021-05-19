from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import numpy as np
import os
from sklearn.metrics import classification_report

train = int(input("Run training? (1/0): "))
print(train)

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
  filepath = prefix + group + '/Inertial Signals/'
  # load all 9 files as a single array
  filenames = list()
  # total acceleration
  filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
  # body acceleration
  filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
  # body gyroscope
  filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
  # load input data
  X = load_group(filenames, filepath)
  # load class output
  y = load_file(prefix + group + '/y_'+group+'.txt')
  return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
  # load all train
  trainX, trainy = load_dataset_group('train', prefix + 'drive/MyDrive/HARDataset/')
  print(trainX.shape, trainy.shape)
  # load all test
  testX, testy = load_dataset_group('test', prefix + 'drive/MyDrive/HARDataset/')
  subjects = read_csv('drive/MyDrive/HARDataset/train/subject_train.txt', header=None, delim_whitespace=True)
  print(testX.shape, testy.shape)
  # zero-offset class values
  trainy = trainy - 1
  testy = testy - 1
  # one hot encode y
  trainy = to_categorical(trainy)
  testy = to_categorical(testy)
  print(trainX.shape, trainy.shape, testX.shape, testy.shape)
  return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
  verbose, epochs, batch_size = 1, 15, 32
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  
  model = Sequential()
  #estrazione featuers
  #primo layer convoluzionale: applica le varie features per classificare
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(pool_size=2))
  #classificazione delle features estratte: 
  model.add(Flatten())
  #numero di neuroni (100) = numero di classi delle attività da individuare
  model.add(Dense(100, activation='relu')) 
  #n_output = number of subjects -> da impostare col n° delle attività
  #nell'ultimo layer il n° di neuroni dev'essere uguale al n° di etichette
  model.add(Dense(n_outputs, activation='softmax')) 
  tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit network
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard], validation_data=(testX, testy)
  )
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")
  # evaluate model
  _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
  return accuracy

# summarize scores
def summarize_results(scores):
  print(scores)
  m, s = mean(scores), std(scores)
  print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

#-----------------------------------------------------------------------------

def predict_prova(): 
  mode = int(input("Classification_report(1) or predict(0)? "))
  trainX, trainy, testX, testy = load_dataset()
  features = read_csv('drive/MyDrive/HARDataset/activity_labels.txt', header=None, delim_whitespace=True)
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")
  loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  _, accuracy = loaded_model.evaluate(testX, testy, batch_size=32, verbose=1)
  print("Accuracy: ",accuracy)
  columns = 6
  if(mode == 1):
    print("Start classification report")
    testX = testX.reshape(len(testX),128,9)
    result = (loaded_model.predict(testX))
    list_result = list()
    list_test = list()
    #Questa parte della funzione stampa la relativa features di tutti i sample
    for r in range(0,len(testy)): 
      index = 0
      max = 0
      for t in range(0,columns):
        prov = float(result[r][t])
        if(prov > max):
          max = prov
          index = t
        if(testy[r][t] == 1):
          list_test.append(t+1)
      list_result.append(index+1)
      #print(r+1, end=' ')
      #print(index+1, end=' ')
      #print(features[1][index])
    print(classification_report(list_test, list_result, target_names=features[1]))
    
  else:
    print("Start predict")
    request = 1
    while(request > 0):
      request = int(input("Inserire l'indice della misurazione da riconoscere (0 per uscire): "))
      target = testX[request]
      target = target.reshape(1,128,9)
      testX = testX.reshape(len(testX),128,9)
      result = (loaded_model.predict(target))
      #result = result.reshape(len(testX),6,1)
      columns = 6
      max = 0
      index = 0
      for t in range(0,columns):
        prov = result[0][t]
        if(prov > max):
          max = prov
          index = t
      print(request, end=' ')
      print(features[1][index], end=' ')
      print("({})" .format(index+1))
      
      print("Correct: ", testy[request])
    print("Fine prediction")
      
#------------------------------------------------------------------------------

# run an experiment
def run_experiment(repeats=10):
  print("Start experiment")
  # load data
  trainX, trainy, testX, testy = load_dataset()
  # repeat experiment
  scores = list()
  for r in range(repeats):
    score = evaluate_model(trainX, trainy, testX, testy)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)
	  # summarize results
    summarize_results(scores)

# run the experiment
if(train == 1):
   run_experiment()
predict_prova()

#identificazione: riconoscimento del soggetto all'interno di un set di utenti
#autenticazione: verifica che un dato sample, sia appartenente a quella determinata persona ***
#addestrare per riconoscere l'attività, e poi per distinguere gli indiivdui andiamo ad estrarre le features e confrontate
#con tecniche quali la verifica della loro distanza 
