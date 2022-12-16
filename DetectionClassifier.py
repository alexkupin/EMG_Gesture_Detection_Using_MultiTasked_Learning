# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:35:28 2022

@author: kupin
"""
import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib.figure import SubplotParams
import matplotlib.pyplot as plot
import pyemgpipeline as pep
import math
import itertools
import operator
from sklearn import preprocessing
from sklearn.svm import SVC

#Function derives the RMS given a window of time
def rms(window):
  sumXSqared=0
  for x in window:
    sumXSqared+=x*x
  return math.sqrt(sumXSqared/len(window))



def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def getMeanAbsoluteValue(window):
  #Get the absolute Value of mean
  sumX=0
  for x in window:
    sumX+=x
  mean = sumX/len(window)
  sumX=0 
  for index in window:
    sumX+=index-mean
  return sumX/len(window)

#function gets the slope sign change given a window
def getSlopeSign(window):
  m=np.diff(window)
  sumX = 0
  for x in m:
    sumX+=x
  mean = sumX/len(m)
  return (int)(0>=(sumX/len(m)))

#function gets the willison Amplitude given a threshold value and a window of time
def getWillisonAmp(window, threshold):
  count = 0
  for x in window:
    if x>threshold:
      count+=1
  return count

def getBestFeaturesDTC(X,Y):
    newfeatureSet=pandas.DataFrame()
    DTC=DecisionTreeClassifier(criterion="gini")
    DTC.fit(X, Y)
    
    featureImportance = zip(X.columns, DTC.feature_importances_)
    newFeatureIndecies=[]
    index=0 
    for feature,value in featureImportance:
        if value>0.01:
            newFeatureIndecies.append(feature)
            
    return newFeatureIndecies        
    
    
    
            
    
    
    

    
##############################################################################################################################
##############################################################################################################################
#begin data processing
df = pandas.DataFrame()
li=[]
#iterate through folders
for x in range(1,37):
  if(x<10):
    pathf = 'C:/Users/kupin/Documents/GitHub/EMG_Gesture_Detection_Using_MultiTasked_Learning/EMG_data_for_gestures-master/0' + str(x)  

  else:
    pathf = 'C:/Users/kupin/Documents/GitHub/EMG_Gesture_Detection_Using_MultiTasked_Learning/EMG_data_for_gestures-master/' + str(x)

  for filename in os.listdir(pathf):
    f = os.path.join(pathf, filename)
    # checking if it is a file
    if os.path.isfile(f):
       li.append(pandas.read_csv(f,delim_whitespace=True))
       
       
#EMG plotting parameters to view plots
emg_plot_params = pep.plots.EMGPlotParams(
    n_rows=8,
    fig_kwargs={
        'figsize': (8, 6),
        'dpi': 80,
        'subplotpars': SubplotParams(wspace=0, hspace=0.6),
    },
    line2d_kwargs={
        'color': 'red',
    }
)

df = pandas.concat(li,ignore_index=True)
dataArr=df.to_numpy()
#choosing a train test split of 80-10-10 because it is a smaller dataset so it needs a larger amount to train on.
[train,test]=train_test_split(dataArr, test_size = 0.2, shuffle = False)
#Setting Up SKlearn Kfold Cross validation
crossValidation = KFold(n_splits=10, random_state=1, shuffle=True)

#split up labels and training
X = train[:,1:9]
Y = train[:,9]

sample_rate = 1000
channel_names=["channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"]

m = pep.wrappers.EMGMeasurement(X, hz=1000, trial_name="Channels",
                                channel_names=channel_names, emg_plot_params=emg_plot_params)
m.plot()

#apply DC offset remover
m.apply_dc_offset_remover()
m.plot()

#apply Bandpass filter
#Lower the low cutoff frequency
m.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=2, bf_cutoff_fq_hi=450)
m.plot()

#apply notch filter 50 hz zone
#TODO


#Get Lowest RMS for each channel
lowestRMS=np.ones(8)
prevTime=-1.0

for channel in range(0,8):
  for index in range(10,len(X),10):
    if(index+10>=len(m.data)):
      break
    else:
       rmsCurr = rms(m.data[index-10:index+10,channel])
       if(rmsCurr<lowestRMS[channel]):
         lowestRMS[channel] = rmsCurr

  #applies the lowest RMS  threshold to the raw data channel to cut out baseline noise.
  m.data[:,channel] = m.data[:,channel] - np.full(np.shape(m.data[:,channel]),lowestRMS[channel])
m.plot()



#bins the X data to respected overlapped windows
#bins the Y data to the overlapped windows
newXData = list()
newYData = list()
for index in range(125,len(X),125):
      if(index+124>=len(m.data)):
        break
      newXData.append(m.data[index-125:index+125,channel])
      #gets the most common Y value and uses that for the label
      newYData.append(most_common(Y[index-125:index+125]))


#normalize data between 0 and 1

min_max_scaler = preprocessing.MinMaxScaler()
normalized_X = min_max_scaler.fit_transform(np.array(newXData))
newXData = normalized_X.tolist() 


#Generate features
newRMSData= np.zeros((len(newXData),8))
meanAbsoluteValue= np.zeros((len(newXData),8))
slopeSignChange = np.zeros((len(newXData),8))
willisonAmplitude = np.zeros((len(newXData),8))
variance = np.zeros((len(newXData),8))
prevTime=-1.0;
for channel in range(0,8):
  count=0
  for index in range(125,len(X),125):
    if(index+124>=len(m.data)):
      break
    else:
      newRMSData[count,channel] = (rms(m.data[index-125:index+125,channel]))
      meanAbsoluteValue[count,channel] = (getMeanAbsoluteValue(m.data[index-125:index+125,channel]))
      slopeSignChange[count,channel] = (getSlopeSign(m.data[index-125:index+125,channel]))
      willisonAmplitude[count,channel] = (getWillisonAmp(m.data[index-125:index+125,channel],1e-5))
      variance[count,channel] = np.var(m.data[index-125:index+125,channel])
    count+=1

      
#put all features in to a dataframe
processedData = pandas.DataFrame(meanAbsoluteValue, columns =["MAVchannel1","MAVchannel2","MAVchannel3","MAVchannel4","MAVchannel5","MAVchannel6","MAVchannel7","MAVchannel8"])
for channel in range(0,8): 
  processedData["RMS" + str(channel+1)] = newRMSData[:,channel]
  processedData["Var" + str(channel+1)] = variance[:,channel]
  processedData["WA" + str(channel+1)] = willisonAmplitude[:,channel]
  processedData["SSC" + str(channel+1)] = slopeSignChange[:,channel]

#bins the X Test data to respected overlapped windows
#bins the Y Test data to the overlapped windows
X_Test = test[:,1:9]
Y_Test = test[:,9]



#D-Tree based Feature combination
prunedData = pandas.DataFrame()
bestFeatureList=getBestFeaturesDTC(processedData,newYData)
for feature in bestFeatureList:
    prunedData[feature]= processedData[feature]



#########################################################################################################
#Run Pipeline on test Data
sample_rate = 1000
channel_names=["channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"]

m_test = pep.wrappers.EMGMeasurement(X_Test, hz=1000, trial_name="Channels",
                                channel_names=channel_names, emg_plot_params=emg_plot_params)
#apply DC offset remover
m_test.apply_dc_offset_remover()
m_test.apply_bandpass_filter(bf_order=4, bf_cutoff_fq_lo=2, bf_cutoff_fq_hi=450)
#Get Lowest RMS for each channel
lowestRMS=np.ones(8)
prevTime=-1.0

for channel in range(0,8):
  for index in range(10,len(X),10):
    if(index+10>=len(m_test.data)):
      break
    else:
       rmsCurr = rms(m_test.data[index-10:index+10,channel])
       if(rmsCurr<lowestRMS[channel]):
         lowestRMS[channel] = rmsCurr

  #applies the lowest RMS  threshold to the raw data channel to cut out baseline noise.
  m_test.data[:,channel] = m_test.data[:,channel] - np.full(np.shape(m_test.data[:,channel]),lowestRMS[channel])

newXTestData = list()
newYTestData = list()
for index in range(125,len(X_Test),125):
      if(index+124>=len(m_test.data)):
        break
      newXTestData.append(m_test.data[index-125:index+125,channel])
      #gets the most common Y value and uses that for the label
      newYTestData.append(most_common(Y_Test[index-125:index+125]))

#normalize data between 0 and 1
normalized_X = min_max_scaler.fit_transform(np.array(newXTestData))
newXTestData = normalized_X.tolist()  


newRMSData= np.zeros((len(newXTestData),8))
meanAbsoluteValue= np.zeros((len(newXTestData),8))
slopeSignChange = np.zeros((len(newXTestData),8))
willisonAmplitude = np.zeros((len(newXTestData),8))
variance = np.zeros((len(newXTestData),8))

#TODO
waveformLength = np.zeros((len(newXTestData),8))
simpleSquareIntegral = np.zeros((len(newXTestData),8))
secondOrderMovement = np.zeros((len(newXTestData),8))
#TODO

prevTime=-1.0;
for channel in range(0,8):
  count=0
  for index in range(125,len(X_Test),125):
    if(index+124>=len(m_test.data)):
      break
    else:
      newRMSData[count,channel] = (rms(m_test.data[index-125:index+125,channel]))
      meanAbsoluteValue[count,channel] = (getMeanAbsoluteValue(m_test.data[index-125:index+125,channel]))
      slopeSignChange[count,channel] = (getSlopeSign(m_test.data[index-125:index+125,channel]))
      willisonAmplitude[count,channel] = (getWillisonAmp(m_test.data[index-125:index+125,channel],1e-5))
      variance[count,channel] = np.var(m_test.data[index-125:index+125,channel])
      
    count+=1

  processedData_test = pandas.DataFrame(meanAbsoluteValue, columns =["MAVchannel1","MAVchannel2","MAVchannel3","MAVchannel4","MAVchannel5","MAVchannel6","MAVchannel7","channel8"])    
  for channel in range(0,8): 
    processedData_test["RMS" + str(channel+1)] = newRMSData[:,channel]
    processedData_test["Var" + str(channel+1)] = variance[:,channel]
    processedData_test["WA" + str(channel+1)] = willisonAmplitude[:,channel]
    processedData_test["SSC" + str(channel+1)] = slopeSignChange[:,channel]

#D Tree Based Feature selection
prunedDataTest = pandas.DataFrame()
for feature in bestFeatureList:
    prunedDataTest[feature] = processedData_test[feature]
    
#Train SVM and KNN to compare to the proposed model. 
svmModel = SVC(gamma = 'auto')
svmModel.fit(prunedData, newYData)
svmScore= svmModel.score(prunedDataTest, newYTestData)
print("SVM Score:" +  str(svmScore))
    
KnnModel = KNeighborsClassifier(7)
KnnModel.fit( prunedData, newYData )
KnnScore= KnnModel.score( prunedDataTest, newYTestData )
print( "SVM Score:" +  str(KnnScore ))


#grid search on different learning rates, estimator types, and number of estimators
weak_learners = [DecisionTreeClassifier(max_depth=1),RandomForestClassifier(n_estimators=1),GradientBoostingClassifier(random_state=0)] 
numEstimators = [10,50,100,200,300]
learning_rates = [0.1, 0.01, 0.001] 
scores = list()
for wl in range(len(weak_learners)):
  for numEst in range(len(numEstimators)):
    for learningRate in range(len(learning_rates)):
      model = AdaBoostClassifier(base_estimator=weak_learners[wl], learning_rate = learning_rates[learningRate], n_estimators=numEstimators[numEst])
      model.fit(prunedData,newYData)
      currScore=model.score(prunedDataTest,newYTestData)
      print("Model: "+ str(wl)+", NumEstimators: "+str(numEstimators[numEst])+", learning Rate: "+str(learning_rates[learningRate])+", Score: "+str(currScore))
      scores.append([wl, numEstimators[numEst], learning_rates[learningRate],currScore])

