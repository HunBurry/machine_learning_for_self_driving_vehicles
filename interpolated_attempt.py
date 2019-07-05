# Author: Hunter Berry
# Date(s): June 3, 2019 (6/3/2019)
# Purpose: 
#    Recreate the results of the study '"My autonomous car is an elephant': A Machine Learning based Detector for Implausible Dimension"
#    This code will be a "proof on concept" to show that it is possible to interact with the Udacity datasets. 

# Description: 
#    I will not be using TensorFlow in this because XGBoost, in terms of speed and accuracy, preforms better. My idea in the future would 
#    be to use TensorFlow for images, and XGBoost for all other data types. Code is described throughout the file,
#    but the basic idea is to use 'time,' or the previous values located in a dataset, to allow the machine to look 
#    back, allowing it to detect abnormalities. 

'''
Contents of our File: 

|       frame_id        |    steering_angle   |   flag    |         id      |
1479425441182877835             -0.080904           0                1
1479425441232704425     	-0.070389           0                2
1479425441282730750	        -0.066530           0                3
1479425441332806714     	-0.061612           0                4
1479425441382790272	        -0.056694           0                5
1479425441432724303     	-0.052919           0                6
1479425441482746958	        -0.050717           0                7
1479425441482760929             -0.323212           1                8

The 'flag' is the column I have added into the data set to determine whether or not the algorithm classifies
the data as 'abnormal' or not. What 'abnormal' actually can be defined as is what we hope to see from this
experemint.

For details on how the random steering angles (the 'abnormal' data) were created, see 'setup_files.py'. 

Later on, during the preprocessing phase, the data will expand by three columns, so that it looks like the following:

|       frame_id        |    steering_angle   |   flag    |         id      |      (x-1)    |     (x-2)    |  diff
1479425441182877835             -0.080904           0                1              NA              NA          NA
1479425441232704425     	-0.070389           0                2                  NA              NA          NA
1479425441282730750	        -0.066530           0                3              -0.070389       -0.080904    -0.04042                    
1479425441332806714     	-0.061612           0                4               -0.066530      -0.070389     .005
1479425441382790272	        -0.056694           0                5               -0.061612      -0.066530      .045
1479425441432724303     	-0.052919           0                6               -0.056694      -0.061612      .004
1479425441482746958	        -0.050717           0                7              -0.052919       -0.056694       .002
1479425441482760929             -0.323212           1                8          -0.050717      -0.052919        .28                                     

This is the 'time' based data. A more detailed description exists within the code. 

'''

#------------------------------------------------------------------------- Imports --------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore") #Ignores depreciation warnings and other non-needed information.

import pandas as pd
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)

#If these are not install, run the following commands in your PC or in a Jupyter notebook.
#pip install tensorflow
#pip install pandas
#pip install sklearn
#pip install numpy

#------------------------------------------------------------------ Setup and Preprocessing --------------------------------------------------------------------

np.set_printoptions(suppress=True) #Easier to print and work with Numpy arrays. 

data = pd.read_csv('interpolated_final_for_attempts.csv') #Import our data.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['angle'] = scaler.fit_transform(data[['angle']])
scaler2 = MinMaxScaler()
data['torque'] = scaler2.fit_transform(data[['torque']])
scaler3 = MinMaxScaler()
data['speed'] = scaler3.fit_transform(data[['speed']])
#scaler4 = MinMaxScaler()
#data['baking'] = scaler4.fit_transform(data[['speed']])
#data.dropna(axis=0, subset=['steering_angle'], inplace=True) #Drop any rows with some kind of missing data.
y = data.flag #Set y to the value we want to predict, the 'flag' value. 
X = data.drop(['flag', 'frame_id', 'timestamp', 'index', 'width', 'height', 'filename', 'lat', 'long', 'alt',], axis=1) 
#Drop the frame_id (since 'id' is being used to simulate time), and flag, since it is our prediction value. Drop other non-useful data.
#X Should now be the id, angle, torque, speed, lat, long, and alt values. 

#To simulate time within the program, we use various lags, which show the value at an earlier time. After doing testing with lags from (X-1) to (X-5),
#having only X, (X-1), and (X-2) proved to be benefitical to the program's accuracy.

X = concat([X.shift(3), X.shift(2), X.shift(1), X], axis=1) #Set our data to contains the two previously mentioned lag datas, as well as the initial data. 
X.columns = ['angle-3', 'torque-3', 'speed-3', 'id3', 'angle-2', 'torque-2', 'speed-2', 'id2', 'angle-1', 'torque-1', 'speed-1', 'id1', 'angle', 'torque', 'speed', 'id'];
#X.columns = ['torque-3', 'angle-3', 'id3', 'torque-2', 'angle-2', 'id2', 'torque-1', 'angle-1', 'id1', 'torque', 'angle', 'id'];
X = X.drop(['id3', 'id2', 'id1'], axis=1) #Drop the previous ID amounts. 

#do we want to inlcude lat/long/ whatever alt is?


#I also created various other features to see if any would prove to improve accuracy within the program.  I tried the average of two numbers
#((X + (X-1))/2), addition (X + (X-1)), and subtraction (X - (X-1)). Out of these, only the difference proved to be benefitcial.

X = X.iloc[3:,]; 
y = y.iloc[3:,];

X["diff-steering-1"] = X["angle"] - X["angle-1"]
X["diff-steering-2"] = X["angle"] - X["angle-2"]
X["diff-steering-3"] = X["angle"] - X["angle-3"]
X["diff-torque-1"] = X["torque"] - X["torque-1"]
X["diff-torque-2"] = X["torque"] - X["torque-2"]
X["diff-torque-3"] = X["torque"] - X["torque-3"]
X["diff-speed-1"] = X["speed"] - X["speed-1"]
X["diff-speed-2"] = X["speed"] - X["speed-2"]
X["diff-speed-3"] = X["speed"] - X["speed-3"]
#X = X.drop(['angle-3', 'torque-3', 'speed-3'], axis=1)
X = X.drop(['angle-3', 'torque-3', 'speed-3', 'angle-2', 'speed-2', 'torque-2', 'torque-1', 'angle-1', 'speed-1', 'angle', 'torque', 'speed', 'id'], axis=1);

'''
Mean Absolute Error : 0.04959350148362157

There were 4144 sections of 'fake' data in the file. We guessed on 3694.
We got 3655 right out of 4144(0.8819980694980695).

We threw 39 false alarms. The false alarms were on the following sections of data.

-------------------

Mean Absolute Error : 0.013687451314959754

There were 4144 sections of 'fake' data in the file. We guessed on 3857.
We got 3816 right out of 4144(0.9208494208494209).

We threw 41 false alarms. The false alarms were on the following sections of data.
'''


order = ['id', 'angle', 'angle-1', 'angle-2', 'diff-steering-1',
        'diff-steering-2', 'torque', 'torque-1', 'torque-2', 'diff-torque-1',
        'diff-torque-2', 'speed', 'speed-1', 'speed-2', 'diff-speed-1', 'diff-speed-2']

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.4, shuffle=False)

'''my_model = XGBRegressor(n_estimators=1000, learning_rate=0.0001) 
my_model.fit(train_X, train_y, early_stopping_rounds=100, 
             eval_set=[(test_X, test_y)], verbose=False)'''

my_model = XGBClassifier(max_depth=15, n_estimators=1000, learning_rate=0.1) 
my_model.fit(train_X, train_y, early_stopping_rounds=100, 
             eval_set=[(test_X, test_y)], verbose=False)

#--------------------------------------------- Predictions and Data ------------------------------------------------

predictions = my_model.predict(test_X)

numOfPredictions = 0; #Total number of values that the algorhtms thinks are fake
correct = 0; #Items that are actually fake. 
myCX = []
myCP = []
falseAlarmsAngles = [];
falseAlarmsTorque = [];
falseAlarmSpeeds = [];
falseAlarmsPredictions = [];
falseAlarmsIDs = [];
numOfOnes = np.count_nonzero(test_y); #Number of 'fake' data points within the testing data. 

#Prints out all of the values that were predicted to be 'fake' values, then stores them for some data analysis later.

for i in range(len(predictions)):
    if (predictions[i] >= .5 - mean_absolute_error(predictions, test_y)):
        numOfPredictions = numOfPredictions + 1
        #print("Predicted a value of " + str(predictions[i]) + " for test_X[" + str(i) + "]. Actual value: " + str(test_y[i]) + ".")
        if(test_y[i] == 1):
            correct = correct + 1;
            #myCP.append(predictions[i])
            #myCX.append([test_X[i][9], test_X[i][10], test_X[i][11]])
        #else:
         #   falseAlarmsAngles.append(test_X[i][9])
          #  falseAlarmsTorque.append(test_X[i][10])
           # falseAlarmSpeeds.append(test_X[i][11])
           ## falseAlarmsIDs.append(test_X[i][12]);
            #falseAlarmsPredictions.append(predictions[i]);

#totalFalseAlarms = pd.DataFrame({'steering_angle': falseAlarmsAngles, 'torque': falseAlarmsTorque, 'speed': falseAlarmSpeeds,
 #                                'predicted_value': falseAlarmsPredictions, 'id': falseAlarmsIDs}); #Puts all false alarm data in a dataframe.


"""
The following simply finds all the values the algorithm failed to classify as 'fake' even though they were, and puts them in a data frame. 
missedAngles = []
missedPredictions = []
missedIDs = [];
allFakes = np.where(test_y == 1);
for i in allFakes:
    for x in i:
        if(test_X[x][4] not in myCX):
            missedPredictions.append(predictions[x])
            missedAngles.append(test_X[x][2])
            missedIDs.append(test_X[x][3]);
            """

#totalMissed = pd.DataFrame({'steering_angle': missedAngles, 'predicted_value': missedPredictions, 'id': missedIDs});

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)) + "\n");

print("There were " + str(numOfOnes) + " sections of 'fake' data in the file. We guessed on " + str(numOfPredictions) + ".")
print("We got " + str(correct) + " right out of " + str(numOfOnes) + "(" + str((correct/numOfOnes)) + ").\n")

print("We threw " + str(abs(numOfPredictions - correct)) + " false alarms. The false alarms were on the following sections of data. \n");
#print(totalFalseAlarms)
print("\n");

#print("The 'fake' data that we failed to correctly classify is in the following data. \n");
#print(totalMissed)
#print("\n");

h1 = test_X[0:1]
h2 = test_X[0:100]

print(h1)

print(h2) 
import time

start = time.clock()
predictions = my_model.predict(h1)
end = time.clock()
#print(start)
#print(end)
print(end - start)

start2 = time.clock()
predictions = my_model.predict(h2)
end2 = time.clock()
print(end2 - start2)



"""
There were 3063 sections of 'fake' data in the file. We guessed on 3390.
We got 2671 right out of 3063(0.8720208945478289).
"""
