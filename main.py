#essentials
import matplotlib.pyplot as plt

#data convert needs
import pandas as pd
import os
import glob
import time

# preparation for net
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import tensorflow as tf

#net
 #from keras.models import Sequential
 from keras.layers import Dense
 from keras.optimizers import Adam
 from keras.callbacks import TensorBoard

#alarm off
import warnings

from tensorflow.python.keras.models import Sequential

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)



# ======= Start ========== Join all the files in one cvs ===============================

# timer
# start = time.time()

# os.chdir(os.getcwd() + "/test")
# extension = 'xyz'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f, sep=" ", header=None) for f in all_filenames])
# #export to csv
# combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
# combined_csv.columns = ["res", "a", "b", "c"]
# print(combined_csv)

# timer
# end = time.time()
# print(end - start)

# ======= End ========== Join all the files in one cvs =================================


#=======================================================================================
#=======================================================================================


# ======= Start ========== Create separate cvs for every point =========================

# data = pd.read_csv('test/combined_csv.csv')
# data.columns = ["res", "a", "b", "c"]
# data_two = data.loc[data["res"] == 2]
# data_two.to_csv( "data_two.csv", index=False, encoding='utf-8-sig')
# data_three = data.loc[data["res"] == 3]
# data_three.to_csv( "data_three.csv", index=False, encoding='utf-8-sig')
# data_four = data.loc[data["res"] == 4]
# data_four.to_csv( "data_four.csv", index=False, encoding='utf-8-sig')
# data_five = data.loc[data["res"] == 5]
# data_five.to_csv( "data_five.csv", index=False, encoding='utf-8-sig')
# data_six = data.loc[data["res"] == 6]
# data_six.to_csv( "data_six.csv", index=False, encoding='utf-8-sig')
# data_seven = data.loc[data["res"] == 7]
# data_seven.to_csv( "data_seven.csv", index=False, encoding='utf-8-sig')

# ======= End ========== Create separate cvs for every point ===========================


#=======================================================================================
#=======================================================================================


# ======= Start ========== Create two databases for learning and test ==================

# examples_per_point = 1000
# #data two
# data = pd.read_csv('point db/data_two.csv')
# data_two = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_two[:examples_per_point]
# data_test = data_two[examples_per_point:]
#
# #data three
# data = pd.read_csv('point db/data_three.csv')
# data_three = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_learn.append(data_three[:examples_per_point])
# data_test = data_test.append(data_three[examples_per_point:])
#
# #data four
# data = pd.read_csv('point db/data_four.csv')
# data_four = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_learn.append(data_four[:examples_per_point])
# data_test = data_test.append(data_four[examples_per_point:])
#
# #data five
# data = pd.read_csv('point db/data_five.csv')
# data_five = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_learn.append(data_five[:examples_per_point])
# data_test = data_test.append(data_five[examples_per_point:])
#
# #data six
# data = pd.read_csv('point db/data_six.csv')
# data_six = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_learn.append(data_six[:examples_per_point])
# data_test = data_test.append(data_six[examples_per_point:])
#
# #data seven
# data = pd.read_csv('point db/data_seven.csv')
# data_seven = data.sample(n=(examples_per_point * 2), replace=True)
# data_learn = data_learn.append(data_seven[:examples_per_point])
# data_test = data_test.append(data_seven[examples_per_point:])
#
# #polish database
# data_learn.columns = ["res", "X", "Y", "Z"]
# data_test.columns = ["res", "X", "Y", "Z"]
# data_learn = data_learn.reset_index(drop=True)
# data_test = data_test.reset_index(drop=True)
#
# # #check
# # pd.set_option("display.max_rows", None, "display.max_columns", None)
# # print(data_learn)
# # print(data_test)
#
# #save databases
# data_learn.to_csv( "data_learn.csv", index=False, encoding='utf-8-sig')
# data_test.to_csv( "data_test.csv", index=False, encoding='utf-8-sig')

# ======= End ========== Create separate cvs for every point ===========================


#=======================================================================================
#=======================================================================================


# ======= Start ========== Neuron Net: NumPy Dataset ==================================================

#Two databases, 60 rows each for 10 examples per point (6 point in total: 2,3,4,5,6,7  ->  points 0,1,8,9 weren't in the files)
# 0 — неопределённая точка
# 1 — точка по умолчанию (Default)
# 2 — грунт (Ground)
# 3 — низкая растительность (трава) (Low vegetation)
# 4 — средняя растительность (кусты) (Medium vegetation)
# 5 — высока растительность (деревья) (High vegetation)
# 6 — здания (крыши) (Building)
# 7 — ложно отраженные точки (Low point)
# 8 — Model keypoints
# 9 — Other points

# columns = ["res", "X", "Y", "Z"]

# data_learn = pd.read_csv('data_learn.csv').sample(frac=1).reset_index(drop=True)
# data_test = pd.read_csv('data_test.csv').sample(frac=1).reset_index(drop=True)
#### С ЭТОГО МЕСТА
# data_learn = pd.read_csv('data_learn.csv')
# data_test = pd.read_csv('data_test.csv')
#
# #numpy target
# data_learn_target = data_learn['res']
# data_test_target = data_test['res']
#
# data_learn_target = data_learn_target.to_numpy()
# data_test_target = data_test_target.to_numpy()
#
# data_learn_target_ = data_learn_target.reshape(-1, 1)
# data_test_target_ = data_test_target.reshape(-1, 1)
# encoder = OneHotEncoder(sparse=False)
# Y_train = encoder.fit_transform(data_learn_target_)
# Y_test = encoder.fit_transform(data_test_target_)
#
# #numpy param
# data_learn = data_learn[['X', 'Y', 'Z']]
# data_test = data_test[['X', 'Y', 'Z']]
#
# X_train = data_learn.to_numpy()
# X_test = data_test.to_numpy()
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# print(X_train)
 #print(Y_train)
# # print(X_test)
# # print(Y_test)

# ======= End ============ Neuron Net: NumPy Dataset ===================================


#=======================================================================================
#=======================================================================================


# ======= Start ========== Neuron Net: Create Net =======================================

model = Sequential()
#neuron_amount = 8
#model.add(Dense(neuron_amount, input_shape=(3,), activation='relu', name='fc1'))
#model.add(Dense(neuron_amount, activation='relu', name='fc2'))
#model.add(Dense(neuron_amount, activation='relu', name='fc3'))
#model.add(Dense(neuron_amount, activation='relu', name='fc4'))
#model.add(Dense(neuron_amount, activation='relu', name='fc5'))
#model.add(Dense(6, activation='softmax', name='output'))
# optimizer = Adam(lr=0.001)
#optimizer = 'adam'
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ======= End ============ Neuron Net: Create Net =======================================


#=======================================================================================
#=======================================================================================


# ======= Start ========== Neuron Net: Train Net =======================================
#
# cb = TensorBoard()
# print('\n\n=========Model=========\n')
# history_callback = model.fit(X_train, Y_train,
#                              batch_size=5,
#                              epochs=50,
#                              verbose=0,
#                              validation_data=(X_test, Y_test),
#                              callbacks=[cb])
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# 2021-01-15 16:00:52
# Test loss: 1.6143566636244455
# Test accuracy: 0.26000000079472857

# 2021-01-15 16:01:41
# Test loss: 1.6328836731115977
# Test accuracy: 0.28

# size was 200 per point. let's do 400 ====

# 2021-01-15 16:04:19
# Test loss: 1.392438674371806
# Test accuracy: 0.36666666666666664

# 2021-01-15 16:05:17
# Test loss: 1.399537358970362
# Test accuracy: 0.35791666666666666

# 2021-01-15 16:09:38
# Test loss: 1.3739230721448743
# Test accuracy: 0.35041666666666665

# size was 400 per point. let's do 1000 ====

# 2021-01-15 16:13:35
# Test loss: 1.314191597767771
# Test accuracy: 0.391

# 2021-01-15 16:18:56
# Test loss: 1.3010446660596595
# Test accuracy: 0.4051666666666667

# 2021-01-15 16:20:54
# Test loss: 1.3094905971005526
# Test accuracy: 0.4071666666666667

# with 1000 it's stable around 4. let's boost amount of neurons from 8 to 12

# 2021-01-15 16:24:59
# Test loss: 1.2996178858183356
# Test accuracy: 0.4095

# 2021-01-15 16:26:39
# Test loss: 1.301056313059443
# Test accuracy: 0.4091666666666667

# 2021-01-15 16:28:44
# Test loss: 1.2986235093831382
# Test accuracy: 0.4066666666666667

# doesn't have an effect. As before (not logged, tryed up to 60 layers). let's go from 1000 to 10 000

# 2021-01-15 16:31:53
# Test loss: 1.2697448645745957
# Test accuracy: 0.42435

# 2021-01-15 16:43:15
# Test loss: 1.2707317158550029
# Test accuracy: 0.41391666666666665

# took around 10 min for no result. back to 1000

# 2021-01-15 16:55:21
# Test loss: 1.3143761766883448
# Test accuracy: 0.3973333333333333

#1000 ex per point. 8 neurons. 3 inner layers.

# 2021-01-15 17:16:37
# Test loss: 1.3121431509643617
# Test accuracy: 0.39966666666666667

# ======= End ============ Neuron Net: Train Net =======================================
