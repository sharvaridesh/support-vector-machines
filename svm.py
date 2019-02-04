#!/usr/bin/env python
# File: svm.py
# Author: Sharvari Deshpande <shdeshpa@ncsu.edu>

import pandas as pd
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC

# -----------------------Reading CSV File--------------------------#
df = pd.read_csv('supportvector.csv',header=None)
print(df, df.shape)

#---------------------------------TASK 1---------------------------#
X = df.drop([2], axis=1)
Y = df[2]
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
x = df.iloc[:, 0]
y = df.iloc[:, 1]
z = df.iloc[:, 2]
ax2.scatter(x, y, z, c=Y, cmap='rainbow')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Label')
plt.show()

#--------------------------TASK 2-------------------------------#
X_scaled = preprocessing.scale(X)
print(X_scaled)

scaler = StandardScaler()
X_1 = scaler.fit_transform(X)

# #-----------------------TASK 3-----------------------------------#
clf = svm.SVC(kernel='rbf')
features = X_scaled.astype(float)
target = Y.astype(float)
clf = clf.fit(features, target)
pred = clf.predict(features)

#------------------------TASKS 3,4,5,6----------------------------------#

#Coarse Search
C_range = np.logspace(-5.0, 15, num=11, base=2.0)
gamma_range = np.logspace(-15.0, 3.0, num=10, base=2.0)

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(n_splits=5)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_scaled, Y)
grid.predict(X_scaled)
print("The best parameters are %s and the maximum accuracy is %0.3f percent"
      % (grid.best_params_, (grid.best_score_*100)))
result = grid.cv_results_
C1 = result['param_C']
g1 = result['param_gamma']
score11 = result['mean_test_score']*100
c3 = C1.tolist()
g3 = g1.tolist()
c4 = []
g4 = []
for i in range(110):
    log_1 = np.log2(c3[i])
    c4.append(log_1)
    log_2 = np.log2(g3[i])
    g4.append(log_2)

#3D Plot - Coarse Search
fig = plt.figure()
ax1 = fig.gca(projection='3d')
surf = ax1.plot_trisurf(c4, g4, score11, cmap='rainbow', linewidth=0.2)
fig.colorbar(surf)
fig.tight_layout()
ax1.set_xlabel('log2(C), log of C to the base 2')
ax1.set_ylabel('log2(gamma), log of gamma to the base 2')
ax1.set_zlabel('Accuracy(%)')
plt.show()

#Finer search

C_range_1 = np.logspace(9.0, 13.0, num=17, base=2.0)
gamma_range_1 = np.logspace(-7.0, -3.0, num=17, base=2.0)

param_grid_1 = dict(gamma=gamma_range_1, C=C_range_1)
cv_1 = StratifiedKFold(n_splits=5)
grid_1 = GridSearchCV(SVC(), param_grid=param_grid_1, cv=cv_1)
grid_1.fit(X_scaled, Y)
print("The best parameters are %s and the maximum accuracy is %0.3f percent"
      % (grid_1.best_params_, (grid_1.best_score_*100)))
result_finer = grid_1.cv_results_
Cf = result_finer['param_C']
gf = result_finer['param_gamma']
score_f = result_finer['mean_test_score']*100
cf1 = Cf.tolist()
gf1 = gf.tolist()
cf2 = []
gf2 = []
for i in range(110):
    log_c = np.log2(cf1[i])
    cf2.append(log_c)
    log_g = np.log2(gf1[i])
    gf2.append(log_g)

#3D Plot - Finer Search
fig = plt.figure()
ax2 = fig.gca(projection='3d')
surf_f = ax2.plot_trisurf(cf2, gf2, score_f, cmap='rainbow', linewidth=0.2)
fig.colorbar(surf_f)
fig.tight_layout()
ax2.set_xlabel('log2(C), log of C to the base 2')
ax2.set_ylabel('log2(gamma), log of gamma to the base 2')
ax2.set_zlabel('Accuracy(%)')
plt.show()


