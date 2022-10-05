### 
# In the "Wine Data Set", there are 3 types of wines and 13 different 
# features of each instance -> [178*14] matrix (178 instances and 14 
# features). In this HW, we will implement the MAP of the classifier
# for 54 instances with their features.
#
# Goal: determine which type of wine according to the given features
# 
# 1. Split dataset to train and test datasets, for each type of wine, 
#    randomly split 18 for testing. train_set:[124*14] test_set:[54*14]
# 2. Evaluate posterior probabilities with likelihood fcns and prior 
#    distribution of the training set.
# 3. Calculate the accuracy rate of the MAP detector (should exceed 90%)
# 4. Plot visualized result of the "testing data" (with built in PCA fcn)
#
# *Note: all features are independent and the distribution of them is Gaussian distribution
###

from cgi import test
from math import exp, sqrt, pi
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Build function for creating a Normal (Gaussain) distribution function
def normal_dist(x, mean, stdev):
    prob_density = 1/sqrt(2*pi)/stdev*exp(-pow(x-mean,2)/2/stdev/stdev)
    # prob_density = pi*stdev*exp(-0.5*((x-mean)/stdev)**2)
    return prob_density

# independent features with gaussian distrobution (14 features)
features = ["Wine type","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", 
            "Total phenols", "Flavanoids", "Non Flavanoid phenols", "Proanthocyanins",
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# Read .csv file  
wineDataSet = pd.read_csv('Wine.csv', header=None)
# Add title to dataset
# wineDataSet.columns = features

# Find the number of different wines in the data set
sets_of_wine = [0,0,0]
for i in range(wineDataSet.shape[0]):
    sets_of_wine[int(wineDataSet[0].values[i])-1]+=1

# Wine 1: 1-59, Wine 2: 60-130, Wine 3: 131-178
wine1 = wineDataSet.iloc[               :sets_of_wine[0]                ,:]
wine2 = wineDataSet.iloc[sets_of_wine[0]:sets_of_wine[0]+sets_of_wine[1],:]
wine3 = wineDataSet.iloc[sets_of_wine[0]+sets_of_wine[1]:               ,:]

# Sample the wine according to its label 
wine1_test = wine1.sample(n=18)
wine1_train = wine1.drop(wine1_test.index)
wine2_test = wine2.sample(n=18)
wine2_train = wine2.drop(wine2_test.index)
wine3_test = wine3.sample(n=18)
wine3_train = wine3.drop(wine3_test.index)

# Build train and test sets 
train_data = pd.concat([wine1_train, wine2_train, wine3_train])
test_data = pd.concat([wine1_test, wine2_test, wine3_test])


# Find the number of different wines in the training set
sets_of_wine = [0,0,0]
for i in range(train_data.shape[0]):
    sets_of_wine[train_data[0].values[i]-1]+=1

# Calculate mean and standard deviation of each feature of different wine. 
# Stored as [mean_feature1, stdev_feature1], ... (13 for each wine)
feature_distribution = []
data_accum = 0
for i in range(3):
    for j in range(13):
        # calculate mean and stdev for train_data[0:sets_of_wine[i]][j+1]
        mean  = np.average(train_data[j+1].values[data_accum:data_accum+sets_of_wine[i]])
        stdev = np.std(train_data[j+1].values[data_accum:data_accum+sets_of_wine[i]])
        feature_distribution.append([mean,stdev])
    data_accum += sets_of_wine[i]

# Calculate prior for each wine
priors = [0,0,0]
train_total = sets_of_wine[0]+sets_of_wine[1]+sets_of_wine[2]
for i in range(3):
    priors[i] = sets_of_wine[i]/train_total

# Thought process
# 1. maybe put in the test data into the individual pdfs and sum them up
# 2. mutiply it by the prior
# 3. compare the 3 values and find the maximum -> the most likely wine!

# 1. go through the test datas, 
# 2. go through the three wines,
# 3. compute the all 13 pdfs for each wine 
# 4. multiply it with the prior
# 5. find the max value of the three wines 
# 6. get the MAP of the three kinds of wines
# 7. compare it with the labels (if correct -> correctly_labeled+=1)
# 8. calculate the accuracy

# Calculate the MAP for all test data
wine_posterior = [0,0,0]
correctly_labeled = 0
prediction = []

for i in range(test_data.shape[0]):
    for j in range(3):
        posterior = 1
        for k in range(13):
            posterior *= normal_dist(test_data[k+1].values[i], feature_distribution[13*j+k][0], feature_distribution[13*j+k][1])
        posterior *= priors[j]
        wine_posterior[j] = posterior
    # print(wine_posterior)
    prediction.append(wine_posterior.index(max(wine_posterior))+1)
    if prediction[i] == test_data[0].values[i]:
        correctly_labeled+=1

accuracy = correctly_labeled/test_data.shape[0]
print(accuracy)
print('%')

print(prediction, len(prediction))

# Show results with sklearn.decomposition.PCA function
test_data_no_label = test_data.iloc[: , 1:]
labels = test_data[0].tolist()
print(labels)

# Plot test data using PCA with dimentions decreased to 3D
pca = PCA(n_components=3) 
pca.fit(test_data_no_label) 
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_)

test_pca_3d = pca.transform(test_data_no_label) 
fig1 = plt.figure() 
ax = Axes3D(fig1, rect=[0, 0, 1, 1], elev=30, azim=20)
for i in range(len(labels)):
    if labels[i] == 1:
        c = 'r'
        target_name = 'wine 1'
    elif labels[i] == 2:
        c = 'g'
        target_name = 'wine 2'
    else:
        c = 'b'
        target_name = 'wine 3'
    ax.scatter(test_pca_3d[i, 0], test_pca_3d[i, 1],test_pca_3d[i, 2],marker='o',c=c, label=target_name)

# Plot test data using PCA with dimentions decreased to 2D
pca2d = PCA(n_components=2) 
pca2d.fit(test_data_no_label) 
print(pca2d.explained_variance_ratio_) 
print(pca2d.explained_variance_)

test_pca_2d = pca2d.transform(test_data_no_label) 
fig2 = plt.figure() 
for i in range(len(labels)):
    if labels[i] == 1:
        c = 'r'
        target_name = 'wine 1'
    elif labels[i] == 2:
        c = 'g'
        target_name = 'wine 2'
    else:
        c = 'b'
        target_name = 'wine 3'
    plt.scatter(test_pca_2d[i, 0], test_pca_2d[i, 1],marker='o',c=c, label=target_name)
plt.show()


# References:
# https://towardsdatascience.com/mle-vs-map-a989f423ae5c
# https://kknews.cc/zh-tw/code/kvzpj5b.html