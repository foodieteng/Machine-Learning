'''
NTHU EE Machine Learning HW2
Author: 張浚騰
Student ID: 106091228
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse


# do not change the name of this function
def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    O1 = 6
    O2 = 6
    train_gre_max = np.max(train_data[:, 0])
    train_gre_min = np.min(train_data[:, 0])
    train_toefl_max =  np.max(train_data[:, 1])
    train_toefl_min =  np.min(train_data[:, 1])
    
    # get the slope of training data
    train_gre_slp =  (train_gre_max - train_gre_min)/(O1 - 1)
    train_toefl_slp =  (train_toefl_max - train_toefl_min)/(O2 - 1)
    train_feature = np.ones((len(train_data), O1 * O2 + 2))

    # preprocess training data
    for i in range(1, O1 + 1):    
        for j in range(1, O2 + 1): 
            k = O2 * (i - 1) + j 
            ui = train_gre_slp * (i - 1) + train_gre_min
            uj = train_toefl_slp * (j - 1) + train_toefl_min
            tmp1 = np.exp( (- ((train_data[:, 0] - ui) ** 2)) / (2 * (train_gre_slp ** 2)) - ((train_data[:, 1] - uj) ** 2) / (2 * (train_toefl_slp ** 2))) #為400*1的array
            train_feature[:, k - 1 ] = tmp1 
    train_feature[:, O1 * O2 ] = train_data[:, 2]  
    train_feature[:, O1 * O2 + 1] = 1   
    train_label = train_data[:, 3]

    # get the slope of testing data
    test_gre_max = np.max(test_data_feature[:, 0])
    test_gre_min = np.min(test_data_feature[:, 0])
    test_toefl_max =  np.max(test_data_feature[:, 1])
    test_toefl_min =  np.min(test_data_feature[:, 1])
    test_gre_slp =  (test_gre_max - test_gre_min)/(O1 - 1)
    test_toefl_slp =  (test_toefl_max - test_toefl_min)/(O2 - 1)
    test_feature = np.ones((len(test_data_feature), O1 * O2 + 2))

    # preprocess testing data
    for i in range(1, O1 + 1):    
        for j in range(1, O2 + 1):
            k = O2 * (i - 1) + j 
            ui = test_gre_slp * (i - 1) + test_gre_min
            uj = test_toefl_slp * (j - 1) + test_toefl_min
            tmp1 = np.exp( (- ((test_data_feature[:, 0] - ui) ** 2)) / (2 * (test_gre_slp ** 2)) - ((test_data_feature[:, 1] - uj) ** 2) / (2 * (test_toefl_slp ** 2))) 
            test_feature[:, k - 1 ] = tmp1 
    test_feature[:, O1 * O2 ] = test_data_feature[:, 2] 
    test_feature[:, O1 * O2 + 1] = 1   

    #belta is the shape parameter for the Gamma distribution prior to the Lambda parameter.
    temp = 0.001
    square_error = 100
    belta = 0
    count = 0

    # get the most fitting belta
    while(count < len(train_data)):
        w = np.dot(train_feature.T, train_feature) + temp * np.eye(O1 * O2 + 2)
        w = np.linalg.inv(w)   
        w = np.dot(w, train_feature.T)  
        w = np.dot(w, train_label)  
        temp2 = 0
        for i in range(len(test_feature)):
            temp2 = temp2 + (np.dot(train_feature[i], w) - train_label[i]) ** 2

        if(temp2 < square_error):
            square_error = temp2
            belta = temp
        temp = temp + 10 ** -6
        count = count + 1

    w = np.dot(train_feature.T, train_feature) + belta * np.eye(O1 * O2 + 2)  
    w = np.linalg.inv(w)   
    w = np.dot(w, train_feature.T) 
    w = np.dot(w, train_label)  

    # predict the chance
    prediction = np.zeros((len(test_feature), 1))   

    for i in range(len(test_feature)):  
        prediction[i] = np.dot(test_feature[i, :], w)    
    
    y_BLRprediction  = prediction
    y_BLRprediction = y_BLRprediction.reshape((100,))



    return y_BLRprediction 



# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    O1 = 4
    O2 = 4
    train_gre_max = np.max(train_data[:, 0])
    train_gre_min = np.min(train_data[:, 0])
    train_toefl_max =  np.max(train_data[:, 1])
    train_toefl_min =  np.min(train_data[:, 1])
    
    # get the slope of training data
    train_gre_slp =  (train_gre_max - train_gre_min)/(O1 - 1)
    train_toefl_slp =  (train_toefl_max - train_toefl_min)/(O2 - 1)
    train_feature = np.ones((len(train_data), O1 * O2 + 2))

    # preprpcess the traning data
    for i in range(1, O1 + 1):   
        for j in range(1, O2 + 1): 
            k = O2 * (i - 1) + j 
            ui = train_gre_slp * (i - 1) + train_gre_min
            uj = train_toefl_slp * (j - 1) + train_toefl_min
            tmp1 = np.exp( (- ((train_data[:, 0] - ui) ** 2)) / (2 * (train_gre_slp ** 2)) - ((train_data[:, 1] - uj) ** 2) / (2 * (train_toefl_slp ** 2))) #為400*1的array
            train_feature[:, k - 1 ] = tmp1 
    train_feature[:, O1 * O2 ] = train_data[:, 2]  
    train_feature[:, O1 * O2 + 1] = 1   
    train_label = train_data[:, 3]

    # caluate the weight
    w = np.dot(train_feature.T, train_feature) 
    w = np.linalg.inv(w)    
    w = np.dot(w, train_feature.T)  
    w = np.dot(w, train_label) 

    test_gre_max = np.max(test_data_feature[:, 0])
    test_gre_min = np.min(test_data_feature[:, 0])
    test_toefl_max =  np.max(test_data_feature[:, 1])
    test_toefl_min =  np.min(test_data_feature[:, 1])
    test_gre_slp =  (test_gre_max - test_gre_min)/(O1 - 1)
    test_toefl_slp =  (test_toefl_max - test_toefl_min)/(O2 - 1)
    test_feature = np.ones((len(test_data_feature), O1 * O2 + 2))
    
    for i in range(1, O1 + 1):   
        for j in range(1, O2 + 1): 
            k = O2 * (i - 1) + j 
            ui = test_gre_slp * (i - 1) + test_gre_min
            uj = test_toefl_slp * (j - 1) + test_toefl_min
            tmp1 = np.exp( (- ((test_data_feature[:, 0] - ui) ** 2)) / (2 * (test_gre_slp ** 2)) - ((test_data_feature[:, 1] - uj) ** 2) / (2 * (test_toefl_slp ** 2))) #為400*1的array
            test_feature[:, k - 1 ] = tmp1 
    test_feature[:, O1 * O2 ] = test_data_feature[:, 2]  
    test_feature[:, O1 * O2 + 1] = 1   
   
    # use suqare error observe the the best one
    squared_error = np.zeros((len(train_data), 1)) 
    prediction = np.zeros((len(test_feature), 1))   
    

    # predict the chance
    for i in range(len(test_feature)):  
        prediction[i] = np.dot(test_feature[i, :], w)    
        
    y_MLLSprediction  = prediction
    y_MLLSprediction = y_MLLSprediction.reshape((100,))

    
    return y_MLLSprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    


    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))
    


if __name__ == '__main__':
    main()