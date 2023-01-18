#Python 3.10.9
#Soham Naik 1/17/2023
#Linear-Regression Project, currently using the boston.txt data set.

"""
 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#Select Data
STR_X = 'LSTAT'
STR_Y = 'MEDV'

Boston_list = []
Data_DCT_Names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
Data_DCT = {}

# Open the file and read the data into a list of dictionaries
with open(r"ML-Journey\Supervised-Learning\Linear-Regression\boston.txt") as fp:
    Temp_DCT = {}
    data_string = ""
    for i, line in enumerate(fp):
        if i%2==0:
            data_string+=line
        else:
            data_string+=line
            string_list = data_string.split()
            float_list = []
            for i in range(len(string_list)):
                float_list.append(float(string_list[i]))
            # Map the values in float_list to the keys in the dictionary
            Temp_DCT["CRIM"] = float_list[0]
            Temp_DCT["ZN"] = float_list[1]
            Temp_DCT["INDUS"] = float_list[2]
            Temp_DCT["CHAS"] = float_list[3]
            Temp_DCT["NOX"] = float_list[4]
            Temp_DCT["RM"] = float_list[5]
            Temp_DCT["AGE"] = float_list[6]
            Temp_DCT["DIS"] = float_list[7]
            Temp_DCT["RAD"] = float_list[8]
            Temp_DCT["TAX"] = float_list[9]
            Temp_DCT["PTRATIO"] = float_list[10]
            Temp_DCT["B"] = float_list[11]
            Temp_DCT["LSTAT"] = float_list[12]
            Temp_DCT["MEDV"] = float_list[13]
            # Add the dictionary to the list    
            Boston_list.append(Temp_DCT)
            # Reset the data string and dictionary
            data_string=""
            Temp_DCT = {}
    fp.close()

#Standardizes data, subtracts the mean and divides by standard deviation.
def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std
# Extract the data for each feature into a separate array
for i in range(len(Data_DCT_Names)):
    temp_arr = []
    for j in range(len(Boston_list)):
        temp_arr.append(Boston_list[j][Data_DCT_Names[i]])
    Data_DCT[Data_DCT_Names[i]] = normalize(np.array(temp_arr))

#find min and max value of the numpy arrays
def min(arr):
    min = arr[0]
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]
    min = math.floor(min)
    return min         
def max(arr):
    max = arr[0]
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
    max = math.ceil(max)
    return max

#linear regression functions
def MSE_Calculus(X,y):
    # Calculate the mean of X and y
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    # Calculate the sum of the squared differences between X and the mean of X
    sum_x_diff_squared = np.sum((X - x_mean)**2)
    # Calculate the slope using the formula:
    # slope = sum((X - mean(X)) * (Y - mean(Y))) / sum((X - mean(X))**2)
    slope = np.sum((X - x_mean) * (y - y_mean)) / sum_x_diff_squared
    # Calculate the intercept using the formula:
    # intercept = mean(Y) - slope * mean(X)
    intercept = y_mean - slope * x_mean
    return intercept, slope
def MSE_Matrix(X, y):
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1,1)))
    # Compute the optimal weights
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    # Extract the slope and intercept from the weights
    intercept = weights[0]
    slope = weights[1]
    return intercept, slope
def Gradient_Descent(X, y, learning_rate=0.01, num_iterations=10000):
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1,1)))
    # Initialize the weights randomly
    weights = np.random.randn(X.shape[1])
    for i in range(num_iterations):
        # Compute the dot product of X and the weights
        y_pred = np.dot(X, weights)
        # Compute the error
        error = y_pred - y
        # Compute the gradient
        gradient = np.dot(X.T, error) / X.shape[0]
        # Update the weights
        weights -= learning_rate * gradient 
    intercept = weights[0]
    slope = weights[1]
    return intercept, slope
def Stochastic_Gradient_Descent(X, y, learning_rate=0.01, num_iterations=10000):
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1,1)))
    # Initialize the weights randomly
    weights = np.random.randn(X.shape[1])
    for i in range(num_iterations):
        # Pick a random training example
        # instead of using the entire dataset to compute the gradient at each step, 
        # it uses only one randomly chosen example from the dataset. 
        # This is what makes it "stochastic" gradient descent.
        rand_ind = np.random.randint(X.shape[0])
        x_i = X[rand_ind]
        y_i = y[rand_ind]
        # Compute the dot product of x_i and the weights
        y_pred = np.dot(x_i, weights)
        # Compute the error
        error = y_pred - y_i
        # Compute the gradient
        gradient = np.dot(x_i.T, error)
        # Update the weights
        weights -= learning_rate * gradient 
    intercept = weights[0]
    slope = weights[1]
    return intercept, slope
def Ridge_Regression(X, y, lambda_=0):
    # Here, the lambda_ parameter controls the strength of the regularization term. 
    # A higher value of lambda_ will result in a smaller magnitude of the weights and therefore a simpler model.
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1,1)))
    # Compute the ridge parameter
    ridge = lambda_ * np.eye(X.shape[1])
    # Compute the weights using the closed-form solution
    weights = np.linalg.inv(X.T @ X + ridge) @ X.T @ y
    intercept = weights[0]
    slope = weights[1]
    return intercept, slope

#create graph, edit linear regression model
def MakeGraph(STR_X,STR_Y):
    #SETUP GRAPH ATRIBUTES
    plt.style.use('_mpl-gallery')
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["figure.autolayout"] = True
    fig, graph = plt.subplots()
    graph.scatter(x=Data_DCT[STR_X], y=Data_DCT[STR_Y], color='blue')
    graph.set_xlabel("X: "+STR_X, fontsize=20, rotation=0, color='black')
    graph.set_ylabel("Y: "+STR_Y, fontsize=20, rotation=90, color='black')
    graph.set(xlim=(min(Data_DCT[STR_X]), max(Data_DCT[STR_X])), xticks=np.arange(min(Data_DCT[STR_X]), max(Data_DCT[STR_X]), (max(Data_DCT[STR_X])-min(Data_DCT[STR_X]))/10), 
              ylim=(min(Data_DCT[STR_Y]), max(Data_DCT[STR_Y])), yticks=np.arange(min(Data_DCT[STR_Y]), max(Data_DCT[STR_Y]), (max(Data_DCT[STR_Y])-min(Data_DCT[STR_Y]))/10))
    
    #PLOT EQUATION ---- Linear Regression Function can be changed here.
    intercept, slope = MSE_Calculus(Data_DCT[STR_X], Data_DCT[STR_Y])
    x_int = np.linspace(min(Data_DCT[STR_X]),max(Data_DCT[STR_X]),100)
    y_int = (slope*x_int)+intercept
    plt.plot(x_int,y_int, color='black', linewidth=4, label='MSE_Calculus')
    print("MSE_Calculus                 : Y = ",slope,"X + ",intercept)

    intercept, slope = Stochastic_Gradient_Descent(Data_DCT[STR_X], Data_DCT[STR_Y])
    x_int = np.linspace(min(Data_DCT[STR_X]),max(Data_DCT[STR_X]),100)
    y_int = (slope*x_int)+intercept
    plt.plot(x_int,y_int, color='red', linewidth=1, label='Stochastic_Gradient_Descent')
    print("Stochastic_Gradient_Descent  : Y = ",slope,"X + ",intercept)
    
    plt.legend()
    plt.show()

MakeGraph(STR_X,STR_Y)