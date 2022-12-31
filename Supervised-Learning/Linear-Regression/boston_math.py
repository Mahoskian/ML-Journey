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
import timeit
import time
import multiprocessing
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True


Boston_list = []
Data_Name_List = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
Boston_Data_DCT = {}

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
# Extract the data for each feature into a separate array
for i in range(len(Data_Name_List)):
    temp_arr = []
    for j in range(len(Boston_list)):
        temp_arr.append(Boston_list[j][Data_Name_List[i]])
    #print(Data_Name_List[i], type(temp_arr[0]))
    Boston_Data_DCT[Data_Name_List[i]] = np.array(temp_arr)

#find min and max of the numpy arrays
def min(arr):
    min = 1000.0
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]
    min = math.floor(min)
    return min         
def max(arr):
    max = -100.0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
    max = math.ceil(max)
    return max

def Matrix_Time_LR(X, y):
# Define the linear regression function(s)
    def Matrix_linear_regression(X, y):
    # Add a column of ones to X
        X = np.column_stack((np.ones(len(X)), X))
        # Convert X and y to matrices
        X = np.matrix(X)
        y = np.matrix(y).T
        # Calculate the weight matrix using the matrix inverse method
        weight_matrix = np.linalg.inv(X.T @ X) @ X.T @ y
        # Extract the weight values from the weight matrix
        intercept = weight_matrix[0,0]
        slope = weight_matrix[1,0]
        return intercept, slope
    execution_time = timeit.timeit(lambda: Matrix_linear_regression(X, y), number=1)
    return execution_time, Matrix_linear_regression(X, y)

def Calc_Time_LR(X,y):
    def Calculus_linear_regression(X, y):
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
    execution_time = timeit.timeit(lambda: Calculus_linear_regression(X, y), number=1)
    return execution_time, Calculus_linear_regression(X, y)

plt.scatter(Boston_Data_DCT['MEDV'], Boston_Data_DCT['LSTAT'])
CalcTime, (intercept, slope) = Calc_Time_LR(Boston_Data_DCT['MEDV'], Boston_Data_DCT['LSTAT'])
MatrixTime, (intercept, slope) = Matrix_Time_LR(Boston_Data_DCT['MEDV'], Boston_Data_DCT['LSTAT'])
x_int = np.linspace(min(Boston_Data_DCT['MEDV']),max(Boston_Data_DCT['MEDV']),100)
y_int = (slope*x_int)+intercept
plt.plot(x_int,y_int, color='red')
plt.show()


elapsed_time = 0
for i in range(len(Data_Name_List)):
    for j in range(len(Data_Name_List)):
        if Data_Name_List[i] != Data_Name_List[j]:
            CalcTime, (intercept, slope) = Calc_Time_LR(Boston_Data_DCT[Data_Name_List[i]], Boston_Data_DCT[Data_Name_List[j]])
            elapsed_time = elapsed_time + CalcTime
            #print(Data_Name_List[i], "+", Data_Name_List[j], "= CalcTime: ", CalcTime, " Intercept: ", intercept, " Slope: ", slope)
  
print(f"Elapsed CALCULUS time: {elapsed_time:.6f} seconds")
elapsed_time = 0
for i in range(len(Data_Name_List)):
    for j in range(len(Data_Name_List)):
        if Data_Name_List[i] != Data_Name_List[j]:
            MatrixTime, (intercept, slope) = Matrix_Time_LR(Boston_Data_DCT[Data_Name_List[i]], Boston_Data_DCT[Data_Name_List[j]])
            elapsed_time = elapsed_time + MatrixTime
            #print(Data_Name_List[i], "+", Data_Name_List[j], "= MatrixTime: ", MatrixTime, " Intercept: ", intercept, " Slope: ", slope)
print(f"Elapsed MATRIX time: {elapsed_time:.6f} seconds")