import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True



with open("salary.csv") as fp:
    Boston_DCT = {}
    data_string = ""
    for i, line in enumerate(fp):
        if i>21:
            if i%2==0:
                data_string+=line
            else:
                counter = counter + 1
                data_string+=line
                string_list = data_string.split()
                float_list = []
                for i in range(len(string_list)):
                    float_list.append(float(string_list[i]))
                
                Boston_DCT["NUMBER"] = counter
                Boston_DCT["CRIM"] = float_list[0]
                Boston_DCT["ZN"] = float_list[1]
                Boston_DCT["INDUS"] = float_list[2]
                Boston_DCT["CHAS"] = float_list[3]
                Boston_DCT["NOX"] = float_list[4]
                Boston_DCT["RM"] = float_list[5]
                Boston_DCT["AGE"] = float_list[6]
                Boston_DCT["DIS"] = float_list[7]
                Boston_DCT["RAD"] = float_list[8]
                Boston_DCT["TAX"] = float_list[9]
                Boston_DCT["PTRATIO"] = float_list[10]
                Boston_DCT["B"] = float_list[11]
                Boston_DCT["LSTAT"] = float_list[12]
                Boston_DCT["MEDV"] = float_list[13]
                
                Boston_list.append(Boston_DCT)
                
                data_string=""
                Boston_DCT = {}
    fp.close()
def ARR_DCT(name):
    C_ARR = []
    for i in range(len(Boston_list)):
        C_ARR.append(Boston_list[i][name])
    return np.array(C_ARR).reshape(-1,1)
ARR_X,ARR_Y,ARR_Z,CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV = ARR_DCT(STR_X),ARR_DCT(STR_Y),ARR_DCT(STR_Z),ARR_DCT("CRIM"),ARR_DCT("ZN"),ARR_DCT("INDUS"),ARR_DCT("CHAS"),ARR_DCT("NOX"),ARR_DCT("RM"),ARR_DCT("AGE"),ARR_DCT("DIS"),ARR_DCT("RAD"),ARR_DCT("TAX"),ARR_DCT("PTRATIO"),ARR_DCT("B"),ARR_DCT("LSTAT"),ARR_DCT("MEDV")
def Lin_Reg(AuX, AuY):
    model = LinearRegression().fit(AuX,AuY)
    r_sq = model.score(AuX, AuY)
    #print('Coef of determination:', r_sq)
    #print('intercept:', model.intercept_[0])
    #print('slope:', model.coef_[0][0])
    #y_pred = model.predict(A_X)
    #print('Predicted response:', y_pred, sep='\n')
    return([r_sq, model.intercept_[0], model.coef_[0][0]])   
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
def MakeGraph(Type):
    fig, graph = plt.subplots()
    graph.scatter(x=ARR_X, y=ARR_Y, color='blue')
    graph.set_xlabel("X: "+STR_X, fontsize=20, rotation=0, color='red')
    graph.set_ylabel("Y: "+STR_Y, fontsize=20, rotation=0, color='red')
    graph.set(xlim=(min(ARR_X), max(ARR_X)), xticks=np.arange(min(ARR_X), max(ARR_X), (max(ARR_X)-min(ARR_X))/10), 
           ylim=(min(ARR_Y), max(ARR_Y)), yticks=np.arange(min(ARR_Y), max(ARR_Y), (max(ARR_Y)-min(ARR_Y))/10))
        
    Slope_Int = Lin_Reg(ARR_X,ARR_Y)
    x_int = np.linspace(min(ARR_X),max(ARR_X),100)
    y_int = (Slope_Int[2]*x_int)+Slope_Int[1]
    graph.plot(x_int,y_int)
        
    print("X-Y PLOT:")
    print("Coef of Determination = ", LinReg_DCT[STR_X][STR_Y][0])
    print("Y - Intercept = ", LinReg_DCT[STR_X][STR_Y][1])
    print("Slope = ", LinReg_DCT[STR_X][STR_Y][2])
    plt.show()
MakeGraph(Dimension)