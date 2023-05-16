from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def parse_boolean(value):
    value = value.lower()

    if value in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value in ['false', 'f', 'no', 'n', '0']:
        return False

    return False

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", '--path',type = str,  required=True, help = "Path to the CSV file of observed data including the name of the CSV file" )
parser.add_argument("-s", '--plot',type = parse_boolean, default = False , help = "show the plot")
args = parser.parse_args()


CSVFile = args.path
x =[]
y = []

with open(CSVFile, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        y.append(float(row[1])/100)    #Voltage --> not mv
        x.append(1/float(row[0]))  #Distance
x = np.array(x).reshape((-1,1))
# x = x.reshape((-1, 1))
y = np.array(y).reshape((-1,1))
# y = y.reshape((-1, 1))

regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y, y_predicted)
print('For y = Voltage, x = Distance')
print('Slope: ', regression_model.coef_)
print('Intercept: ', regression_model.intercept_)
print('Root Mean Square:', rmse)


print('------for  y = Distance------')
print(f'Slope: {1/regression_model.coef_}')
print(f'Intercept: {regression_model.intercept_/regression_model.coef_}')
if args.plot:
    plt.scatter(x, y, s =10)
    plt.xlabel('x(Distance)')
    plt.ylabel('y(Voltage (V))')

    plt.plot(x, y_predicted, color = 'g')
    plt.show()
