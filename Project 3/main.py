import matplotlib.pyplot as plt
import numpy as np
import pandas as pa


def read_column(path, observation_vector, remained_data):
    temp = data.head(335)[path].to_list()
    for i in range(318):
        observation_vector[i] = temp[i]

    for i in range(17):
        remained_data[i] = temp[i + 318]
    return temp


# make Design matrix
def make_x(num):
    if (num == 1):
        A = np.column_stack((np.ones(318), np.arange(318)))
    else:
        A = np.column_stack((np.ones(318), np.arange(318), np.square(np.arange(318))))
    return A


# obtain parameter vector and then estimate observation vector
def estimate(x, observation_vector, parameter_vector, num):
    transpose_x = np.transpose(x)
    transpose_xdot_x = np.dot(transpose_x, x);
    transpose_xdot_observation_vector = np.dot(transpose_x, observation_vector)
    temp = np.linalg.solve(transpose_xdot_x, transpose_xdot_observation_vector)
    for i in range(len(parameter_vector)):
        parameter_vector[i] = temp[i]
    estimate_vector = np.zeros(318);
    for i in range(318):
        if (num == 1):
            estimate_vector[i] = parameter_vector[0] + parameter_vector[1] * i
        else:
            estimate_vector[i] = parameter_vector[0] + parameter_vector[1] * (i) + parameter_vector[2] * (i * i)
    return estimate_vector


# show real & estimated value and the error
def show(observation_vector, estimate_vector, remained_data, parameter_vector, estimate_remained_data, num):
    for i in range(17):
        if (num == 1):
            estimate_remained_data[i] = parameter_vector[0] + parameter_vector[1] * (i + 318)
        else:
            estimate_remained_data[i] = parameter_vector[0] + parameter_vector[1] * (i + 318) + parameter_vector[2] * (
                    (i + 318) * (i + 318))
    if (num == 1):
        print('\ncompare some remained  data  with their estimates in Linear Regression\n')
    else:
        print('\ncompare some remained data with their estimates in  Quadratic Regression\n')
    test = [observation_vector[300], observation_vector[310], remained_data[0], remained_data[8], remained_data[16]]
    estimate_test = [estimate_vector[300], estimate_vector[310], estimate_remained_data[0], estimate_remained_data[8],
                     estimate_remained_data[16]]
    for i in range(5):
        print('Real value: ', test[i])
        print('Estimated value: ', estimate_test[i])
        print('Error: ', estimate_test[i] - test[i])
        print('\n')


# 1 for Linear Regression and 2 for Quadratic Regression
num = 2
data = pa.read_csv('covid_cases.csv')
observation_vector = np.zeros(318)
remained_data = np.zeros(17)
estimate_remained_data = np.zeros(17)
temp = read_column('World', observation_vector, remained_data)
input = np.zeros(335)
for i in range(335):
    input[i] = i
x = np.array(make_x(num))
parameter_vector = np.zeros((num + 1))
estimate_vector = estimate(x, observation_vector, parameter_vector, num)
show(observation_vector, estimate_vector, remained_data, parameter_vector, estimate_remained_data, num)
estimate_vector = np.concatenate((estimate_vector, estimate_remained_data))
plt.plot(input, estimate_vector, color='blue')
plt.plot(input, temp, color='red')
x_ = [300, 310, 318, 326, 334]
estimate_value = [estimate_vector[300], estimate_vector[310], estimate_vector[318], estimate_vector[326],
                  estimate_vector[334]]
plt.scatter(x_, estimate_value, color='green')
plt.legend(["Estimated Polynomial", "Actual Values", "Test Values"], loc="upper left")
plt.show()
