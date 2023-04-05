import numpy as np
import csv


def read_in_data(file):
    data = []
    with open(file, newline='') as readFile:
        #   make new row whenever there's a line break (aka changes from one image to another (starts with 0/1))
        csvreader = csv.reader(readFile, delimiter=',')
        for row in csvreader:
            tempArray = []
            for value in row:
                tempArray.append(int(value))
            data.append(tempArray)
    return np.array(data)


train_data = read_in_data("data/mnist_train_0_1.csv")
test_data = read_in_data("data/mnist_test_0_1.csv")


def prepare_data(row):
    class_value = row[0]
    array_class_value = np.array([class_value])

    row = np.array([row[1:]])

    return array_class_value, row


# defining activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define neural network
def neural_network(inputs, input_weights, hidden_weights, hidden_bias, output_bias):
    # sum the 784 weights and inputs
    input_sum = np.dot(inputs, input_weights)

    # hidden layer
    input_g = sigmoid(input_sum) # + hidden_bias.sum()     # will apply to each element in the array bc python is goated, will return array

    hidden_sum = np.dot(input_g, hidden_weights)
    output = sigmoid(hidden_sum + output_bias)

    return output, input_g


# backpropation/updating weights
def backprop(inputs, class_value, output, input_g, hidden_weights, output_bias, hidden_bias):

    # Output Layer
    output_delta = (class_value - output) * output * (1 - output)

    # Interior Layer
    hidden_delta = np.dot(hidden_weights, output_delta) * (input_g * (1 - input_g)).T

    # weight updates
        # interior layer
    updated_input_weights = input_weights + (alpha * hidden_delta * inputs).T  # np.dot(inputs.T, hidden_error)
    updated_hidden_bias = hidden_bias + (alpha * hidden_delta).T

        # output layer  # hidden weights needs to stay as 28, 1
    updated_hidden_weights = hidden_weights + alpha * output_delta * output  # np.dot(input_g.T, output_delta) * inputs
    updated_output_bias = output_bias + alpha * output_delta

    return updated_input_weights, updated_hidden_weights, updated_output_bias, updated_hidden_bias


output_bias = 1
alpha = .01

# setting up size of network
input_size = len(train_data[0]) - 1  # was len(data_example[0])
hidden_size = 28
output_size = 1
hidden_bias = np.empty([input_size, 1])
hidden_bias.fill(1)

# Initializing random weights
input_weights = np.random.randn(input_size, hidden_size)
hidden_weights = np.array(np.random.randn(hidden_size, output_size))
hidden_values = np.array([range(hidden_size)])

# Training
max_epoch_num = 100

num_total = len(train_data)
for i in range(max_epoch_num):
    num_correct = 0
    for index in range(len(train_data)):
        class_value_example, data_example = prepare_data(train_data[index])
        class_value_score = int(class_value_example)

        # feedforward pass
        output, input_g = neural_network(data_example, input_weights, hidden_weights, hidden_bias, output_bias)

        if class_value_score == 0:
            if output < .5:
                num_correct = num_correct + 1

        if class_value_score == 1:
            if output >= .5:
                num_correct = num_correct + 1

        # weight updates/backprop
        updated_input_weights, updated_hidden_weights, updated_output_bias, updated_hidden_bias = backprop(data_example, class_value_example, output, input_g, hidden_weights, output_bias, hidden_bias)

        # weight/bias updates
        input_weights = updated_input_weights
        hidden_weights = updated_hidden_weights
        output_bias = updated_output_bias
        hidden_bias = updated_hidden_bias

    accuracy_percentage = (num_correct / num_total) * 100
    print("Accuracy of training set " + str(i) + ": " + str(accuracy_percentage) + "%")
    if accuracy_percentage > 99.5:
        break


# test
num_test_total = len(test_data)
num_test_correct = 0
for index in range(len(test_data)):
    class_value_example, data_example = prepare_data(test_data[index])
    class_value_score = int(class_value_example)

    # feedforward pass
    output, input_g = neural_network(data_example, input_weights, hidden_weights, hidden_bias, output_bias)

    if class_value_score == 0:
        if output < .5:
            num_test_correct = num_test_correct + 1

    if class_value_score == 1:
        if output >= .5:
            num_test_correct = num_test_correct + 1

accuracy_percentage = (num_test_correct / num_test_total) * 100
print("Accuracy of test: " + str(accuracy_percentage) + "%")