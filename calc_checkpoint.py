import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os


num_nums = 10000 # number of numbers to use in the training data from each operation

learning_rate = 0.01 # how fast to learn
batch_size = 32 # number of training examples to use in each training step
epochs = 10_000 # number of times to run the training data through the model
epoch_save_freq = 1000 # how often to save the model
save_interval = 100000 # how often to save the model
#model_file = 'C:\Users\austi\vs_code\coursera\model.h5' # file to save the model to
optimizer = Adam(lr=learning_rate) # optimizer to use


#===================================================================================================
# Create traing data 
#===================================================================================================
def create_data():
    # create an array of 500 random numbers between -1000 and 1000 called x
    a = np.expand_dims(np.random.randint(-1000, 1000, num_nums),axis=1)
    # create an array of 500 random numbers between -1000 and 1000 called y
    b = np.expand_dims(np.random.randint(-1000, 1000, num_nums),axis=1)
    # create an array of 500 zeros called add_op_id representing the addition operation
    add_op_id = np.expand_dims(np.zeros(num_nums),axis=1)
    # create an array that is the sum of x and y called z
    c = a + b
    # combine arrays to get 500 rows of 3 columns of test data
    x_data = np.concatenate((a, b, add_op_id), axis=1)
    # save y data as the result of the addition
    y_data = c

    print(F"X data shape: {x_data.shape}, Y data shape: {y_data.shape}")

    # now use same x and y arrays but subract them and add a new operation id
    c_sub = a - b
    y_data = np.concatenate((y_data, c_sub), axis=0)
    sub_x_data = np.concatenate((a, b, np.expand_dims(np.ones(num_nums), axis=1)), axis=1)
    x_data = np.concatenate((x_data, sub_x_data), axis=0)

    print(F"X data shape: {x_data.shape}, Y data shape: {y_data.shape}")

    # now the same for multiplication
    c_mul = a * b
    y_data = np.concatenate((y_data, c_mul), axis=0)
    mul_x_data = np.concatenate((a, b, np.expand_dims(np.ones(num_nums)*2, axis=1)), axis=1)
    x_data = np.concatenate((x_data, mul_x_data), axis=0)

    print(F"X data shape: {x_data.shape}, Y data shape: {y_data.shape}")

    # now the same for division but check for divide by zero if y is zero do not add to data
    try:
        c_div = a / b
        y_data = np.concatenate((y_data, c_div), axis=0)
        div_x_data = np.concatenate((a, b, np.expand_dims(np.ones(num_nums)*3, axis=1)), axis=1)
        x_data = np.concatenate((x_data, div_x_data), axis=0)
    except ZeroDivisionError:
        #print("Can't divide by zero")
        pass

    print(F"X data shape: {x_data.shape}, Y data shape: {y_data.shape}")

    
    return x_data, y_data

#===================================================================================================
# Split the data into training and test sets
#===================================================================================================


# Split the data into training and test sets
def split_data(x_data, y_data):
    # split the data into training and test sets
    # 80% of the data will be used for training and 20% for testing
    train_size = int(len(x_data) * 0.8)
    test_size = len(x_data) - train_size
    train_x, test_x = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
    train_y, test_y = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])

    return train_x, train_y, test_x, test_y

#===================================================================================================
# Create the model
#===================================================================================================

def create_model(weights=None):
    # create the model
    # create a sequential model
    model = tf.keras.Sequential()
    # add a dense layer with 100 neurons and an input shape of 3, initialize with saved weights if provided
    model.add(tf.keras.layers.Dense(100, input_shape=(3,), weights=weights[0] if weights else None))
    # add a relu activation function
    model.add(tf.keras.layers.Activation('relu'))
    # add a dense layer with 100 neurons, initialize with saved weights if provided
    model.add(tf.keras.layers.Dense(100, weights=weights[1] if weights else None))
    # add a relu activation function
    model.add(tf.keras.layers.Activation('relu'))
    # add a dense layer with 100 neurons, initialize with saved weights if provided
    model.add(tf.keras.layers.Dense(100, weights=weights[2] if weights else None))
    # add a relu activation function
    model.add(tf.keras.layers.Activation('relu'))
    # add a dense layer with 1 neuron, initialize with saved weights if provided
    model.add(tf.keras.layers.Dense(1, weights=weights[3] if weights else None))

    # compile the model
    # use the adam optimizer
    # use mean squared error as the loss function
    # use accuracy as the metric
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    return model

#===================================================================================================
# Train the model
#===================================================================================================

def train_model(model, train_x, train_y, test_x, test_y):   
    # if the model file exists, load the weights and bias from the file
    # create a checkpoint to save the model weights and bias after each epoch
    checkpoint = tf.keras.callbacks.ModelCheckpoint('add_sub_mul_div_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq=save_interval)
    
    if os.path.exists('add_sub_mul_div_model.h5'):
        model.load_weights('add_sub_mul_div_model.h5')
    # train the model
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, validation_data=(test_x, test_y), callbacks=[checkpoint])

    # save the model to a file
    model.save('add_sub_mul_div_model.h5')

    return model


#===================================================================================================
# Predict the result
#===================================================================================================

# Predict the result
def predict_result(model, x_data, y_data):
    # predict the result
    # use the model to predict the result of the test data
    # print the predicted result
    # print the actual result
    # print the difference between the predicted and actual result
    # load the model from the file
    model = tf.keras.models.load_model('add_sub_mul_div_model.h5')
    predictions = model.predict(x_data)
    print(F"Predicted result: {predictions}")
    print(F"Actual result: {y_data}")
    print(F"Difference: {predictions - y_data}")

#===================================================================================================
# Main
#===================================================================================================

def main():
    # create the data
    x_data, y_data = create_data()
    # split the data into training and test sets
    train_x, train_y, test_x, test_y = split_data(x_data, y_data)
    # create the model
    model = create_model()
    # train the model
    model = train_model(model, train_x, train_y, test_x, test_y)
    # predict the result
    predict_result(model, test_x, test_y)

#===================================================================================================
# convert operator to integer
#===================================================================================================
def op_to_int(op):
    if op == '+':
        return 0
    elif op == '-':
        return 1
    elif op == '*':
        return 2
    elif op == '/':
        return 3
    else:
        sys.exit(F"Invalid operator: {op}")

#===================================================================================================
# use the model to predict the result
#===================================================================================================
def use_calc():
    # prompt the user to enter 2 numbers and an operator
    # use the model to predict the result
    # load the model from the file
    model = tf.keras.models.load_model('add_sub_mul_div_model.h5')
    x, op, y = input("Enter 2 numbers and an operator (num1 op num2): ").split(" ")
    # use the model to predict the result and give error if zero division
    try:
        predict_result(model, np.array([[int(x), int(y), op_to_int(op)]]).astype(np.float32), np.array([0]))
    except ZeroDivisionError:
        print("Error: Division by zero")

    # print result vs actual result
    print(F"Result: {x} {op} {y} = {predict_result(model, np.array([[int(x), int(y), op_to_int(op)]]), np.array([0]))}")

#===================================================================================================
# Run the main function
#===================================================================================================

# main() conditional to run main() or use_calc()
if len(sys.argv) == 1:
    main()
elif len(sys.argv) == 2:
    use_calc()

# conditional to run use_calc() or main()






#print 6 new empty lines
print('\n\n\n\n\n\n')