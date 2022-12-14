import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


num_nums = 5000 # number of numbers to use in the training data from each operation
epochs = 10000 # number of times to run the training data through the model
learning_rate = 0.01 # how fast to learn
batch_size = 100 # number of training examples to use in each training step
save_interval = 100 # how often to save the model


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

def create_model():
    # create the model
    # create a sequential model
    model = tf.keras.Sequential()
    # add a dense layer with 100 neurons and an input shape of 3
    model.add(tf.keras.layers.Dense(100, input_shape=(3,)))
    # add a relu activation function
    model.add(tf.keras.layers.Activation('relu'))
    # add a dense layer with 100 neurons
    model.add(tf.keras.layers.Dense(100))
    # add a relu activation function
    model.add(tf.keras.layers.Activation('relu'))
    # add a dense layer with 1 neuron
    model.add(tf.keras.layers.Dense(1))

    # compile the model
    # use mean squared error as the loss function
    # use adam as the optimizer
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))


    return model

#===================================================================================================
# Train the model
#===================================================================================================

def train_model(model, train_x, train_y, test_x, test_y):    
    # train the model
    # train the model for 100 epochs
    # use 20% of the data for validation
    # use the test data as the validation data
    # print the loss and accuracy after each epoch
    # save the model to a file
    checkpoint = ModelCheckpoint(filepath='/path/to/save/model.h5', save_freq=10)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, validation_data=(test_x, test_y), callbacks=[checkpoint])
    model.save('add_sub_mul_div_model.h5')

    return model

#===================================================================================================
# Test the model
#===================================================================================================

# Test the model
def test_model(model, test_x, test_y):
    # test the model
    # use the test data to test the model
    # print the loss and accuracy
    # load the model from the file
    model = tf.keras.models.load_model('add_sub_mul_div_model.h5')
    loss = model.evaluate(test_x, test_y)

   

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

    # test the model
    test_model(model, test_x, test_y)

    # predict the result
    predict_result(model, test_x, test_y)

#===================================================================================================
# Run the main function
#===================================================================================================

if __name__ == '__main__':
    main()






#print 6 new empty lines
print('\n\n\n\n\n\n')