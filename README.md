# ML-Calculator 
Welcome to my python script, a 221 line exercise in floccinaucinihilipilification (yes, that is a noun in the english language. 
and no it could not be better suited for anything else.)
Wherein I attempt to replicate a primitive calculator using sophisticated computer hardware, because there is clearly no greater pursuit.

She was made in Alberta, and inspired by the TI 89 Titanium, (Texas Instruments)
So naturally we will refer to this one as the AI 0.89 Arsenic, (Alberta Instruments, arsenic  to keep with the matal theme
but eluding to it efficacy, because if you had to do any serious math, neither of these would be good for your health. Also, AI little play on words there ;), please,
hold your applause.) Alright let's dive in. 

This code uses the TensorFlow library to train a neural network to perform basic arithmetic operations. 
The network is trained using a set of random numbers and the corresponding results of four different operations: addition,
subtraction, multiplication, and division.

To start, the code imports the necessary libraries, 
```
import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
```

It also defines several parameters that will be used during training, such as the number of numbers to use in the training data, 
the learning rate, the batch size, and the number of epochs.

Next, the code defines a function called ```create_data()``` that is used to generate the training and validation data. 
This function creates two arrays of random numbers, called a and b, each with the specified number of elements.
It then uses these arrays to compute the results of the four different operations, 
and combines the results with the corresponding operation IDs in a single array called x_data. 
The results of the operations are stored in a separate array called y_data.

After the data has been generated, 
the code splits it into training and validation sets using a specified ratio. 
The training data is then used to train the neural network using the Adam optimizer and the specified learning rate and batch size. 
The network is trained for the specified number of epochs, and its performance is evaluated on the validation set after each epoch.

After training is complete, the code saves the trained model to a file, and prints out its final performance on the validation set. 
This trained model can then be used to make predictions on new data.

Overall, this code provides a simple example of how to use TensorFlow to train a neural network for performing basic arithmetic operations. 
By adjusting the hyperparameters and adding more data, it could be further improved and applied to more complex tasks. 


# Usage

# training

incase your calculator has been damaged and no longer works, nor do any other calculators for that matter, and you need to do
some exceptionally simple math, and forget how to use the python math functions, you can use mine :), thank me later!


The 2 main functions of the code are main() and use_calc()
main is used to train the model

```python calculator.py```
in cmd prompt will invoke training of the model, given len(sys.argv[1]) == 1

you first need specify the number of data points you wish to collect within the program; via this line of code.
```
num_nums = 10_000 # number of numbers to use in the training data from each operation
```

You can then adjust these hyper-parameters as you see fit.

```
learning_rate = 0.01 # how fast to learn
batch_size = 32 # number of training examples to use in each training step
epochs = 10_000 # number of times to run the training data through the model
save_interval = 100 # how often to save the model
optimizer = Adam(lr=learning_rate) # optimizer to use
```

Now you are off to the races, 

** ALTERNATIVELY - if you do not particularly fancy training this model, (but fancy using the ensuing operation? weird.) 
you can use the h5 attached. 1 like = 1 h5.

# using

To invoke usage of said calculator
you need pass a single cmd arg, anything will do.
but if you want to stay on theme, I personally recommend using
```python calculator.py -- use_AI_0.89_Arsenic```
but any amount of sys.argv length > 1 will have the same effect.

You will then be prompted to input your 2 numbers and the operation, (num op num) ie. 2 + 2. (the + gets changed the its identifier than it was trained with that 
arbitrarilly represents '+' operation, I didn't cheat, okay.

should all go as intended (granted I have any level of competancy) you should arrive at output = 4

Yes, I did in fact spend 1.12 days programming an ML model so that you can add 2 and 2, thanks for asking... What's that?  The most rewarding part, you ask?
Writing this README, by several orders of magnitude.

Thank you for wasting your time with me and getting to know a little bit about how to wildly over-complicate a task that was solved circa 1961 shout out ANITA!
(side note:
not sure how we got a nuclear bomb 16 years prior to a calculator??? Priorities I suppose...)
anyways...


If you would like to purchase the rights to my calculator it will be for sale on my etsy store for 1 million dollars. Use coupon code "Thanks" to add an 18% gratuity 
at check-out! If I don't sell atleast 9 I'm going to pursue a career in satire instead, so tell a friend! Buy 1 get 1, 1% off

Cheers!!!










