import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)

plt.plot(house_size, house_price, 'bx')
plt.xlabel("Size")
plt.ylabel("Price")

plt.show()

#Noramlize the array of data first, this is also referred to as scaling


def normalize(array):
    return (array-array.mean())/array.std()

num_train_samples = math.floor(num_house*0.7)

train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

tf_house_size = tf.placeholder("float",name="house_size")
tf_house_price = tf.placeholder("float",name="house_price")

tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

#Now that our data is ready, we can focus on the prediction model.
#Here's a simple model that predicts the prices based on the house size and two other variables i.e. size_factor and price_offset
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

#After defining the modeel our aim is to reduce the cost so that our model finds a best fit in to the data and predicts accurately.
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_house_price,2))/(2*num_train_samples)

#Now we define a learning rate for our model which depends on Gradient descent to find the best fit in to the data.
learning_rate = 0.1

#Now finally use tensorflow's gradient descent optimizer to minimize the cost of the predictor
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):
        for (x,y) in zip(train_house_size_norm,train_house_price_norm):
            sess.run(optimizer,feed_dict={tf_house_size:x,tf_house_price:y})

            if (iteration + 1) % display_every == 0 :
                c = sess.run(tf_cost,feed_dict={tf_house_size: train_house_size_norm,tf_house_price: train_house_price_norm})
                print("iteration #:", '%04d' % (iteration+1), "cost=","{:9f}".format(c),\
                "size_factor=", sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))
    print("Optimization finished")
    training_cost = sess.run(tf_cost,feed_dict={tf_house_size: train_house_size_norm,tf_house_price: train_house_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
