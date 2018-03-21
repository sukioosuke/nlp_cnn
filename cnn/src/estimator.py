#import data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=tf.nn.relu):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + np.log(np.abs(x_data + noise)) - x_data

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer(x_data, 1, 10)
predition = add_layer(layer1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for step in range(2000):
    session.run(train, feed_dict={xs:x_data, ys:y_data})
    if step % 50 == 0:
        try:
            ax.lines.remove(line[0])
        except Exception:
            pass
        predition_value = session.run(predition, feed_dict={xs: x_data})
        line = ax.plot(x_data, predition_value, 'r', lw=5)
        plt.pause(1)
        # print(session.run(loss, feed_dict={xs:x_data, ys:y_data}))
        # print(step, session.run(Weights1), session.run(Weights2), session.run(biases))
session.close()
