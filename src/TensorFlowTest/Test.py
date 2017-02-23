import tensorflow as tf

x1 = tf.constant([5])
x2 = tf.constant([6])

res = tf.matmul(x1,x2)

print (res)
