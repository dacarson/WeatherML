import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# Large enough to use GPU
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])
c = tf.matmul(a, b)
print("Done")
