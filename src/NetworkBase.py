
import numpy as np
import tensorflow as tf

class NetworkBase:

    def __init__(self, summary_level, features, labels, mode, options):
        self.summary_level = summary_level
        self.fc_dropout_keep = tf.constant(options["fc_dropout_keep"], dtype=tf.float32)
        self.is_train = tf.constant(options["is_train"], dtype=tf.bool)
        self.x = features
        self.y = labels
        self.mode = mode

        # days = int(features.get_shape()[1])

        # rates = tf.reshape(self.x[:,:,:,0:1],[-1,days])
        # o1 = tf.one_hot(indices=tf.cast(tf.multiply(tf.subtract(float(1), rates), 100), tf.int32), depth=100, on_value=1.0, off_value=0.0, axis=-1)
        # self.image_summary('x1', o1, days, 100, 1)

        # if features.get_shape()[3] > 1:
        #     diffs = tf.reshape(self.x[:,:,:,1:2],[-1,days])
        #     o2 = tf.one_hot(indices=tf.cast(tf.add(float(50), tf.multiply(diffs, float(-100))), tf.int32), depth=100, on_value=1.0, off_value=0.0, axis=-1)
        #     self.image_summary('x2', o2, days, 100, 1)

    def exit_layer(self, name, x, dropout_keep):
        with tf.compat.v1.name_scope(name):
            fc1 = self.full_layer('fc1', x, 1024, True)

            fc1_drop = tf.nn.dropout(fc1, rate=1 - (dropout_keep))
            if self.summary_level >= 2:
                tf.compat.v1.summary.histogram('fc1_drop', fc1_drop)

            fc2 = self.full_layer('fc2', fc1_drop, 2, False)

            return fc2

    def conv_layer(self, name, in_layer, width, stride, out_dim, relu, use_batch_norm, padding='SAME'):
        with tf.compat.v1.name_scope(name):
            in_layer_shape = int(in_layer.get_shape()[3])

            initializer = tf.initializers.GlorotNormal()
            W = tf.Variable(lambda: initializer([width, 1, in_layer_shape, out_dim]))
            self.variable_summaries('W', W)
            # image_summary('W', W, 5, 1 out_dim)

            initializer = tf.compat.v1.constant_initializer(0.0)
            b = tf.Variable(lambda: initializer([out_dim]))
            self.variable_summaries(name, b)

            r = tf.nn.conv2d(input=in_layer, filters=W, strides=[1, stride, 1, 1], padding=padding, name=name) + b
            if self.summary_level >= 2:
                tf.compat.v1.summary.histogram('r', r)

            result = r

            if use_batch_norm:
                result = tf.keras.layers.BatchNormalization(center=True, scale=True, )(result)

            if relu:
                result = tf.nn.leaky_relu(result)
                if self.summary_level >= 2:
                    tf.compat.v1.summary.histogram('relu', result)

            return result

    def full_layer(self, name, in_layer, out_dim, relu):
        with tf.compat.v1.name_scope(name):
            in_layer_shape = int(np.prod(in_layer.get_shape()[1:]))

            initializer = tf.initializers.GlorotNormal()
            W = tf.Variable(initializer([in_layer_shape, out_dim]))
            self.variable_summaries('W', W)

            initializer = tf.compat.v1.constant_initializer(0.0)
            b = tf.Variable(initializer([out_dim]))
            self.variable_summaries('b', b)

            in_layer_flat = tf.reshape(in_layer, [-1, in_layer_shape])

            r = tf.matmul(in_layer_flat, W) + b
            if self.summary_level >= 2:
                tf.compat.v1.summary.histogram('r', r)

            if relu:
                r = tf.nn.leaky_relu(r)
                if self.summary_level >= 2:
                    tf.compat.v1.summary.histogram('relu', r)

            return r

    def max_pool_layer(self, name, x, width, stride, padding='SAME'):
        with tf.compat.v1.variable_scope(name):
            pool = tf.nn.max_pool2d(input=x, ksize=[1, width, 1, 1], strides=[1, stride, 1, 1], padding=padding, name=name)
            if self.summary_level >= 2:
                tf.compat.v1.summary.histogram('value', pool)
            return pool

    def avg_pool_layer(self, name, x, width, stride, padding='SAME'):
        with tf.compat.v1.variable_scope(name):
            pool = tf.nn.avg_pool2d(input=x, ksize=[1, width, 1, 1], strides=[1, stride, 1, 1], padding=padding, name=name)
            if self.summary_level >= 2:
                tf.compat.v1.summary.histogram('value', pool)
            return pool

    def variable_summaries(self, name, var):
        if self.summary_level >= 2:
            with tf.compat.v1.variable_scope(name):
                mean = tf.reduce_mean(input_tensor=var)
                tf.compat.v1.summary.scalar('mean', mean)
                with tf.compat.v1.variable_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
                tf.compat.v1.summary.scalar('stddev', stddev)
                tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
                tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
                tf.compat.v1.summary.histogram('histogram', var)

    def image_summary(self, name, V, width, height, dimensions):
        V = tf.reshape(V, (tf.shape(input=V)[0], width, height, dimensions))
        V = tf.transpose(a=V, perm=(0, 2, 1, 3))
        tf.compat.v1.summary.image(name, V, max_outputs=10)
