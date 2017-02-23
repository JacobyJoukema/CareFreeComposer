import tensorflow as tf
import numpy as np
from tqdm import tqdm
import midi_manipulation
from tensorflow.python.ops import control_flow_ops
import Utils

class NeuralNet:
    def __init__ (self):
        lowest = midi_manipulation.lowerBound
        highest = midi_manipulation.upperBound
        nRange = highest-lowest

        self.numTimesteps = 20
        visible =2*nRange*self.numTimesteps
        hidden = 50

        self.cycles = 200
        self.batch = 100

        lr = tf.constant (.005, tf.float32)

        self.x = tf.placeholder(tf.float32, [None, visible], name = "x")
        self.W = tf.Variable(tf.random_normal([visible, hidden], .01), name="W")
        self.bh = tf.Variable(tf.zeros([1,hidden],tf.float32, name="bh"))
        self.bv = tf.Variable(tf.zeros([1,visible], tf.float32, name="bv"))

        xSample = self.gibbsSample(1)

        h = sample(tf.sigmoid(tf.matmul(x,W)+bh))
        hSample = sample(tf.sigmoid(tf.matmul(xSample, W)+bh))

        sizeBT = tf.cast(tf.shape(x)[0], tf.float32)
        wAdder = tf.mul(lr/sizeBT, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(xSample), hSample)))
        bvAdder = tf.mul(lr/sizeBT, tf.reduce_sum(tf.sub(x, xSample), 0, True))
        bhAdder = tf.mul(lr/sizeBT, tf.reduce_sum(tf.sub(h, hSample), 0, True))

        updt = [W.assign_add(WAdder), bv.assign_add(bvAdder), bh.assign_add(bhAdder)]

        self.saver = tf.train.Saver()

    #Ref https://github.com/llSourcell/Music_Generator_Demo
    def gibbsSample (self, k):
        def gibbsStep (count, k, xk):
            hk = self.sample(tf.sigmoid(tf.matmul(xk, self.W)+self.bh))
            xk = self.sample(tf.sigmoid(tf.matmul(hk, tf.transpose(self.W)) + self.bv))
            return count+1, k, xk
        ct = tf.constant(0)
        count = tf.constant(0)
        cond = lambda cond: tf.less(count,k)
        body = gibbsStep
        [_,_, xSample] = tf.while_loop(cond, body, [ct, tf.constant(k), self.x])
        xSample = tf.stop_gradient(xSample)
        return xSample
    def sample(self, probs):
        return tf.floor(probs+tf.random_uniform(tf.shape(probs),0,1))

    def train (self, path="Data/Models/model.ckpt"):
        songs = compileCompositions(path)
        with tf.Session () as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            for cycle in tqdm(range(cycles)):
                for song in songs:
                    song = np.array(song)

                    for i in range(1,len(song), batch):
                        trX = song[i:i+batch]
                        sess.run(updt, feed_dict={x: trX})
            path = saver.save(sess, path)
            print("Saved Model to " + path)

    def generateComposition (self, path="Data/Models/model.ckpt"):
        with tf.Session () as sess:
            saver.restore(sess, path)
            print("Restored Model from " + path)

            sample = self.gibbsSample(1).eval(session=sess, feed_dict={x:np.zeros((10,visible))})
            for i in range (sample.shape[0]):
                if not any(sample[i,:]):
                    continue

                s = np.reshape(sample[i,:], (numTimesteps, 2*range))
                midi_manipulation.noteStateMatrixToMidi(S, (str(filename)).format(i))
NN = NeuralNet()
NN.train("Data")
