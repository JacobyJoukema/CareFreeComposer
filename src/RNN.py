import tensorflow as tf
import numpy as np
from tqdm import tqdm
import midi_manipulation
import Utils

class NeuralNet:
    def __init__ (self):
        lowest = midi_manipulation.lowerBound
        highest = midi_manipulation.uppderBound
        nRange = highest-lowest

        self.numTimesteps = 20
        visible =2*nRange*numTimesteps
        hidden = 50

        self.cycles = 200
        self.batch = 100

        lr = tf.constant (.005, tf.float32)

        x = tf.placeholder(tf.float32, [None, visible], name = "x")
        W = tf.Variable(tf.random_normal([visible, hidden], .01), name="W")
        bh = tf.Variable(tf.zeros([1,hidden],tf.float32, name="bh"))
        bv = tf.Variable(tf.zeros([1,visible], tf.float32, name="bv"))

        xSample = gibbs_sample(1)

        h = sample(tf.sigmoid(tf.matmul(x,W)+bh))
        hSample = sample(tf.sigmoid(tf.matmul(xSample, W)+bh))

        sizeBT = tf.cast(tf.shape(x)[0], tf.float32)
        wAdder = tf.mul(lr/sizeBT, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(xSample), hSample)))
        bvAdder = tf.mul(lr/sizeBT, tf.reduce_sum(tf.sub(x, xSample), 0, True))
        bhAdder = tf.mul(lr/sizeBT, tf.reduce_sum(tf.sub(h, hSample), 0, True))

        updt = [W.assign_add(WAdder), bv.assign_add(bvAdder), bh.assign_add(bhAdder)]

        self.saver = tf.train.Saver()

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

            sample = gibbs_sample(1).eval(session=sess, feed_dict={x:np.zeros((10,visible))})
            for i in range (sample.shape[0]):
                if not any(sample[i,:]):
                    continue

                s = np.reshape(sample[i,:], (numTimesteps, 2*range))
                midi_manipulation.noteStateMatrixToMidi(S, (str(filename)).format(i))
NN = NeuralNet()
NN.train("Data")
