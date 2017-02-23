import tensorflow as tf
import numpy as np
from tqdm import tqdm
import midi_manipulation
from Utils import *
import os

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

        h = self.sample(tf.sigmoid(tf.matmul(self.x, self.W)+ self.bh))
        hSample = self.sample(tf.sigmoid(tf.matmul(xSample, self.W)+ self.bh))

        sizeBT = tf.cast(tf.shape(self.x)[0], tf.float32)
        wAdder = tf.multiply (lr/sizeBT, tf.subtract(tf.matmul(tf.transpose(self.x), h), tf.matmul(tf.transpose(xSample), hSample)))
        bvAdder = tf.multiply (lr/sizeBT, tf.reduce_sum(tf.subtract(self.x, xSample), 0, True))
        bhAdder = tf.multiply (lr/sizeBT, tf.reduce_sum(tf.subtract(h, hSample), 0, True))

        self.updt = [self.W.assign_add(wAdder), self.bv.assign_add(bvAdder), self.bh.assign_add(bhAdder)]

        self.saver = tf.train.Saver()


    def gibbsSample (self, k):
        def gibbsStep (count, k, xk):
            hk = self.sample(tf.sigmoid(tf.matmul(xk, self.W)+self.bh))
            xk = self.sample(tf.sigmoid(tf.matmul(hk, tf.transpose(self.W)) + self.bv))
            return count+1, k, xk
        ct = tf.constant(0)
        cond = lambda count, k, x: tf.less(count,k)
        [_,_, xSample] = tf.while_loop(cond, gibbsStep, [ct, tf.constant(k), self.x])
        xSample = tf.stop_gradient(xSample)
        return xSample
    def sample(self, probs):
        return tf.floor(probs+tf.random_uniform(tf.shape(probs),0,1))

    def train (self, trainPath= "Data/", savePath="Data/Models/model.ckpt"):
        songs = compileCompositions(trainPath)
        with tf.Session () as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for cycle in tqdm(range(self.cycles)):
                for song in songs:
                    song = np.array(song)
                    song = song[:int (np.floor(song.shape[0]/self.numTimesteps)*self.numTimesteps)]
                    song = np.reshape(song, [song.shape[0]/self.numTimesteps, song.shape[1]*self.numTimesteps])
                    for i in range(1,len(song), self.batch):
                        trX = song[i:i+self.batch]
                        sess.run(self.updt, feed_dict={self.x: trX})
            path = self.saver.save(sess, savePath)
            print("Saved Model to " + savePath)

    def generateComposition (self, path="Data/Models/model.ckpt"):
        with tf.Session () as sess:
            self.saver.restore(sess, path)
            print("Restored Model from " + path)

            sample = self.gibbsSample(1).eval(session=sess, feed_dict={x:np.zeros((10,visible))})
            for i in range (sample.shape[0]):
                if not any(sample[i,:]):
                    continue

                s = np.reshape(sample[i,:], (numTimesteps, 2*range))
                midi_manipulation.noteStateMatrixToMidi(S, (str(filename)).format(i))
NN = NeuralNet()
NN.train(trainPath="Data/")
