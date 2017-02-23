import tensorflow as tf
import numpy as np
from tqdm import tqdm
import midi_manipulation

lowest = midi_manipulation.lowerBound
highest = midi_manipulation.uppderBound
nRange = highest-lowest

numTimesteps = 20
visible =2*nRange*numTimesteps
hidden = 50

cycles = 200
samples = 6
