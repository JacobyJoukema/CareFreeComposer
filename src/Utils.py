import glob
from tqdm import tqdm
import numpy as np
import midi_manipulation

def compileCompositions(directory):
    out = []
    files = glob.glob('{}/*.mid'.format(directory))

    for phial in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(phial))
            out.append(song)
        except Exception as e:
            raise e

    return out
