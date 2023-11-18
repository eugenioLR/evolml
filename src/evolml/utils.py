from __future__ import annotations
import numpy as np

def restart_model(trained_model):
    return type(trained_model)(**trained_model.get_params())

def sequence_encoding(time_series, nsteps):
    sequence_vec = np.empty((time_series.shape[0]-nsteps, nsteps))
    for idx, val in enumerate(time_series[:-nsteps]):
        sequence_vec[idx] = time_series[idx:idx+nsteps]
    return sequence_vec

if __name__ == "__main__":
    ex1 = np.array([0,1,2,3,4,5,4,3,4,5,6,7,8,9,8,7,6,5,4,3,2,3,4,3,2,1,0])
    seq_4 = sequence_encoding(ex1, 4)
    seq_7 = sequence_encoding(ex1, 7)

    print(ex1)
    print(ex1.shape)
    print(seq_4)
    print(seq_4.shape)
    print(seq_7)
    print(seq_7.shape)