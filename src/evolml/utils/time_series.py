from __future__ import annotations
import numpy as np


def sequence_encoding(time_series, nsteps):
    sequence_vec = np.empty((time_series.shape[0] - nsteps, nsteps))
    for idx, val in enumerate(time_series[:-nsteps]):
        sequence_vec[idx] = time_series[idx : idx + nsteps]
    return sequence_vec


def apply_lag_time(X, y, lag):
    return X[:-lag], y[lag:]


def apply_lead_time(X, y, lead):
    return sequence_encoding(X, lead), y[lead:]


# if __name__ == "__main__":
#     ex1 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 3, 2, 1, 0])
#     seq_4 = sequence_encoding(ex1, 4)
#     seq_7 = sequence_encoding(ex1, 7)

#     print("Sequence encoding")
#     print(ex1)
#     print(ex1.shape)
#     print(seq_4)
#     print(seq_4.shape)
#     print(seq_7)
#     print(seq_7.shape)

#     X = np.linspace(0, 1, 21)
#     y = np.arange(20)
#     X_lagged, y_lagged = apply_lag_time(X, y, 2)

#     print("Lag time")
#     print(X_lagged)
#     print(y_lagged)

#     X_lead, y_lead = apply_lead_time(X, y, 2)

#     print("Lead time")
#     print(X_lead)
#     print(y_lead)

#     X_ll, y_ll = apply_lag_time(*apply_lead_time(X, y, 2), 2)

#     print("Lead time")
#     print(X_ll)
#     print(y_ll)