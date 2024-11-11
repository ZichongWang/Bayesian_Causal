import numpy as np

# Conversion of df.m
def df(a):
    """Derivative of function f."""
    return 1 / (1 + np.exp(a))

# Conversion of f.m
def f(a):
    """Function f."""
    return -np.log(1 + np.exp(a))

# Conversion of Loss.m
def Loss(y, qBD, qLS, qLF, alLS, alLF, w, local, delta):
    """
    Loss function calculation.
    w is a weight vector in the order:
    [w0; weps; w0BD; w0LS; w0LF; wLSBD; wLFBD; wBDy; wLSy; wLFy; weBD; weLS; weLF; waLS; waLF]
    """
    # Log-transform y to avoid issues with very small values
    y = np.log(1e-6 + y)
    f_qBD = np.minimum(qBD + 1e-6, 1 - (1e-6))
    f_qLS = np.minimum(qLS + 1e-6, 1 - (1e-6))
    f_qLF = np.minimum(qLF + 1e-6, 1 - (1e-6))

    # LBD Component
    LBD = ((local == 3) * qBD * (qLS * f(-w[3] - w[6] + (w[13] ** 2) / 2) + (1 - qLS) * f(-w[3] + (w[13] ** 2) / 2)) +
           (local == 4) * qBD * (qLF * f(-w[3] - w[7] + (w[13] ** 2) / 2) + (1 - qLF) * f(-w[3] + (w[13] ** 2) / 2)) +
           (local == 3) * (1 - qBD) * (qLS * f(w[3] + w[6] + (w[13] ** 2) / 2) + (1 - qLS) * f(w[3] + (w[13] ** 2) / 2)) +
           (local == 4) * (1 - qBD) * (qLF * f(w[3] + w[7] + (w[13] ** 2) / 2) + (1 - qLF) * f(w[3] + (w[13] ** 2) / 2)) +
           (local == 6) * (qBD * (qLS * qLF * f(-w[3] - w[6] - w[7] + (w[13] ** 2) / 2) +
                                  qLS * (1 - qLF) * f(-w[3] - w[6] + (w[13] ** 2) / 2) +
                                  (1 - qLS) * qLF * f(-w[3] - w[7] + (w[13] ** 2) / 2) +
                                  (1 - qLS) * (1 - qLF) * f(-w[3] + (w[13] ** 2) / 2)) +
                     (1 - qBD) * (qLS * qLF * f(w[3] + w[6] + w[7] + (w[13] ** 2) / 2) +
                                  qLS * (1 - qLF) * f(w[3] + w[6] + (w[13] ** 2) / 2) +
                                  (1 - qLS) * qLF * f(w[3] + w[7] + (w[13] ** 2) / 2) +
                                  (1 - qLS) * (1 - qLF) * f(w[3] + (w[13] ** 2) / 2)))) - \
          ((local == 3) | (local == 4) | (local == 6)) * (qBD * np.log(f_qBD) + (1 - qBD) * np.log(1 - f_qBD))
    LBD[(local == 0) | (local == 1) | (local == 2) | (local == 5)] = 0

    # LLS Component
    LLS = ((local == 1) | (local == 3) | (local == 5) | (local == 6)) * \
          (qLS * f(-w[4] - w[14] * alLS + (w[11] ** 2) / 2) +
           (1 - qLS) * f(w[4] + w[14] * alLS + (w[11] ** 2) / 2) -
           qLS * np.log(f_qLS) - (1 - qLS) * np.log(1 - f_qLS))
    LLS[(local == 0) | (local == 2) | (local == 4)] = 0

    # LLF Component
    LLF = ((local == 2) | (local == 4) | (local == 5) | (local == 6)) * \
          (qLF * f(-w[5] - w[15] * alLF + (w[12] ** 2) / 2) +
           (1 - qLF) * f(w[5] + w[15] * alLF + (w[12] ** 2) / 2) -
           qLF * np.log(f_qLF) - (1 - qLF) * np.log(1 - f_qLF))
    LLF[(local == 0) | (local == 1) | (local == 3)] = 0

    # Leps Component
    Leps = -(0.5 * np.log(w[2] ** 2) + (y - w[1]) * (y - w[1]) / (2 * (w[2] ** 2))) - \
           ((1 / (2 * (w[2] ** 2))) * ((local == 3 | local == 4 | local == 6) * w[8] * qBD * (w[8] - 2 * y + 2 * w[1]) +
                                       (local == 1 | local == 3 | local == 5 | local == 6) * w[9] * qLS * (w[9] - 2 * y + 2 * w[1]) +
                                       (local == 2 | local == 4 | local == 5 | local == 6) * w[10] * qLF * (w[10] - 2 * y + 2 * w[1]))) - \
           ((1 / (w[2] ** 2)) * ((local == 3) * w[8] * w[9] * qBD * qLS +
                                 (local == 4) * w[8] * w[10] * qBD * qLF +
                                 (local == 5) * w[9] * w[10] * qLS * qLF +
                                 (local == 6) * w[8] * w[9] * w[10] * qBD * qLS * qLF)) - y

    # L_ex Component
    L_ex = -((local == 5) | (local == 6)) * (f_qLS * f_qLF) / (2 * delta)
    L_ex = np.maximum(L_ex, -1e+2)

    # Total Loss
    L = LBD + LLS + LLF + Leps + L_ex

    return L

