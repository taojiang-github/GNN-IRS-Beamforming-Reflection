import numpy as np


def combine_channel(channel_bs_user_k, channel_irs_user_k, channel_bs_irs, phase_shifts):
    channel_combine_irs = channel_bs_irs @ np.diag(phase_shifts)
    channel_combine = channel_bs_user_k + channel_combine_irs @ channel_irs_user_k
    # channel_combine_irs2 = channel_bs_irs @ np.diag(channel_irs_user_k) @  phase_shifts
    return channel_combine, channel_combine_irs


def batch_combine_channel(channel_bs_user_k, channel_irs_user_k, channel_bs_irs, phase_shifts):
    (num_sample, num_antenna_bs, num_elements_ir) = channel_bs_irs.shape
    len_pilots = phase_shifts.shape[1]

    channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape((num_sample, 1, num_elements_ir))
    channel_bs_user_k = np.repeat(channel_bs_user_k, len_pilots, axis=1)
    channel_combine = channel_bs_user_k.reshape((num_sample, num_antenna_bs, len_pilots)) \
                      + channel_combine_irs @ phase_shifts

    return channel_combine


def random_beamforming(num_test, num_antenna_bs, num_elements_irs, num_user):
    w_rnd = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_antenna_bs, num_user]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_antenna_bs, num_user])
    w_rnd_norm = np.linalg.norm(w_rnd, axis=(1, 2), keepdims=True)
    w_rnd = w_rnd / w_rnd_norm

    theta_rnd = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_elements_irs]) \
                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_elements_irs])
    theta_rnd = theta_rnd / np.abs(theta_rnd)

    return w_rnd, theta_rnd


def ls_estimator(y, x):
    """
    y = h *x + n
    y: batch_size*m*l
    h: batch_size*m*n
    x: batch_size*n*l

    Output: h = y*x^H*(x*x^H)^-1
    """
    n, ell = x.shape[0], x.shape[1]
    x_H = np.transpose(x.conjugate())
    if ell < n:
        x_Hx = np.matmul(x_H, x)
        # print('Cond number:',np.linalg.cond(x_Hx))
        x_Hx_inv = np.linalg.inv(x_Hx)
        h = np.matmul(y, x_Hx_inv)
        h = np.matmul(h, x_H)
    elif ell == n:
        # print('Cond number:',np.linalg.cond(x))
        h = np.linalg.inv(x)
        h = np.matmul(y, h)
    else:
        xx_H = np.matmul(x, x_H)
        # print('Cond number:',np.linalg.cond(xx_H))
        xx_H_inv = np.linalg.inv(xx_H)
        h = np.matmul(y, x_H)
        h = np.matmul(h, xx_H_inv)
    return h


def lmmse_estimator(Y, Q, C_A, C_Y, mean_A, mean_Y):
    # # Y = AQ+N

    # ================================================
    # A = np.matmul(Y,np.linalg.inv(C_Y))
    # A = np.matmul(A,np.transpose(Q.conjugate()))
    # A = np.matmul(A,C_A)

    Y = Y - mean_Y
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    n, ell = Q.shape[0], Q.shape[1]
    if ell > n:
        QQ_H = np.matmul(Q, Q_H)
        C_A_inv = np.linalg.inv(C_A)
        tmp = np.linalg.inv(gamma_n * C_A_inv + QQ_H)
        tmp = np.matmul(tmp, QQ_H)
        tmp = np.matmul(C_A_inv, tmp)
        tmp = np.matmul(tmp, C_A)
        A = ls_estimator(Y, Q)
        A = np.matmul(A, tmp)
    else:
        tmp = np.matmul(Q_H, C_A)
        tmp = np.matmul(tmp, Q)
        tmp = tmp + gamma_n * np.eye(ell)
        tmp = np.linalg.inv(tmp)
        A = np.matmul(Y, tmp)
        A = np.matmul(A, Q_H)
        A = np.matmul(A, C_A)

    return A + mean_A
