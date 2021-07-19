from sum_rate_bcd.generate_channel import generate_channel, channel_complex2real
import numpy as np
from scipy.linalg import dft
from util_func import combine_channel, batch_combine_channel, ls_estimator, lmmse_estimator
import matplotlib.pyplot as plt

def generate_pilots_nn(len_pilot, num_elements_irs, num_user):
    phase_shifts = np.random.randn(num_elements_irs, len_pilot) + 1j * np.random.randn(num_elements_irs, len_pilot)
    phase_shifts = phase_shifts / np.abs(phase_shifts)
    if len_pilot >= num_user:
        pilots = dft(len_pilot)
        pilots = pilots[:, 0:num_user]
    else:
        raise ValueError('Pilot length is smaller than number of users')
    return phase_shifts, pilots


def generate_pilots_bl(len_pilot, num_elements_irs, num_user):
    len_frame = num_user
    num_frame = len_pilot // len_frame
    if num_frame > num_elements_irs + 1:
        phase_shifts = dft(num_frame)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]
    else:
        phase_shifts = dft(num_elements_irs + 1)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]
    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    phase_shifts = np.delete(phase_shifts, 0, axis=0)
    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, num_user])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(X)), X)), '\n')
    return phase_shifts, pilots


def generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user):
    len_frame = num_user
    num_frame = len_pilot // len_frame
    phase_shifts = np.random.randn(num_elements_irs, num_frame) + \
                   1j * np.random.randn(num_elements_irs, num_frame)
    phase_shifts = phase_shifts / np.abs(phase_shifts)
    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    # pilots = dft(len_pilot)
    # pilots = pilots[:, 0:num_user]
    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, num_user])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(pilots)), pilots)), '\n')
    return phase_shifts, pilots


def generate_received_pilots(channels, phase_shifts, pilots, noise_power_db, scale_factor=100, Pt=15):
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    (num_samples, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]
    len_pilots = phase_shifts.shape[1]

    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))

    y = []
    y_real = []

    for ii in range(num_samples):
        y_tmp = np.zeros((num_antenna_bs, len_pilots), dtype=complex)
        for ell in range(len_pilots):
            for k in range(num_user):
                channel_bs_user_k = channel_bs_user[ii, :, k]
                channel_irs_user_k = channel_irs_user[ii, :, k]
                channel_bs_irs_i = channel_bs_irs[ii]
                channel_combine, _ = combine_channel(channel_bs_user_k, channel_irs_user_k,
                                                     channel_bs_irs_i, phase_shifts[:, ell])
                pilots_k = pilots[:, k]
                y_tmp[:, ell] = y_tmp[:, ell] + channel_combine * pilots_k[ell]

        noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, len_pilots]) \
                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, len_pilots])
        y_tmp = y_tmp + noise_sqrt * noise
        y.append(y_tmp)
        y_tmp_real = np.concatenate([y_tmp.real, y_tmp.imag], axis=0)
        y_real.append(y_tmp_real)

    return np.array(y), np.array(y_real)


def generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, scale_factor=100, Pt=15):
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    (num_samples, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]
    len_pilots = phase_shifts.shape[1]

    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))

    y = np.zeros((num_samples, num_antenna_bs, len_pilots), dtype=complex)
    for kk in range(num_user):
        channel_bs_user_k = channel_bs_user[:, :, kk]
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine = batch_combine_channel(channel_bs_user_k, channel_irs_user_k,
                                                channel_bs_irs, phase_shifts)
        pilots_k = pilots[:, kk]
        pilots_k = np.array([pilots_k] * num_samples)
        pilots_k = pilots_k.reshape((num_samples, 1, len_pilots))
        y = y + channel_combine * pilots_k

    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots])
    y = y + noise_sqrt * noise

    y_real = np.concatenate([y.real, y.imag], axis=1)

    return np.array(y), np.array(y_real)


def decorrelation(received_pilots, pilots):
    (len_pilots, num_user) = pilots.shape
    (num_samples, num_antenna_bs, _) = received_pilots.shape
    pilots = np.array([pilots] * num_samples)
    pilots = pilots.reshape((num_samples, len_pilots, num_user))

    len_frame = num_user
    num_frame = len_pilots // len_frame

    x_tmp = np.conjugate(pilots[:, 0:len_frame, :])
    y_decode = np.zeros([num_samples, num_antenna_bs, num_user, num_frame], dtype=complex)
    for jj in range(num_frame):
        y_k = received_pilots[:, :, jj * len_frame:(jj + 1) * len_frame]
        y_decode_tmp = y_k @ x_tmp / len_frame
        y_decode[:, :, :, jj] = y_decode_tmp
    return y_decode


def channel_estimation_ls(params_system, y, pilots, phase_shifts, stat_info=None):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]

    y_d = decorrelation(y, pilots)
    len_frame = num_user
    channel_bs_user_est = np.zeros((num_sample, num_antenna_bs, num_user), dtype=complex)
    channel_bs_irs_user_est = np.zeros((num_sample, num_antenna_bs, num_elements_irs, num_user), dtype=complex)
    for kk in range(num_user):
        y_k = y_d[:, :, kk, :]
        ones = np.ones((1, len_pilot))
        phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
        phaseshifts_frame = phaseshifts_new[:, 0:len_pilot:len_frame]
        channel_est = ls_estimator(y_k, phaseshifts_frame)
        channel_bs_user_est[:, :, kk] = channel_est[:, :, 0]
        channel_bs_irs_user_est[:, :, :, kk] = channel_est[:, :, 1:num_elements_irs + 1]

    return channel_bs_user_est, channel_bs_irs_user_est


def compute_stat_info(params_system, noise_power_db, location_user, Rician_factor, num_samples=10000):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = num_user * 4
    len_frame = num_user
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)
    channels, set_location_user = generate_channel(params_system,location_user_initial=location_user,
                                                   Rician_factor=Rician_factor, num_samples=num_samples)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    y, _ = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
    Y = decorrelation(y, pilots)
    A, Hd, = channel_bs_irs_user, channel_bs_user

    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    A, Hd, Y = A[:, :, :, 0], Hd[:, :, 0], Y[:, :, 0, :]
    # A_h = np.zeros([num_samples, num_antenna_bs, num_elements_irs + 1]) + 1j * np.zeros(
    #     [num_samples, num_antenna_bs, num_elements_irs + 1])
    # for ii in range(num_samples):
    #     A_h[ii, :, :] = np.concatenate((Hd[ii, :].reshape(-1, 1), A[ii, :, :]), axis=1)
    A_h = np.concatenate((Hd.reshape(-1, num_antenna_bs, 1), A), axis=2)
    A = A_h

    mean_A, mean_Y = np.mean(A, axis=0, keepdims=True), np.mean(Y, axis=0, keepdims=True)
    # print(mean_Y - mean_A @ Q)
    A = A - mean_A
    C_A = np.sum(np.matmul(np.transpose(A.conjugate(), (0, 2, 1)), A), axis=0) / num_samples
    Y = Y - mean_Y
    # print(Y-A@Q)
    C_Y = np.sum(np.matmul(np.transpose(Y.conjugate(), (0, 2, 1)), Y), axis=0) / num_samples
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    stat_info = (gamma_n, C_A, mean_A)
    return stat_info


def channel_estimation_lmmse(params_system, y, pilots, phase_shifts, stat_info):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]

    len_frame = num_user
    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    (gamma_n, C_A, mean_A) = stat_info
    C_Y = np.matmul(np.matmul(np.transpose(Q.conjugate()), C_A), Q) + gamma_n * np.eye(Q.shape[1])
    mean_Y = np.matmul(mean_A, Q)

    y_d = decorrelation(y, pilots)
    channel_bs_user_est = np.zeros((num_sample, num_antenna_bs, num_user), dtype=complex)
    channel_bs_irs_user_est = np.zeros((num_sample, num_antenna_bs, num_elements_irs, num_user), dtype=complex)
    for kk in range(num_user):
        y_k = y_d[:, :, kk, :]

        channel_est = lmmse_estimator(y_k, Q, C_A, C_Y, mean_A, mean_Y)
        channel_bs_user_est[:, :, kk] = channel_est[:, :, 0]
        channel_bs_irs_user_est[:, :, :, kk] = channel_est[:, :, 1:num_elements_irs + 1]

    return channel_bs_user_est, channel_bs_irs_user_est


def test_channel_estimation_lmmse(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    # phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)
    phase_shifts, pilots = generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user)

    # print(phase_shifts, np.abs(phase_shifts))
    # print(pilots, '\n\n', np.diag(pilots @ np.transpose(pilots.conjugate())))
    channels, set_location_user = generate_channel(params_system,
                                                   num_samples=num_sample, location_user_initial=location_user,
                                                   Rician_factor=Rician_factor)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
    stat_info = compute_stat_info(params_system, noise_power_db, location_user, Rician_factor)

    # ===channel estimation===
    channel_bs_user_est, channel_bs_irs_user_est = channel_estimation_lmmse(params_system, y, pilots, phase_shifts,stat_info)

    #---MSE---
    # err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2
    # err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2
    # ---NMSE---
    err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2/np.linalg.norm(channel_bs_user, axis=(1))**2
    err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2/np.linalg.norm(channel_bs_irs_user, axis=(1,2))**2

    # print('Direct link estimation error (num_sample, num_user):\n', err_bs_user)
    # print('Cascaded link estimation error (num_sample, num_user):\n', err_bs_irs_user)
    return np.mean(err_bs_user), np.mean(err_bs_irs_user)


def main():
    num_antenna_bs, num_elements_irs, num_user, num_sample = 8, 100, 3, 1000
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    noise_power_db, Rician_factor = -100, 10
    location_user = None
    set_len_pilot = np.array([3,4,5,6,7,8,9,10,20,30,40])*num_user
    err_lmmse = []

    for len_pilot in set_len_pilot:
        err3, err4 = test_channel_estimation_lmmse(params_system, len_pilot, noise_power_db, location_user, Rician_factor,
                                                   num_sample)
        print('lmmse estimation:', err3, err4)
        err_lmmse.append(err3+err4)
    print(err_lmmse)

    #----------
    plt.figure()
    plt.title('Error')
    plt.plot(set_len_pilot,err_lmmse,'s-',label='lmmse')
    plt.xlabel('Pilot length')
    plt.ylabel('NMSE')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
