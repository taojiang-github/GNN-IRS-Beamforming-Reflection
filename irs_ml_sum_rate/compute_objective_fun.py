import numpy as np
from util_func import combine_channel, random_beamforming
from sum_rate_bcd.generate_channel import generate_channel, channel_complex2real
import time


def compute_rate(w, theta, chs, sigma2=1):
    # w: (num_samples, num_antenna_bs, num_user)
    # theta: (num_samples, num_elements_irs)
    (ch_bs_user, ch_irs_user, ch_bs_irs) = chs
    (num_samples, num_antenna_bs, num_user) = w.shape
    num_elements_irs = theta.shape[1]
    rate_sum, rate_all = 0, []

    for ii in range(num_samples):
        w_i, theta_i = w[ii], theta[ii]
        ch_bs_irs_i = ch_bs_irs[ii]
        tmp_rate_sum = 0
        for kk in range(num_user):
            w_i_k = w_i[:, kk]
            ch_bs_user_k, ch_irs_user_k = ch_bs_user[ii, :, kk], ch_irs_user[ii, :, kk]
            h_combine, _ = combine_channel(ch_bs_user_k, ch_irs_user_k, ch_bs_irs_i, theta_i.reshape(-1))
            tmp_k = np.abs(np.dot(h_combine, w_i_k)) ** 2
            tmp_all = 0
            for jj in range(num_user):
                if jj != kk:
                    w_i_j = w_i[:, jj]
                    tmp_all = tmp_all + np.abs(np.dot(h_combine, w_i_j)) ** 2
            sinr_k = tmp_k / (tmp_all + sigma2)
            rate_k = np.log2(1 + sinr_k)
            tmp_rate_sum = tmp_rate_sum + rate_k
        rate_sum = rate_sum + tmp_rate_sum
    rate_sum = rate_sum / num_samples
    return rate_sum


def compute_rate_batch(w, theta, chs, Pt = 0, sigma2=1):
    # w: (num_samples, num_antenna_bs, num_user)
    # theta: (num_samples, num_elements_irs)

    (ch_bs_user, ch_irs_user, ch_bs_irs) = chs
    (num_samples, num_antenna_bs, num_user) = w.shape
    num_elements_irs = theta.shape[1]

    # check constraint on w and theta
    Pt =10**(Pt/10)
    w_norm, w_check = np.squeeze(np.linalg.norm(w, axis=(1, 2)))**2, np.ones(num_samples)
    err_w = np.linalg.norm(w_norm/Pt - w_check)
    theta_abs, theta_check = np.squeeze(np.abs(theta)), np.ones((num_samples, num_elements_irs))
    err_theta = np.linalg.norm(theta_abs - theta_check)
    print('Constrain gap: ', err_w, err_theta)
    if err_w > 1e-3 or err_theta > 1e-3:
        raise ValueError('Beamformers do not meet the constraints')

    rate_sum = 0
    rate_all = []
    ch_combine_set = []
    w_set = []
    SINR_set = []
    for kk in range(num_user):
        ch_bs_user_k = ch_bs_user[:, :, kk]
        ch_irs_user_k = ch_irs_user[:, :, kk]
        ch_bs_irs_user = ch_bs_irs * ch_irs_user_k.reshape(num_samples, 1, num_elements_irs)
        ch_combine_k = ch_bs_user_k.reshape(num_samples, num_antenna_bs, 1) \
                       + ch_bs_irs_user @ theta.reshape(num_samples, num_elements_irs, 1)
        # print(np.matmul(A[:,:,:,kk], theta.reshape(num_samples, num_elements_irs, 1)))
        # aaa=ch_bs_irs_user @ theta.reshape(num_samples, num_elements_irs, 1)
        # print(aaa[0])
        ch_combine_set.append(np.linalg.norm(ch_combine_k,axis=1))
        w_k = w[:, :, kk]
        w_set.append(np.linalg.norm(w_k,axis=1))
        SINRs_numerator = w_k.reshape(num_samples, 1, num_antenna_bs) @ ch_combine_k.reshape(num_samples,
                                                                                             num_antenna_bs, 1)
        SINRs_numerator = np.abs(np.squeeze(SINRs_numerator)) ** 2
        SINRs_denominator = 0
        for jj in range(num_user):
            if jj != kk:
                w_j = w[:, :, jj]
                tmp = w_j.reshape(num_samples, 1, num_antenna_bs) @ ch_combine_k.reshape(num_samples, num_antenna_bs, 1)
                SINRs_denominator = SINRs_denominator + np.abs(np.squeeze(tmp)) ** 2

        SINR_k = SINRs_numerator / (SINRs_denominator + sigma2)
        SINR_set.append(SINR_k)
        rate_k = np.sum(np.log2(1 + SINR_k)) / num_samples
        rate_all.append(rate_k)
        # rate_sum = rate_sum + rate_k

    rate_sum = np.sum(rate_all)
    return rate_sum, rate_all


def test():
    num_antenna_bs, num_elements_irs, num_user = 2, 4, 3
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    num_test = 1000
    chs, set_location_user = generate_channel(params_system, num_samples=num_test)
    _, _, A = channel_complex2real(chs)

    w_rnd, theta_rnd = random_beamforming(num_test, num_antenna_bs, num_elements_irs, num_user)
    rate_rnd = compute_rate_batch(w_rnd, theta_rnd, chs)
    # rate_rnd = compute_rate(w_rnd, theta_rnd, chs)
    print('rate_rnd:%0.5f' % rate_rnd)


if __name__ == '__main__':
    ts = time.time()
    test()
    print('Running time: %0.3f mins ' % ((time.time() - ts) / 60))
