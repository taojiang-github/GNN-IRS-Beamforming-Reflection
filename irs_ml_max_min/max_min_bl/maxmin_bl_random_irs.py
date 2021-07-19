import time
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from max_min_bl.generate_channel import  channel_complex2real
from compute_objective_fun import compute_rate_batch,compute_minrate_batch
import scipy.io as sio
import os


def combine_channel(H0, hd, theta):
    num_antenna_bs, num_elements_irs, num_user = H0.shape
    H = np.empty((num_antenna_bs, num_user), dtype=complex)
    for k in range(num_user):
        tmp = hd[:, k].reshape((num_antenna_bs, 1)) + np.matmul(H0[:, :, k], theta)
        # print(tmp)
        H[:, k] = tmp.reshape((num_antenna_bs,))
    # print(H)
    return H


def compute_m(H, q, sigma2):
    num_antenna_bs, num_user = H.shape
    m = np.empty((num_user, num_antenna_bs, num_antenna_bs), dtype=complex)
    for k in range(num_user):
        tmp = sigma2 * np.eye(num_antenna_bs)
        for ii in range(num_user):
            if ii != k:
                h_i = H[:, ii].reshape((num_antenna_bs, 1))
                hh = np.matmul(h_i, np.transpose(h_i.conjugate()))
                tmp = (q[ii] / num_user) * hh + tmp
        m[k, :, :] = tmp
    return m


def compute_tau(H, m, P_max):
    num_antenna_bs, num_user = H.shape
    tmp = 0
    for k in range(num_user):
        h_k = H[:, k].reshape((num_antenna_bs, 1))
        m_k = m[k, :, :]
        m_inv = np.linalg.inv(m_k)
        hmh = np.matmul(np.transpose(h_k.conjugate()), m_inv)
        hmh = np.matmul(hmh, h_k) / num_user
        hmh = np.squeeze(np.real(hmh))
        tmp = tmp + 1 / hmh
    tau = num_user * P_max / tmp
    return tau


def find_q_fixed_point(H, sigma2, P_max):
    num_antenna_bs, num_user = H.shape
    q = np.random.randn(num_user)
    iter_max = 100
    for ii in range(iter_max):
        m = compute_m(H, q, sigma2)
        tau = compute_tau(H, m, P_max)
        q_old = np.copy(q)
        for k in range(num_user):
            h_k = H[:, k].reshape((num_antenna_bs, 1))
            m_k = m[k, :, :]
            m_inv = np.linalg.inv(m_k)
            q_k = np.matmul(np.transpose(h_k.conjugate()), m_inv)
            q_k = np.real(np.matmul(q_k, h_k)) / num_user
            q[k] = tau / np.squeeze(q_k)
        err = np.linalg.norm(q - q_old)
        # print(err)
        if err <= 1e-5:
            break
    return q


def compute_beamforming_vec(H, sigma2, P_max):
    num_antenna_bs, num_user = H.shape
    g = np.empty((num_user, num_antenna_bs), dtype=complex)
    q = find_q_fixed_point(H, sigma2, P_max)
    m = compute_m(H, q, sigma2)
    tau = compute_tau(H, m, P_max)
    for k in range(num_user):
        h_k = H[:, k].reshape((num_antenna_bs, 1))
        m_k = m[k, :, :]
        m_inv = np.linalg.inv(m_k)
        g_k = np.matmul(m_inv, h_k)
        g_k = g_k / np.linalg.norm(g_k)
        g[k, :] = g_k.reshape(-1)
    return g, tau


def compute_power_allocation(H, g, sigma2, tau):
    num_antenna_bs, num_user = H.shape
    d, f = np.zeros((num_user, num_user)), np.zeros((num_user, num_user))
    for k in range(num_user):
        h_k = H[:, k].reshape((num_antenna_bs, 1))
        g_k = g[k, :].reshape((num_antenna_bs, 1))
        d[k, k] = 1 / (np.abs(np.matmul(np.transpose(h_k.conjugate()), g_k)) ** 2 / num_user)
        for ii in range(num_user):
            if ii != k:
                g_i = g[ii, :].reshape((num_antenna_bs, 1))
                f[k, ii] = np.abs(np.matmul(np.transpose(h_k.conjugate()), g_i)) ** 2 / num_user
    tmp = np.eye(num_user) - tau * np.matmul(d, f)
    tmp_inv = np.linalg.inv(tmp)
    p = np.matmul(d, np.ones((num_user, 1)))
    p = np.matmul(tmp_inv, p) * tau * sigma2
    return p


def compute_maxmin_rate(H, g, p, sigma2):
    num_antenna_bs, num_user = H.shape
    rate = np.empty((num_user))
    for k in range(num_user):
        h_k = H[:, k].reshape((num_antenna_bs, 1))
        g_k = g[k, :].reshape((num_antenna_bs, 1))
        gamma_k_n = (p[k] / num_user) * (np.abs(np.matmul(np.transpose(h_k.conjugate()), g_k)) ** 2)
        gamma_k_d = sigma2
        for ii in range(num_user):
            if ii != k:
                g_i = g[ii, :].reshape((num_antenna_bs, 1))
                gamma_k_d = gamma_k_d + (p[ii] / num_user) * (
                        np.abs(np.matmul(np.transpose(h_k.conjugate()), g_i)) ** 2)
        gamma_k = gamma_k_n / gamma_k_d
        rate[k] = np.log2(1 + gamma_k)
    rate_min = min(rate)
    return rate_min, rate


def compute_min_rate(v,r,b,p,num_user,sigma2):
    rate = []
    for kk in range(num_user):
        rate_numerator = (p[kk]/num_user) * (np.real(np.transpose(v.conjugate()) @ r[kk,kk] @ v)+np.abs(b[kk,kk])**2)
        for ii in range(num_user):
            rate_demo = sigma2
            if ii != kk:
                rate_demo = rate_demo + (p[ii]/num_user) * (np.real(np.transpose(v.conjugate()) @ r[kk,ii] @ v)+np.abs(b[kk,ii])**2)
        rate.append(np.log2(1+rate_numerator/rate_demo))
    rate_min = np.min(rate)
    return  rate_min


def find_theta(g, p, H0, hd, sigma2, eps1=1e-4):
    num_antenna_bs, num_elements_irs, num_user = H0.shape
    r, b = {}, {}
    for kk in range(num_user):
        H0_k = H0[:, :, kk]
        hd_k = hd[:, kk].reshape((num_antenna_bs, 1))
        for ii in range(num_user):
            g_i = g[ii, :].reshape((num_antenna_bs, 1))
            a_ki = np.matmul(np.transpose(H0_k.conjugate()), g_i)
            b[kk, ii] = np.matmul(np.transpose(hd_k.conjugate()), g_i)
            aa = np.matmul(a_ki, np.transpose(a_ki.conjugate()))
            ab = a_ki * b[kk, ii].conjugate()
            ba = np.transpose(ab.conjugate())
            # rr = np.block([ba,0])
            r[kk, ii] = np.block([[aa, ab], [ba, 0+0j]])

    iter_max = 100
    lamb = 0
    for ii in range(iter_max):
        n = num_elements_irs + 1
        x = cp.Variable((n, n), complex=True)
        y = cp.Variable((n, n), hermitian=True)
        # The operator >> denotes matrix inequality.
        constraints = [x >> 0]
        constraints += [x == y]
        constraints += [
            x[i, i] == 1 + 0j for i in range(n)
        ]
        n_d, n_over_d = [], []
        for kk in range(num_user):
            n_k = cp.multiply((cp.real(cp.trace(r[kk, kk] @ x)) + np.abs(b[kk, kk]) ** 2), p[kk]) / num_user
            for ii in range(num_user):
                d_k = 0 + sigma2
                if ii != kk:
                    d_k = d_k + cp.multiply((cp.real(cp.trace(r[kk, ii] @ x)) + np.abs(b[kk, ii]) ** 2), p[ii]) / num_user
            n_d.append(n_k - lamb * d_k)
            n_over_d.append(n_k / d_k)
        n_over_d_min = cp.min(cp.vstack(n_over_d))
        prob = cp.Problem(cp.Minimize(-cp.min(cp.vstack(n_d))),
                          constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)
        # prob.solve(solver = cp.SCS, verbose=True)
        # print(-prob.value)
        if -prob.value <= eps1:
            break
        else:
            lamb = n_over_d_min.value

    # Print result.
    # print("The optimal value is", prob.value)
    # print("A solution X is")
    # print(X.value)
    vv = x.value
    w, v = np.linalg.eig(vv)
    w = np.abs(np.real(w.reshape((n, 1))))
    if len(w) == 1 or np.abs(w[1]) <= 1e-3:
        v0 =  v[:, 0].reshape([-1, 1])
    else:  # Gaussian randomization
        v0 = v[:, 0].reshape([-1, 1])
        rate_min_old = compute_min_rate(v0, r, b, p, num_user, sigma2)
        for jj in range(100):
            r_vec = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[len(w), 1]) \
                    + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[len(w), 1])
            vw = v*np.sqrt(w.reshape([1,-1]))
            v0_tmp = np.matmul(vw, r_vec)
            rate_min = compute_min_rate(v0_tmp, r, b, p, num_user, sigma2)
            if rate_min>rate_min_old:
                v0 = np.copy(v0_tmp)
                rate_min_old = np.copy(rate_min)
                # print('rate_min:',rate_min)
        print('Failed to return rank one matrix')

    theta = v0[0:-1] / v0[-1]
    theta = theta / np.abs(theta)

    return theta


def alter_max(A, Hd, P_max=1, sigma2=0.1):
    num_samples, num_antenna_bs, num_elements_irs, num_user = A.shape

    theta_all, g_all, p_all, w_all = [], [], [], []
    rate_min_all, rate_all = [], []
    for ii in range(num_samples):
        print('===================\n Sample: ', ii, '\n===================')
        H0, hd = A[ii, :, :, :], Hd[ii, :, :]

        # ------------theta initialization---------------------
        #---random initialization---
        theta = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, 1]) \
                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, 1])
        theta = theta / np.abs(theta)

        #---center of means initialization---
        # d_k = np.linalg.norm(np.array([120,35]) - np.array([100,0]))
        # cos_phi_3_k = (120 - 100) / d_k
        # uu = np.array(range(num_elements_irs))
        # theta = np.exp(1j * np.pi * cos_phi_3_k * uu)
        # theta = theta.reshape([num_elements_irs,1])

        rate_min_old = 0
        iter_max = 1
        for jj in range(iter_max):
            # ------------combine channel---------------------------
            H = combine_channel(H0, hd, theta)

            # ------------compute beamforming vector at BS-------------------------
            g, tau = compute_beamforming_vec(H, sigma2, P_max)

            # ------------compute power allocation vector at BS-------------------------
            p = compute_power_allocation(H, g, sigma2, tau)

            rate_min, rate = compute_maxmin_rate(H, g, p, sigma2)
            print('optimize g,p:   ', rate, rate_min)
            delta_increase = rate_min - rate_min_old
            if delta_increase < 1e-3:
                break
            else:
                rate_min_old = np.copy(rate_min)

            theta = find_theta(g, p, H0, hd, sigma2)

            # H = combine_channel(H0, hd, theta)
            # rate_min, rate = compute_maxmin_rate(H,g,p,sigma2)
            # print('optimize theta:',rate, rate_min)

        theta_all.append(theta)
        g_all.append(g)
        p_all.append(p)
        rate_min_all.append(rate_min)
        rate_all.append(rate)
        w = np.matmul(np.diag(np.sqrt(p.reshape(-1) / num_user)), g)
        w = np.transpose(w)
        print('w_norm:', np.linalg.norm(w) ** 2)
        w_all.append(w)

    return np.array(theta_all), g_all, p_all, rate_min_all, rate_all, np.array(w_all)


def maxmin_perfect_csi(params_system, Rician_factor):
    file_path_channel = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    # if os.path.isfile(file_path_channel) is False:
    #     print('-------------Generating channel------------')
    #     channels, set_location_user = generate_channel(params_system,
    #                                                    num_samples=num_test,
    #                                                    location_user_0=location_user,
    #                                                    Rician_factor=Rician_factor)
    # else:
    #     print('-------------Loading channel------------')
    #     channel = sio.loadmat(file_path_channel)
    #     channels = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])
    channel = sio.loadmat(file_path_channel)
    channels = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])

    _, _, A = channel_complex2real(channels)
    Hd = channels[0]
    theta_all, g, p, rate_min_all, rate_all, w_all = alter_max(A, Hd)

    rate_nn, rate_nn_all = compute_rate_batch(w_all.conjugate(), theta_all, channels,sigma2=0.1)
    rate_min, rate_min_all = compute_minrate_batch(w_all.conjugate(), theta_all, channels,sigma2=0.1)

    file_path = './channel_data/perfect_csi_rnd_irs' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    sio.savemat(file_path,
                {'w_all': w_all,
                 'theta_all': theta_all,
                 'rate_min' : rate_min,
                 'rate_min_all':rate_min_all,
                 'rate_sum':rate_nn,
                 'rate_all':rate_nn_all
                 })

    print('\n===Simulation parameters====')
    print('params_system:', params_system)
    # print('Rician_factor, noise_power_db:', Rician_factor, noise_power_db)
    # print('len_pilot:', len_pilot)
    print('===Simulation results===')
    print('rate %0.3f:' % rate_nn)
    print('rate per user', rate_nn_all)
    print('rate_min %0.3f:' % np.mean(rate_min))
    print('rate_min_all', rate_min_all)
    print('=============================\n')


def maxmin_imperfect_csi(params_system, len_pilot, Rician_factor, noise_power_db):
    file_path_channel = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor)  + '_' + str(len_pilot) + '_' + str(noise_power_db) + '.mat'
    channel_est = sio.loadmat(file_path_channel)
    Hd,A = channel_est['channel_bs_user'], channel_est['channel_bs_irs_user']
    theta_all, g, p, rate_min_all, rate_all, w_all = alter_max(A, Hd)

    file_path_channel = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor)  + '.mat'
    channel = sio.loadmat(file_path_channel)
    channel_true = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])
    rate_nn, rate_nn_all = compute_rate_batch(w_all.conjugate(), theta_all, channel_true,sigma2=0.1)
    rate_min, rate_min_all = compute_minrate_batch(w_all.conjugate(), theta_all, channel_true,sigma2=0.1)

    file_path = './channel_data/imperfect_csi_rnd_irs' + str(params_system) + '_' + str(Rician_factor)  + '_' + str(len_pilot) + '_' + str(noise_power_db) + '.mat'
    sio.savemat(file_path,
                {'w_all': w_all,
                 'theta_all': theta_all,
                 'rate_min' : rate_min,
                 'rate_min_all':rate_min_all,
                 'rate_sum':rate_nn,
                 'rate_all':rate_nn_all
                 })

    print('\n===Simulation parameters====')
    print('params_system:', params_system)
    # print('Rician_factor, noise_power_db:', Rician_factor, noise_power_db)
    # print('len_pilot:', len_pilot)
    print('===Simulation results===')
    print('rate %0.3f:' % rate_nn)
    print('rate per user', rate_nn_all)
    print('rate_min %0.3f:' % np.mean(rate_min))
    print('rate_min_all', rate_min_all)
    print('=============================\n')
    return  rate_min,rate_min_all


def main_imperfect_csi(params_system,Rician_factor,noise_power_db):
    num_user = params_system[2]
    set_len_pilot = np.array([25])*num_user
    set_rate_min,set_rate_all = [],[]
    for len_pilot in set_len_pilot:
        print('len_pilot',len_pilot)
        rate_min, rate_all = maxmin_imperfect_csi(params_system, len_pilot, Rician_factor, noise_power_db)
        set_rate_min.append(rate_min)
        set_rate_all.append(rate_all)

        filename = 'maxmin_bl_rnd_irs'+ str(params_system) + '_' + str(Rician_factor) + '_'  + str(noise_power_db)+ '_.mat'
        sio.savemat(filename,{'set_rate_min':set_rate_min,
                              'set_rate_all':set_rate_all,
                              'set_len_pilot':set_len_pilot})

    plt.plot(set_len_pilot,set_rate_min,'o-')
    plt.title('Maxmin Baseline')
    plt.xlabel('Pilot length')
    plt.ylabel('Rate')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    ts = time.time()
    Rician_factor = 10
    params_system = (4, 20, 3)
    noise_power_db = -100
    maxmin_perfect_csi(params_system, Rician_factor)
    # main_imperfect_csi(params_system, Rician_factor, noise_power_db)
    print('Running time: %0.3f mins: ' % ((time.time() - ts) / 60))
