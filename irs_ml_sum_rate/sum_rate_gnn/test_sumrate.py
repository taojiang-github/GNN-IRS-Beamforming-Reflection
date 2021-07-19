from sum_rate_gnn.sumrate_model import SumRateModel
from sum_rate_bcd.generate_channel import generate_channel
from sum_rate_bcd.generate_received_pilots import generate_received_pilots_batch, channel_complex2real, decorrelation, \
    generate_pilots_bl, generate_pilots_bl_v2
from compute_objective_fun import compute_rate_batch
from scipy.linalg import dft
import numpy as np
import scipy.io as sio
import time, os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def nn_prediction_model(received_pilots, params_system, Pt, model_path, input_flag):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = received_pilots.shape[2]

    nn = SumRateModel(num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, Pt)
    nn.create_network()
    # nn.create_optimizer(training_algorithm='Adam')
    nn.create_initializer()
    initial_run = False
    nn.initialize(initial_run, model_path)

    return nn


def testing(Pt_u, Pt_d):
    num_sample = 10000
    num_antenna_bs, num_user = 8, 3
    # num_antenna_bs, num_user = 8, 3
    num_elements_irs = 100
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    input_flag = 0
    Rician_factor = 10
    noise_power_db = -100  # dBm

    ####
    # Pt_d_train = Pt_d
    # Pt_u_train = Pt_u

    ####
    Pt_d_train = 10
    Pt_u_train = 15

    location_user = None
    # channels, set_location_user = generate_channel(params_system,location_user_initial=location_user, num_samples=num_sample, Rician_factor=Rician_factor)
    file_path_channel = '../sum_rate_bcd/channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    channel = sio.loadmat(file_path_channel)
    channels = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])
    set_location_user = channel['location_user']

    # set_len_pilot = np.array([1,2,3,4,5,10,20,40])*num_user
    set_len_pilot = np.array([15]) * num_user
    rate_sum = []
    for len_pilot in set_len_pilot:
        path_pilots = './param_pilots_phaseshifts/phase_shifts_pilots' + str(params_system) + '_' + str(len_pilot) + '_' \
                      + str(Rician_factor) + '_' + str(noise_power_db) + '_' + str(Pt_u_train) + '_' + str(
            Pt_d_train) + '_' + str(input_flag) + '.mat'
        phase_shifts_pilots = sio.loadmat(path_pilots)
        phase_shifts, pilots = phase_shifts_pilots['phase_shifts'], phase_shifts_pilots['pilots']

        y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt=Pt_u)
        y_ks_tmp = decorrelation(y, pilots)
        y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
        y_mean, y_std = np.mean(y_real_tmp, axis=0), np.std(y_real_tmp, axis=0)
        y_real = (y_real_tmp - y_mean) / y_std
        location_mean, location_std = np.mean(set_location_user, axis=0), np.std(set_location_user, axis=0) + 1e-15
        set_location_user = (set_location_user - location_mean) / location_std
        y_ks_mean, y_ks_std = np.mean(y_ks, axis=0), np.std(y_ks, axis=0)
        y_ks_real = (y_ks - y_ks_mean) / y_ks_std

        model_path = './param_model/model' + str(params_system) + '_' + str(len_pilot) + '_' + str(
            Rician_factor) + '_' + str(noise_power_db) + '_' + str(Pt_u_train) + '_' + str(Pt_d_train) + '_' + str(
            input_flag)
        nn = nn_prediction_model(y_real, params_system, Pt_d, model_path, input_flag)
        w_nn_real = nn.get_w(y_real, y_ks_real, set_location_user)
        w_nn = w_nn_real[:, 0:num_antenna_bs, :] + 1j * w_nn_real[:, num_antenna_bs:2 * num_antenna_bs, :]
        # tmp = np.linalg.norm(w_nn,axis=1)
        theta_nn_real = nn.get_theta(y_real, y_ks_real, set_location_user)
        theta_nn = theta_nn_real[:, 0:num_elements_irs] + 1j * theta_nn_real[:, num_elements_irs:2 * num_elements_irs]

        rate_nn, rate_all = compute_rate_batch(w_nn, theta_nn, channels, Pt=Pt_d, sigma2=1)
        rate_sum.append(rate_nn)

        print('Testing sum rate:', rate_nn)
        print('Testing all rate:', rate_all)

    # sio.savemat('results_test_sumrate'+str(input_flag)+'.mat',{'set_len_pilot':set_len_pilot,'rate_sum':rate_sum})
    return rate_nn


def pilots_phase_shift_new(phase_shifts_train, num_user_train, num_user_test, len_pilots_train, len_pilots_test):
    phase_shifts = phase_shifts_train[:, 0:len_pilots_train:num_user_train]
    len_frame = num_user_test
    num_frame = len_pilots_test // len_frame
    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)

    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user_test]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilots_test, num_user_test])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(pilots)), pilots)), '\n')
    return phase_shifts, pilots


def testing_K(num_user_test):
    num_sample = 1000
    input_flag = 0
    Rician_factor = 10
    noise_power_db = -100  # dBm
    Pt_d = 10
    Pt_u = 15
    location_user = None

    num_antenna_bs, num_elements_irs, num_user_train = 8, 100, 6
    params_system_train = (num_antenna_bs, num_elements_irs, num_user_train)
    params_system_test = (num_antenna_bs, num_elements_irs, num_user_test)

    channels, set_location_user = generate_channel(params_system_test, location_user_initial=location_user,
                                                   num_samples=num_sample, Rician_factor=Rician_factor)

    set_len_pilot = np.array([25]) * num_user_test
    len_pilot_train = 25 * num_user_train
    rate_sum = []
    for len_pilot in set_len_pilot:
        path_pilots_train = './param_pilots_phaseshifts/phase_shifts_pilots' + str(params_system_train) + '_' \
                            + str(len_pilot_train) + '_' + str(Rician_factor) + '_' + str(noise_power_db) + '_'\
                            + str(Pt_u) + '_' + str(Pt_d) + '_' + str(input_flag) + '.mat'
        phase_shifts_pilots_train = sio.loadmat(path_pilots_train)
        phase_shifts_train,pilots_trian = phase_shifts_pilots_train['phase_shifts'], phase_shifts_pilots_train['pilots']
        phase_shifts, pilots = pilots_phase_shift_new(phase_shifts_train, num_user_train, num_user_test,
                                                      len_pilot_train, len_pilot)

        y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt=Pt_u)
        y_ks_tmp = decorrelation(y, pilots)
        y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
        y_mean, y_std = np.mean(y_real_tmp, axis=0), np.std(y_real_tmp, axis=0)
        y_real = (y_real_tmp - y_mean) / y_std
        location_mean, location_std = np.mean(set_location_user, axis=0), np.std(set_location_user, axis=0) + 1e-15
        set_location_user = (set_location_user - location_mean) / location_std
        y_ks_mean, y_ks_std = np.mean(y_ks, axis=0), np.std(y_ks, axis=0)
        y_ks_real = (y_ks - y_ks_mean) / y_ks_std

        model_path = './param_model/model' + str(params_system_train) + '_' + str(len_pilot_train) + '_' + str(
            Rician_factor) + '_' + str(noise_power_db) + '_' + str(Pt_u) + '_' + str(Pt_d) + '_' + str(input_flag)
        nn = nn_prediction_model(y_real, params_system_test, Pt_d, model_path, input_flag)
        w_nn_real = nn.get_w(y_real, y_ks_real, set_location_user)
        w_nn = w_nn_real[:, 0:num_antenna_bs, :] + 1j * w_nn_real[:, num_antenna_bs:2 * num_antenna_bs, :]
        theta_nn_real = nn.get_theta(y_real, y_ks_real, set_location_user)
        theta_nn = theta_nn_real[:, 0:num_elements_irs] + 1j * theta_nn_real[:, num_elements_irs:2 * num_elements_irs]

        # theta_nn = np.zeros((num_sample,num_elements_irs),dtype=complex)
        rate_nn, rate_all = compute_rate_batch(w_nn, theta_nn, channels, Pt=Pt_d, sigma2=1)
        rate_sum.append(rate_nn)

        print('Testing sum rate:', rate_nn)
        print('Testing all rate:', rate_all)

    return rate_nn

#########generalizability to downlink transmit power
def rate_vs_Ptd():
    Pt_u = 15
    rate_set = []
    P_set = [0, 2, 4, 6, 8, 10]
    for Pt_d in P_set:
        rate = testing(Pt_u=Pt_u, Pt_d=Pt_d)
        rate_set.append(rate)
    sio.savemat('results_test_sumrate_Pt_d_genarization2.mat', {'set_len_pilot': P_set, 'rate_sum': rate_set})

#########generalizability to uplink transmit power
def rate_vs_Ptu():
    Pt_d = 5
    rate_set = []
    P_set = [10, 12, 14, 16, 18, 20]
    for Pt_u in P_set:
        rate = testing(Pt_u=Pt_u, Pt_d=Pt_d)
        rate_set.append(rate)
    # sio.savemat('results_test_sumrate_Pt_u.mat',{'set_len_pilot':P_set,'rate_sum':rate_set})
    sio.savemat('results_test_sumrate_Pt_u_generalization.mat', {'set_len_pilot': P_set, 'rate_sum': rate_set})

#########generalizability to number of users K
def rate_vs_K():
    num_user_test = [2, 3, 4, 5, 6, 7, 8]
    rate_set = []
    for ii in num_user_test:
        rate = testing_K(ii)
        rate_set.append(rate)
    sio.savemat('results_test_rate_vs_num_user10.mat',
                {'num_user_test': num_user_test, 'rate_sum': rate_set})


if __name__ == '__main__':
    ts = time.time()
    # rate_vs_Ptd()
    # rate_vs_Ptu()
    # rate_vs_K()
    testing(Pt_d=10, Pt_u=15)
    print('Running time: %0.3f sec: ' % ((time.time() - ts)))
