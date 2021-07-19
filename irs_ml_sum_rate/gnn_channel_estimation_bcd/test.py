from gnn_channel_estimation_bcd.channel_est_model import CEModel
from sum_rate_bcd.generate_channel import generate_channel
from sum_rate_bcd.generate_received_pilots import generate_received_pilots_batch,channel_complex2real,decorrelation,generate_pilots_bl
from compute_objective_fun import compute_rate_batch
from scipy.linalg import dft
import numpy as np
import scipy.io as sio
import time,os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def nn_prediction_model(received_pilots, params_system, model_path,input_flag):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = received_pilots.shape[2]

    nn = CEModel(num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, sigma2=1)
    nn.create_network()
    # nn.create_optimizer(training_algorithm='Adam')
    nn.create_initializer()
    initial_run = False
    nn.initialize(initial_run, model_path)

    return nn

def testing():
    # num_sample = 10000
    num_antenna_bs, num_elements_irs, num_user = 8, 100, 3
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    input_flag = 0
    Rician_factor = 10
    noise_power_db = -100  # dBm
    location_user = None
    # channels, set_location_user = generate_channel(params_system,location_user_initial=location_user, num_samples=num_sample, Rician_factor=Rician_factor)
    file_path_channel = '../sum_rate_bcd/channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    channel = sio.loadmat(file_path_channel)
    channels = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])
    set_location_user = channel['location_user']
    A_T, Hd, A_c = channel_complex2real(channels)

    set_len_pilot = np.array([1,3,5,10,20,25,40])*num_user
    rate_sum = []
    err_dB_set = []
    for len_pilot in set_len_pilot:
        path_pilots = './param_pilots_phaseshifts/phase_shifts_pilots' + str(params_system) + '_'+ str(len_pilot) + '_'\
                      + str(Rician_factor) + '_'+ str(noise_power_db) + '_'+ str(input_flag) + '.mat'
        phase_shifts_pilots = sio.loadmat(path_pilots)
        phase_shifts, pilots = phase_shifts_pilots['phase_shifts'], phase_shifts_pilots['pilots']
        # y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)

        y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
        y_ks_tmp = decorrelation(y, pilots)
        y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
        y_mean, y_std = np.mean(y_real_tmp, axis=0), np.std(y_real_tmp, axis=0)
        y_real = (y_real_tmp - y_mean) / y_std
        location_mean, location_std = np.mean(set_location_user, axis=0), np.std(set_location_user, axis=0)
        set_location_user = (set_location_user - location_mean) / location_std
        y_ks_mean, y_ks_std = np.mean(y_ks, axis=0), np.std(y_ks, axis=0)
        y_ks_real = (y_ks - y_ks_mean) / y_ks_std

        model_path = './param_model/model'+ str(params_system) + '_' + str(len_pilot) + '_' + str(Rician_factor)\
                     + '_' + str(noise_power_db)+ '_' + str(input_flag)
        nn = nn_prediction_model(y_real, params_system, model_path,input_flag)
        val_loss = nn.get_loss(y_real, y_ks_real, set_location_user, A_T, Hd)

        A_hat_real = nn.get_A(y_real, y_ks_real, set_location_user)
        A_hat_nn = A_hat_real[:, :, 0:num_antenna_bs, :] + 1j * A_hat_real[:, :, num_antenna_bs:2 * num_antenna_bs, :]
        A_hat_nn = np.transpose(A_hat_nn, (0, 2, 1, 3))

        H_hat_real = nn.get_H(y_real, y_ks_real, set_location_user)
        H_nn = H_hat_real[:, 0:num_antenna_bs, :] + 1j * H_hat_real[:, num_antenna_bs:2 * num_antenna_bs, :]


        err = np.linalg.norm(H_nn - channels[0], axis=(1)) ** 2 +np.linalg.norm(A_hat_nn - A_c, axis=(1, 2)) ** 2
        err = np.sum(err,axis=1)
        err_dB = 10*np.log10(err)
        err_dB_set.append(np.mean(err_dB))

        print('err:', np.mean(err_dB),val_loss)
        # sio.savemat('DNN_CE'+str(params_system) + '_' + str(len_pilot) + '_' + str(Rician_factor) + '_'
        #             + str(noise_power_db) + '_' + str(input_flag)+'.mat',{'channel_bs_irs_user':A_hat_nn,'channel_bs_user':H_nn})
    # sio.savemat('err_dnn.mat', {'len_pilot': set_len_pilot, 'err': err_dB_set})

if __name__ == '__main__':
    ts = time.time()
    testing()
    print('Running time: %0.3f sec: ' % ((time.time() - ts)))
