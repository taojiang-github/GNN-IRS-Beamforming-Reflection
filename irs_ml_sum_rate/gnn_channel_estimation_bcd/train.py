from sum_rate_bcd.generate_channel import generate_channel, channel_complex2real
from sum_rate_bcd.generate_received_pilots import generate_pilots_nn, generate_pilots_bl, generate_pilots_bl_v2, \
    generate_received_pilots_batch, decorrelation
from gnn_channel_estimation_bcd.channel_est_model import CEModel
from util_func import random_beamforming
from compute_objective_fun import compute_rate_batch
import numpy as np
import scipy.io as sio
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train_network(param_training, params_system, noise_power_db, phase_shifts, pilots, data_test, model_path,
                  location_user, Rician_factor, verbose=True):
    (Y_test, A_test, H_d_test, location_test, y_ks_test, y_mean, y_std,
     location_mean, location_std, y_ks_mean, y_ks_std, channels_t) = data_test
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    (learning_rate, max_epochs, batch_size, iter_per_epoch, initial_run, save_model, input_flag) = param_training
    len_pilot = pilots.shape[0]

    nn = CEModel(num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, sigma2=1)
    nn.create_network()
    nn.create_optimizer(training_algorithm='Adam', learning_rate=learning_rate)
    nn.create_initializer()
    nn.initialize(initial_run, model_path)

    no_increase = 0
    train_loss = np.zeros((max_epochs + 1))
    val_loss = np.zeros((max_epochs + 1))
    train_loss[0] = None
    val_loss[0] = nn.get_loss(Y_test, y_ks_test, location_test, A_test, H_d_test)
    best_loss = val_loss[0]
    if verbose:
        print('\n===Training====')
    for epoch in range(max_epochs):
        for jj in range(iter_per_epoch):
            channels_train, locations_train = generate_channel(params_system, num_samples=batch_size,
                                                               location_user_initial=location_user,
                                                               Rician_factor=Rician_factor)
            y_complex, Y_batch = generate_received_pilots_batch(channels_train, phase_shifts, pilots, noise_power_db)
            y_ks_tmp = decorrelation(y_complex, pilots)
            y_ks_batch = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
            y_ks_batch = (y_ks_batch - y_ks_mean) / y_ks_std
            Y_batch = (Y_batch - y_mean) / y_std
            locations_train = (locations_train - location_mean) / location_std
            A_T_batch, H_d_batch, _ = channel_complex2real(channels_train)
            nn.train(Y_batch, y_ks_batch, locations_train, A_T_batch, H_d_batch)

        train_loss[epoch + 1]= nn.get_loss(Y_batch, y_ks_batch, locations_train, A_T_batch, H_d_batch)
        val_loss[epoch + 1]= nn.get_loss(Y_test, y_ks_test, location_test, A_test, H_d_test)
        if best_loss > val_loss[epoch + 1]:
            best_loss = val_loss[epoch + 1]
            if save_model:
                if not os.path.exists('param_model'):
                    os.makedirs('param_model')
                nn.save_model(model_path)
            no_increase = 0
        else:
            no_increase = no_increase + 1
            if no_increase > 10:
                break

        if verbose:
            print('epoch %3d:' % (epoch + 1), 'lr:%0.3e' % nn.learning_rate(), 'no_increase:%d' % no_increase,
                  'train_loss:%0.4f' % train_loss[epoch + 1], 'test_loss:%0.4f' % val_loss[epoch + 1])

    return nn, train_loss, val_loss


def main_rate_vs_pilot(len_pilots_tmp, params_system, input_flag, Rician_factor,noise_power_db,Pt_u):
    initial_run, save_model = True, True
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    location_user = None
    learning_rate, max_epochs, batch_size, iter_per_epoch = 6e-4, 100, 1024, 100
    param_training = (learning_rate, max_epochs, batch_size, iter_per_epoch, initial_run, save_model, input_flag)

    num_test = 10000
    channels, set_location_user = generate_channel(params_system, num_samples=num_test,
                                                   location_user_initial=location_user,
                                                   Rician_factor=Rician_factor)
    # file_path_channel = '../sum_rate_bcd/channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    # channel = sio.loadmat(file_path_channel)
    # channels = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])
    # set_location_user = channel['location_user']
    A_T, Hd, A_c = channel_complex2real(channels)

    set_len_pilot = len_pilots_tmp * num_user
    A_hat_set, H_hat_set = [], []
    train_loss_set, test_loss_set = [], []
    err_set=[]
    err_set_A=[]
    err_set_H=[]
    for len_pilot in set_len_pilot:
        path_pilots = './param_pilots_phaseshifts/phase_shifts_pilots' + str(params_system) + '_' + str(len_pilot) + '_' \
                      + str(Rician_factor) + '_' + str(noise_power_db) + '_' + str(input_flag) + '.mat'
        if initial_run:
            if len_pilot < (num_elements_irs + 1) * num_user:
                phase_shifts, pilots = generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user)
            else:
                phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)

            if save_model:
                sio.savemat(path_pilots, {'phase_shifts': phase_shifts, 'pilots': pilots})
        else:
            phase_shifts_pilots = sio.loadmat(path_pilots)
            phase_shifts, pilots = phase_shifts_pilots['phase_shifts'], phase_shifts_pilots['pilots']
        y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db,Pt=Pt_u)
        y_ks_tmp = decorrelation(y, pilots)
        y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
        y_mean, y_std = np.mean(y_real_tmp, axis=0), np.std(y_real_tmp, axis=0)
        y_real = (y_real_tmp - y_mean) / y_std
        location_mean, location_std = np.mean(set_location_user, axis=0), np.std(set_location_user, axis=0)
        set_location_user = (set_location_user - location_mean) / location_std
        y_ks_mean, y_ks_std = np.mean(y_ks, axis=0), np.std(y_ks, axis=0)
        y_ks_real = (y_ks - y_ks_mean) / y_ks_std

        # data_test = (y_real, A_T, Hd, set_location_user, y_mean, y_std, location_mean, location_std)
        data_test = (y_real, A_T, Hd, set_location_user, y_ks_real, y_mean, y_std,
                     location_mean, location_std, y_ks_mean, y_ks_std, channels)

        model_path = './param_model/model' + str(params_system) + '_' + str(len_pilot) + '_' + str(
            Rician_factor) + '_' + str(noise_power_db) + '_' + str(input_flag)
        nn, train_loss, test_loss = train_network(param_training, params_system, noise_power_db, phase_shifts, pilots,
                                                  data_test, model_path, location_user, Rician_factor=Rician_factor,
                                                  verbose=True)
        train_loss_set.append(train_loss)
        test_loss_set.append(test_loss)

        A_hat_real = nn.get_A(y_real, y_ks_real, set_location_user)
        A_hat_nn = A_hat_real[:, :, 0:num_antenna_bs, :] + 1j * A_hat_real[:, :, num_antenna_bs:2 * num_antenna_bs, :]
        A_hat_nn = np.transpose(A_hat_nn, (0, 2, 1, 3))
        A_hat_set.append(A_hat_nn)

        H_hat_real = nn.get_H(y_real, y_ks_real, set_location_user)
        H_nn = H_hat_real[:, 0:num_antenna_bs, :] + 1j * H_hat_real[:, num_antenna_bs:2 * num_antenna_bs, :]
        H_hat_set.append(H_nn)

        err_bs_user = np.linalg.norm(H_nn - channels[0], axis=(1)) ** 2 / np.linalg.norm(channels[0], axis=(1)) ** 2
        err_bs_user = np.mean(err_bs_user)
        err_bs_irs_user = np.linalg.norm(A_hat_nn-A_c, axis=(1, 2)) ** 2 /np.linalg.norm(A_c, axis=(1, 2)) ** 2
        err_bs_irs_user = np.mean(err_bs_irs_user)
        err = (err_bs_irs_user + err_bs_user)
        print('err:', err)

        err_set.append(err)

        sio.savemat('error_nn_vs_pilot.mat', {'err_set': err_set, 'err_set_A': err_set_A, 'err_set_H': err_set_H})
        sio.savemat('DNN_CE'+str(params_system) + '_' + str(len_pilot) + '_' + str(Rician_factor) + '_'
                    + str(noise_power_db) + '_' + str(input_flag)+'.mat',{'channel_bs_irs_user':A_hat_nn,'channel_bs_user':H_nn})


if __name__ == '__main__':
    ts = time.time()
    main_rate_vs_pilot(len_pilots_tmp=np.array([1,5,15,25,45,65,85]),
                       input_flag=0,
                       params_system=(8, 100, 3),
                       Rician_factor=10,
                       noise_power_db=-100,
                       Pt_u=15
                       )

    print('Running time: %0.3f hours: ' % ((time.time() - ts) / 3600))
