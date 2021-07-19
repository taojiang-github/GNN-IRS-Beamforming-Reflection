import numpy as np
import scipy.io as sio
from sum_rate_bcd.generate_channel import generate_channel
from sum_rate_bcd.generate_received_pilots import generate_received_pilots_batch, channel_estimation_lmmse, channel_complex2real
from sum_rate_bcd.generate_received_pilots import compute_stat_info,generate_pilots_bl_v2,generate_pilots_bl
import os

def generate_testing_channel(params_system, Rician_factor, num_samples, file_path, location_user=None):
    channel_true, set_location_user = generate_channel(params_system, num_samples=num_samples,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channel_true
    # _, _, channel_bs_irs_user = channel_complex2real(channel_true)
    sio.savemat(file_path,
                {'channel_bs_user': channel_bs_user,
                 'channel_irs_user': channel_irs_user,
                 'channel_bs_irs': channel_bs_irs,
                 'location_user': set_location_user})


def main_channel_estimation(params_system, len_pilot, noise_power_db, Pt, location_user, Rician_factor, path):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    channel_true = sio.loadmat(path)
    channels = (channel_true['channel_bs_user'], channel_true['channel_irs_user'], channel_true['channel_bs_irs'])
    _, _, channel_bs_irs_user = channel_complex2real(channels)

    if len_pilot<(num_elements_irs+1)*num_user:
        phase_shifts, pilots = generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user)
    else:
        phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)

    y, _ = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, Pt=Pt)
    stat_info = compute_stat_info(params_system, noise_power_db, location_user, Rician_factor,num_samples=10000)
    channel_bs_user_est,channel_bs_irs_user_est = channel_estimation_lmmse(params_system,y,pilots,phase_shifts,stat_info)

    err_bs_user = np.linalg.norm(channel_bs_user_est - channel_true['channel_bs_user'], axis=(1)) ** 2 \
                  / np.linalg.norm(channel_true['channel_bs_user'],axis=(1)) ** 2
    err_bs_user = np.mean(err_bs_user)
    err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2)) ** 2 \
                      / np.linalg.norm(channel_bs_irs_user, axis=(1, 2)) ** 2
    err_bs_irs_user = np.mean(err_bs_irs_user)

    # print('Direct link estimation error (num_sample, num_user):\n', np.mean(err_bs_user))
    # print('Cascaded link estimation error (num_sample, num_user):\n', np.mean(err_bs_irs_user))
    print('Error_d: %0.3f'%err_bs_user, 'Error_r: %0.3f'%err_bs_irs_user, 'Error_sum: %0.3f'%(err_bs_user+err_bs_irs_user))

    path = path[0:-4]
    file_path = path + '_' + str(len_pilot) + '_' + str(noise_power_db) + '_'+str(Pt)+ '.mat'
    sio.savemat(file_path,
                {'channel_bs_user': channel_bs_user_est,
                 'channel_bs_irs_user': channel_bs_irs_user_est})

    err = np.linalg.norm(channel_bs_user_est - channel_true['channel_bs_user'], axis=(1)) ** 2 + np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2)) ** 2
    err = np.sum(err, axis=1)
    err_dB = 10 * np.log10(err)

    return file_path,err_bs_user,err_bs_irs_user, np.mean(err_dB)



def main():
    num_antenna_bs, num_elements_irs, num_user = 8, 100, 3
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    Rician_factor = 10
    location_user = None
    num_samples = 1000
    noise_power_db = -100
    Pt = 15

    file_path = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    if not os.path.exists(file_path):
        generate_testing_channel(params_system, Rician_factor, num_samples, file_path, location_user=location_user)
    else:
        print('Testing channel data exists')
    # generate_testing_channel(params_system, Rician_factor, num_samples, file_path, location_user=location_user)

    # len_pilots_set = np.array([1,5,15,25,35,55,75,95,105])*num_user
    len_pilots_set = np.array([15])*num_user

    err_bs_user_set,err_bs_irs_user_set,err_sum_set = [],[],[]
    err_dB_set = []
    for len_pilots in len_pilots_set:
        _, err_bs_user,err_bs_irs_user, err_dB = main_channel_estimation(params_system, len_pilots, noise_power_db, Pt, location_user, Rician_factor, file_path)
        err_bs_user_set.append(err_bs_user)
        err_bs_irs_user_set.append(err_bs_irs_user)
        err_sum_set.append(err_bs_user+err_bs_irs_user)
        err_dB_set.append(err_dB)
    sio.savemat('lmmse_error.mat', {'err_bs_user_set':err_bs_user_set,'err_bs_irs_user_set':err_bs_irs_user_set,'err_sum_set':err_sum_set})
    # sio.savemat('err_lmmse.mat', {'len_pilots_set':len_pilots_set,'err_dB_set':err_dB_set})


if __name__ == '__main__':
    main()
