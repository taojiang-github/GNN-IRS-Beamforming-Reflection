from compute_objective_fun import compute_rate_batch
import numpy as np
import scipy.io as sio

def bcd_beamforming_given_csi(file_path_channel, file_path_beamforming='bcd_perfect_csi.mat'):
    channel = sio.loadmat(file_path_channel)
    channel_true = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])

    beamforming_data = sio.loadmat(file_path_beamforming)
    w_bcd, theta_bcd = beamforming_data['w_all'], beamforming_data['theta_all']
    if len(w_bcd.shape) == 2:
        w_bcd = np.reshape(w_bcd, (w_bcd.shape[0], w_bcd.shape[1], 1))
    w_bcd = w_bcd.conjugate()
    rate_sum, rate_all = compute_rate_batch(w_bcd, theta_bcd, channel_true,Pt=5)
    rate_sum_matlab = np.squeeze(beamforming_data['rate_perfect'])
    print('===Beamforming via BCD given csi:')
    print('Sum rate:', rate_sum, rate_sum_matlab)
    # print('All rate:', rate_all)
    return rate_sum, rate_all

def main():
    num_user = 3
    params_system = (8,100,num_user)
    Rician_factor = 10
    noise_power = -100
    file_path_channel =   '../sum_rate_bcd/channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    # file_path_channel = './channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    # rate_perfectCSI, _ = bcd_beamforming_given_csi(file_path_channel, file_path_beamforming='bcd_perfect_csi.mat')
    rate_imperfectCSI = []
    len_pilot_set = np.array([1,3,5,10,20,25,40])*num_user

    # len_pilot_set = np.array([6,10,14,18,22,26,30,60,90,120])*num_user
    for len_pilot in len_pilot_set:
        len_pilot = len_pilot
        filename = 'bcd_imperfect_csi_'+str(params_system)+'_'+str(Rician_factor)+'_'+str(len_pilot)+'_'+str(noise_power)+'.mat'
        tmp,_=bcd_beamforming_given_csi(file_path_channel, file_path_beamforming=filename)
        rate_imperfectCSI.append(tmp)
    sio.savemat('results_sumrate_dnnce'+str(params_system)+'_'+str(Rician_factor)+'_'+str(noise_power)+'.mat',
                {'len_pilot_set':len_pilot_set,'rate_imperfectCSI':rate_imperfectCSI})

if __name__ == '__main__':
    main()
