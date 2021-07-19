from compute_objective_fun import compute_rate_batch
import numpy as np
import scipy.io as sio

def bcd_beamforming_given_csi(file_path_channel, Pt_d, file_path_beamforming='bcd_perfect_csi.mat'):
    channel = sio.loadmat(file_path_channel)
    channel_true = (channel['channel_bs_user'], channel['channel_irs_user'], channel['channel_bs_irs'])

    beamforming_data = sio.loadmat(file_path_beamforming)
    w_bcd, theta_bcd = beamforming_data['w_all'], beamforming_data['theta_all']
    if len(w_bcd.shape) == 2:
        w_bcd = np.reshape(w_bcd, (w_bcd.shape[0], w_bcd.shape[1], 1))
    w_bcd = w_bcd.conjugate()
    # np.linalg.norm(w_bcd, axis=1) ** 2
    rate_sum, rate_all = compute_rate_batch(w_bcd, theta_bcd, channel_true,Pt=Pt_d)
    rate_sum_matlab = np.squeeze(beamforming_data['rate_perfect'])
    print('===Beamforming via BCD given csi:')
    print('Sum rate:', rate_sum, rate_sum_matlab)
    # print('All rate:', rate_all)
    return rate_sum, rate_all

def main():
    num_user = 3
    params_system = (16, 100, num_user)
    # params_system = (4, 60, num_user)
    Rician_factor = 10
    noise_power = -100
    Pt_u = 15
    Pt_d = 10

    file_path_channel = '../sum_rate_bcd/channel_data/channel' + str(params_system) + '_' + str(Rician_factor) + '.mat'
    file_path_perfectcsi = './results_data/bcd_perfect_csi_'+ str(params_system) + '_' + str(Rician_factor) + '_' + str(Pt_d) + '.mat'
    rate_perfectCSI, _ = bcd_beamforming_given_csi(file_path_channel, Pt_d, file_path_beamforming=file_path_perfectcsi)
    rate_imperfectCSI = []

    # len_pilot_set = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40])*num_user
    len_pilot_set = np.array([15])*num_user
    for len_pilot in len_pilot_set:
        len_pilot = len_pilot
        filename = './results_data/bcd_imperfect_csi_'+str(params_system)+'_'+str(Rician_factor)+'_'+str(len_pilot)+'_'\
                   +str(noise_power)+'_'+str(Pt_u)+'_'+str(Pt_d)+'.mat'
        tmp,_ = bcd_beamforming_given_csi(file_path_channel, Pt_d, file_path_beamforming=filename)
        rate_imperfectCSI.append(tmp)
    # sio.savemat('results_sumrate_bl'+str(params_system)+'_'+str(Rician_factor)+'_'+str(noise_power)+'_'+str(Pt_u)+'_'+str(Pt_d)+'.mat',
    #             {'len_pilot_set':len_pilot_set,'rate_perfectCSI':rate_perfectCSI,'rate_imperfectCSI':rate_imperfectCSI})


if __name__ == '__main__':
    main()
