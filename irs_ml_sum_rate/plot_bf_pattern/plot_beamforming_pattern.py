from plot_bf_pattern.sumrate_model import SumRateModel
from plot_bf_pattern.generate_channel import generate_channel, channel_complex2real
from plot_bf_pattern.generate_received_pilots import generate_received_pilots_batch,channel_complex2real,decorrelation,generate_pilots_bl
from compute_objective_fun import compute_rate_batch
from scipy.linalg import dft
import numpy as np
import scipy.io as sio
import time,os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_nn_model(params_system,len_pilot, Pt, model_path, input_flag):
    (num_antenna_bs, num_elements_irs, num_user) = params_system

    nn = SumRateModel(num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, Pt)
    nn.create_network()
    # nn.create_optimizer(training_algorithm='Adam')
    nn.create_initializer()
    initial_run = False
    nn.initialize(initial_run, model_path)

    return nn

def generate_spatial_signature(aoa_aod, num_antenna_bs, num_elements_irs,irs_Nh, cos_theta_bs=1 ):
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod
    a_bs = np.exp(1j * np.pi * aoa_bs *cos_theta_bs * np.arange(num_antenna_bs))
    i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
    i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
    a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
    a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y + i2 * aoa_irs_z))

    return a_bs.reshape((-1,1)), a_irs_bs.reshape((-1,1)), a_irs_user.reshape((-1,1))

def compute_array_response_irs(a_bs_irs, a_irs_user, v):
    tmp = v*a_bs_irs.conjugate()
    tmp = tmp.reshape((1,-1))@a_irs_user
    return np.abs(tmp)

def compute_array_response_bs(a_bs, w):
    tmp = w.reshape((1,-1))@a_bs
    return np.abs(tmp)

def plot_array_response_irs(v, params_system, aoa_aod_true, phi_true, irs_cos_z_true, irs_Nh, num_points):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod_true
    (aod_irs_cos_z,aoa_irs_cos_z) = irs_cos_z_true
    (phi_1_true, phi_2_true, phi_3_true) = phi_true

    phi_2_set = np.block([np.linspace(-np.pi/2, np.pi/2, num_points),phi_2_true])
    phi_2_set = np.sort(phi_2_set)
    phi_3_set = np.block([np.linspace(-np.pi/2, np.pi/2, num_points), phi_3_true])
    phi_3_set = np.sort(phi_3_set)

    array_response = np.zeros((num_points+1, num_points+1))

    for ii in range(num_points+1):
        phi_2 = phi_2_set[ii]
        aod_irs_y_i = np.sin(phi_2)*aod_irs_cos_z
        # aod_irs_y_i = np.sin(phi_2_true)*aod_irs_cos_z
        for jj in range(num_points+1):
            phi_3 = phi_3_set[jj]
            aoa_irs_y_j = np.sin(phi_3)*aoa_irs_cos_z
            # aoa_irs_y_j = np.sin(phi_3_true)*aoa_irs_cos_z

            aoa_aod = (aoa_bs, aod_irs_y_i, aod_irs_z, aoa_irs_y_j, aoa_irs_z)
            _, a_bs_irs, a_irs_user = generate_spatial_signature(aoa_aod, num_antenna_bs, num_elements_irs, irs_Nh)
            array_response[ii,jj] = compute_array_response_irs(a_bs_irs, a_irs_user, v)

    #--- plot
    plt.imshow(array_response, cmap='viridis', interpolation='nearest',extent = [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2], aspect = 'auto')
    plt.colorbar()
    plt.xlabel('phi_2')
    plt.ylabel('phi_3')
    plt.show()
    print('phi_true',phi_true)
    return array_response,phi_2_set,phi_3_set


def plot_array_response_irs_fix_phi2(v, params_system, aoa_aod_true, phi_true, irs_cos_z_true, irs_Nh, num_points):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod_true
    (aod_irs_cos_z,aoa_irs_cos_z) = irs_cos_z_true
    (phi_1_true, phi_2_true, phi_3_true) = phi_true

    theta_3_set = np.block([np.linspace(-np.pi/2, np.pi/2, num_points),-np.arccos(aoa_irs_cos_z)])
    theta_3_set = np.sort(theta_3_set)
    phi_3_set = np.block([np.linspace(-np.pi/2, np.pi/2, num_points), phi_3_true])
    phi_3_set = np.sort(phi_3_set)

    array_response = np.zeros((num_points+1, num_points+1))
    phi_2 = phi_2_true
    aod_irs_y_i = np.sin(phi_2) * aod_irs_cos_z
    for ii in range(num_points+1):
        theta_3 = theta_3_set[ii]
        aoa_irs_cos_z_i=np.cos(theta_3)
        aoa_irs_z =np.sin(theta_3)
        for jj in range(num_points+1):
            phi_3 = phi_3_set[jj]
            aoa_irs_y_j = np.sin(phi_3)*aoa_irs_cos_z_i
            # aoa_irs_y_j = np.sin(phi_3_true)*aoa_irs_cos_z
            aoa_aod = (aoa_bs, aod_irs_y_i, aod_irs_z, aoa_irs_y_j, aoa_irs_z)
            _, a_bs_irs, a_irs_user = generate_spatial_signature(aoa_aod, num_antenna_bs, num_elements_irs, irs_Nh)
            array_response[ii,jj] = compute_array_response_irs(a_bs_irs, a_irs_user, v)

    #--- plot
    plt.imshow(array_response, cmap='viridis', interpolation='nearest',extent = [-np.pi, np.pi, -np.pi, np.pi], aspect = 'auto')
    plt.colorbar()
    plt.xlabel('phi_2')
    plt.ylabel('phi_3')
    plt.show()
    print('phi_true',phi_true)
    return array_response,theta_3_set, phi_3_set


def plot_array_response_bs(w, params_system, aoa_aod_true, phi_true, irs_cos_z_true, irs_Nh, num_points):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod_true
    (aod_irs_cos_z,aoa_irs_cos_z) = irs_cos_z_true
    (phi_1_true, phi_2_true, phi_3_true) = phi_true

    phi_1_set = np.block([np.linspace(0, np.pi, num_points),phi_1_true])
    phi_1_set = np.sort(phi_1_set)


    array_response = np.zeros((num_points+1,))

    for ii in range(num_points+1):
        phi_1 = phi_1_set[ii]
        aoa_bs_i = np.cos(phi_1)
        aoa_aod = (aoa_bs_i, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
        a_bs, a_bs_irs, a_irs_user = generate_spatial_signature(aoa_aod, num_antenna_bs, num_elements_irs, irs_Nh)
        array_response[ii] = compute_array_response_bs(a_bs, w)

    #--- plot
    plt.plot(phi_1_set,array_response)
    plt.xlabel('phi_1')
    plt.ylabel('Array response')
    plt.show()
    print('phi_true',phi_true)
    return array_response,phi_1_set

def plot_array_response_bs_2d(w, params_system, aoa_aod_true, phi_true, irs_cos_z_true, irs_Nh, num_points):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod_true
    (aod_irs_cos_z,aoa_irs_cos_z) = irs_cos_z_true
    (phi_1_true, phi_2_true, phi_3_true) = phi_true

    phi_1_set = np.block([np.linspace(0, np.pi, num_points),phi_1_true])
    phi_1_set = np.sort(phi_1_set)
    theta_1_set = np.block([np.linspace(-np.pi/2, np.pi/2, num_points),0])
    theta_1_set = np.sort(theta_1_set)

    array_response = np.zeros((num_points+1,num_points+1))

    for ii in range(num_points+1):
        phi_1 = phi_1_set[ii]
        aoa_bs_i = np.cos(phi_1)
        for jj in range(num_points+1):
            aoa_bs_z = np.cos(theta_1_set[jj])
            aoa_aod = (aoa_bs_i, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
            a_bs, a_bs_irs, a_irs_user = generate_spatial_signature(aoa_aod, num_antenna_bs, num_elements_irs, irs_Nh,cos_theta_bs=aoa_bs_z)
            array_response[ii,jj] = compute_array_response_bs(a_bs, w)

    return array_response,phi_1_set,theta_1_set


def run_1():
    params_system = (8, 100, 1)
    (num_antenna_bs,num_elements_irs,num_user) = params_system
    Rician_factor = 10
    location_user = np.array([[30, 20, -20]])
    input_flag = 0
    len_pilot = 25
    noise_power_db = -100
    Pt_u = 15
    Pt_d = 0
    path_pilots = './param_pilots_phaseshifts/phase_shifts_pilots' + str(params_system) + '_' + str(len_pilot) + '_' \
        + str(Rician_factor) + '_' + str(noise_power_db) + '_' + str(Pt_u) + '_' + str(Pt_d) + '_' + str(input_flag) + '.mat'
    phase_shifts_pilots = sio.loadmat(path_pilots)
    phase_shifts, pilots = phase_shifts_pilots['phase_shifts'], phase_shifts_pilots['pilots']

    #---compute mean-----
    channels, set_location_user = generate_channel(params_system, num_samples=10000, location_user_initial=None,
                                                   Rician_factor=Rician_factor)
    y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
    y_ks_tmp = decorrelation(y, pilots)
    y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
    y_mean, y_std = np.mean(y_real_tmp, axis=0), np.std(y_real_tmp, axis=0)
    y_ks_mean, y_ks_std = np.mean(y_ks, axis=0), np.std(y_ks, axis=0)
    location_mean, location_std = np.mean(set_location_user, axis=0), np.std(set_location_user, axis=0)

    #---generate channel----
    channels, set_location_user = generate_channel(params_system, num_samples=1, location_user_initial=location_user,
                                                   Rician_factor=Rician_factor)
    y, y_real_tmp = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db,Pt=Pt_u)
    y_ks_tmp = decorrelation(y, pilots)
    y_ks = np.concatenate([y_ks_tmp.real, y_ks_tmp.imag], axis=1)
    y_real = (y_real_tmp - y_mean) / y_std
    y_ks_real = (y_ks - y_ks_mean) / y_ks_std
    set_location_user = (set_location_user - location_mean) / (location_std+1e-15)

    # ----------------------------------neural network beamforming--------------------------
    model_path = './param_model/model' + str(params_system) + '_' + str(len_pilot) + '_' + str(
        Rician_factor) + '_' + str(noise_power_db) + '_' + str(Pt_u) + '_' + str(Pt_d) + '_' + str(input_flag)

    nn = load_nn_model(params_system,len_pilot, Pt_d, model_path, input_flag)
    w_nn_real = nn.get_w(y_real, y_ks_real, set_location_user)
    w_nn = w_nn_real[:, 0:num_antenna_bs, :] + 1j * w_nn_real[:, num_antenna_bs:2 * num_antenna_bs, :]
    theta_nn_real = nn.get_theta(y_real, y_ks_real, set_location_user)
    theta_nn = theta_nn_real[:, 0:num_elements_irs] + 1j * theta_nn_real[:, num_elements_irs:2 * num_elements_irs]
    rate_nn, rate_nn_all = compute_rate_batch(w_nn, theta_nn, channels,Pt=Pt_d)

    print('\n===Simulation parameters====')
    print('params_system:', params_system)
    print('Rician_factor, noise_power_db:', Rician_factor, noise_power_db)
    print('len_pilot:', len_pilot)
    print('input_flag:', input_flag)
    print('===Simulation results===')
    print('rate %0.3f:' % rate_nn)
    print('rate per user', rate_nn_all)
    # print('rate_min_all', rate_min_all)
    print('=============================\n')

    location_irs = np.array([0, 0, 0])
    location_bs = np.array([100, -100, 0])
    d0 = np.linalg.norm(location_bs - location_irs)
    aoa_bs = (location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    d_k = np.linalg.norm(location_user - location_irs)
    aoa_irs_y_k = (location_user[0][1] - location_irs[1]) / d_k
    aoa_irs_z_k = (location_user[0][2] - location_irs[2]) / d_k
    # print('aoa_irs_y_k','aoa_irs_z_k',aoa_irs_y_k,aoa_irs_z_k)
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y_k, aoa_irs_z_k)
    phi_1 = np.arccos((location_irs[0] - location_bs[0])/np.linalg.norm(location_bs[0:2]-location_irs[0:2]))
    phi_2 = np.arcsin((location_bs[1]-location_irs[1])/np.linalg.norm(location_bs[0:2]-location_irs[0:2]))
    phi_3 = np.arcsin((location_user[0][1]-location_irs[1])/np.linalg.norm(location_user[0][0:2]-location_irs[0:2]))
    phi = (phi_1,phi_2,phi_3)
    aod_irs_cos = np.linalg.norm(location_bs[0:2]-location_irs[0:2])/d0
    aoa_irs_cos = np.linalg.norm(location_user[0][0:2]-location_irs[0:2])/d_k
    irs_cos_z = (aod_irs_cos,aoa_irs_cos)
    irs_theta = (np.arccos(aod_irs_cos),-np.arccos(aoa_irs_cos))

    array_response_irs1,set_phi_2,set_phi_3 = plot_array_response_irs(theta_nn, params_system, aoa_aod, phi, irs_cos_z, irs_Nh=10,num_points=1000)
    array_response_irs2,set_theta_3,set_phi_3_2 = plot_array_response_irs_fix_phi2(theta_nn, params_system, aoa_aod, phi, irs_cos_z, irs_Nh=10,num_points=1000)
    sio.savemat('array_response_irs'+str(location_user)+ '.mat',{'array_response_irs1':array_response_irs1,
                                                                 'array_response_irs2': array_response_irs2,
                                                                  'set_phi_2':set_phi_2,
                                                                  'set_phi_3':set_phi_3,
                                                                  'set_phi_3_2': set_phi_3_2,
                                                                  'set_theta_3':set_theta_3,
                                                                  'aoa_aod':aoa_aod,
                                                                  'phi_true':phi,
                                                                  'irs_theta':irs_theta})

    array_response_bs, set_phi_1 = plot_array_response_bs(w_nn,params_system,aoa_aod, phi,irs_cos_z, irs_Nh=10,num_points=1000)
    array_response_bs_2d, set_phi_1_2d, set_theta_1_2d = plot_array_response_bs_2d(w_nn,params_system,aoa_aod, phi,irs_cos_z, irs_Nh=10,num_points=1000)

    sio.savemat('array_response_bs'+str(location_user)+ '.mat',{'array_response_bs':array_response_bs,
                                      'set_phi_1':set_phi_1,
                                      'phi_true':phi,
                                      'array_response_bs_2d':array_response_bs_2d,
                                      'set_phi_1_2d':set_phi_1_2d,
                                      'set_theta_1_2d':set_theta_1_2d
                                      })

if __name__ == '__main__':
    ts = time.time()
    run_1()
    print('Running time: %0.3f sec: ' % ((time.time() - ts)))
