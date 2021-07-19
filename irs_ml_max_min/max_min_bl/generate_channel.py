import numpy as np
import scipy.io as sio


def generate_location(num_users):
    location_user = np.empty([num_users, 3])
    for k in range(num_users):
        x = np.random.uniform(5, 15)
        y = np.random.uniform(-15, 15)
        z = -20
        coordinate_k = np.array([x, y, z])
        location_user[k, :] = coordinate_k
    return location_user


def path_loss_r(d):
    loss = 30 + 22.0 * np.log10(d)
    return loss


def path_loss_d(d):
    loss = 32.6 + 36.7 * np.log10(d)
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """

    num_user = location_user.shape[0]
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs)
        pathloss_irs_user.append(path_loss_r(d_k))
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k
        aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k
        aoa_irs_y.append(aoa_irs_y_k)
        aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user))
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z)
    return pathloss, aoa_aod


def generate_channel(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 10):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10))
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)
        channel_bs_user.append(tmp)

        # tmp: (num_antenna_bs,num_elements_irs) channel between IRS and BS
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh)
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh)
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z))
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate())
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs
        channel_bs_irs.append(tmp)

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, set_location_user


def channel_complex2real(channels):
    channel_bs_user, channel_irs_user, channel_bs_irs = channels
    (num_sample, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]

    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user])
    # Hd_real = np.zeros([num_sample, 2 * num_antenna_bs, num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex)

    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape(num_sample, 1, num_elements_irs)
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs
        A_tmp_tran = np.transpose(channel_combine_irs, (0, 2, 1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, A_tmp_tran.imag], axis=2)
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag, A_tmp_tran.real], axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real

    Hd_real = np.concatenate([channel_bs_user.real, channel_bs_user.imag], axis=1)

    return A_T_real, Hd_real, np.array(set_channel_combine_irs)


def main(num_user):
    num_test = 100
    num_antenna_bs, num_elements_irs = 4, 20
    params_system = (num_antenna_bs, num_elements_irs, num_user)
    Rician_factor = 10
    location_user = None

    channel_true, set_location_user = generate_channel(params_system,Rician_factor=Rician_factor,
                                                       num_samples=num_test)
    _, _, channel_bs_irs_user = channel_complex2real(channel_true)

    print('channel_bs_user:\n',np.mean(np.abs(channel_true[0])**2))
    print('channel_irs_user:\n',np.mean(np.abs(channel_true[1])**2))
    print('channel_bs_irs:\n',np.mean(np.abs(channel_true[2])**2))
    print('channel_bs_irs_user:\n',np.mean(np.abs(channel_bs_irs_user)**2))


if __name__ == '__main__':
    main(num_user=3)
