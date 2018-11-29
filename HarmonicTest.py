import numpy as np
import math
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib as mpl


def fiveD_data_FT(harmonic_number,array_data):

    # ======================================================
    # ------------------------------------------------------
    # Function description:
    # this function calculates G and S using the Spectral Phasor equation
    # INPUT: global array_data
    # OUTPUT: global G, S
    #
    # data structure for G and S
    # G(Time_Point, Z_position, G_values_x, G_values_y)
    # S(Time_Point, Z_position, G_values_x, G_values_y)
    # ======================================================

    # global G
    # global S

    # if harmonic has not been inserted
    spectral_channels = 32
    if harmonic_number == 0:
        print("Here is the box for harmonic_number.")
        return

    freq = harmonic_number / spectral_channels

    omega = 2 * math.pi * freq
    omega_delta_t = omega * 1 * 1

    # initializing matrices
    last_channel = spectral_channels - 1
    data_dim = array_data.ndim
    data_shape = array_data.shape

    # if there is zstack or timepoints
    if data_dim > 3:

        # data structure for G and S
        gs_shape = data_shape[:2] + data_shape[3:]
        G = np.zeros(gs_shape, dtype='float')
        S = np.zeros(gs_shape, dtype='float')

        spec = np.arange(last_channel + 1, dtype='float64')
        for time_point in range(0, data_shape[0]):
            for z_position in range(0, data_shape[1]):

                array_data = array_data[time_point, z_position, 0:last_channel + 1, :, :].astype('float64')

                Gn = np.cos(omega_delta_t * (spec + 1 - 0.5))
                Gn = np.tensordot(Gn, array_data, axes=([0], [0]))
                # Gn = Gn.sum(axis=0)
                Sn = np.sin(omega_delta_t * (spec + 1 - 0.5))
                Sn = np.tensordot(Sn, array_data, axes=([0], [0]))
                # Sn = Sn.sum(axis=0)
                area = array_data[1:last_channel].sum(axis=0) + 0.5 * array_data[(0, last_channel), :, :].sum(
                    axis=0)

                G[time_point, z_position, :, :] = Gn / area
                S[time_point, z_position, :, :] = Sn / area

    # if there is only one image
    if data_dim == 3:
        time_point = 0
        z_position = 0
        # data structure for G and S
        G = np.zeros(
            (1, 1, array_data.shape[1], array_data.shape[2]), dtype='float64')
        S = np.zeros(
            (1, 1, array_data.shape[1], array_data.shape[2]), dtype='float64')

        spec = np.arange(last_channel + 1, dtype='float64')

        array_data = array_data[0:last_channel + 1, :, :].astype('float64')

        Gn = np.cos(omega_delta_t * (spec + 1 - 0.5))
        Gn = np.tensordot(Gn, array_data, axes=([0], [0]))
        Sn = np.sin(omega_delta_t * (spec + 1 - 0.5))
        Sn = np.tensordot(Sn, array_data, axes=([0], [0]))
        area = array_data[1:last_channel].sum(axis=0) + 0.5 * array_data[(0, last_channel), :, :].sum(axis=0)

        G[time_point, z_position, :, :] = Gn / area
        S[time_point, z_position, :, :] = Sn / area


    # put all the NaN and g,s > 1 and g,s < -1 to -2,-2
    G[G > 1] = np.nan
    G[G < -1] = np.nan
    S[S > 1] = np.nan
    S[S < -1] = np.nan
    G[np.isnan(G)] = -2
    S[np.isnan(S)] = -2

    ##put all the NaN to 2,2
    # Variables_Module.G[np.isnan(Variables_Module.G)] = 2
    # Variables_Module.S[np.isnan(Variables_Module.S)] = 2
    return G,S


def channel2wavelength():
    # mat_contents = sio.loadmat(
    #     '/Users/Wen/Desktop/Fraser lab/PROJECTS/PhasorMaps/Wen_checkerboard/Franco Simulator Checker/fluorescein_literature_LSM780.mat')
    dir = '.'

    mat_contents = sio.loadmat(dir + '/fluorescein_literature_LSM780.mat')
    zeiss = mat_contents['zeiss_780_wav']
    # zeiss shape: (1,32)
    return zeiss


def gaussian_spectrum_generator(zeiss, peak_lambda, sigma):
    # w shape: (32,)
    spectrum = np.exp((-np.power(np.linalg.norm(zeiss - peak_lambda, axis=0), 2)) / (2 * sigma ** 2))
    spectrum = spectrum / np.sum(spectrum)
    return spectrum


def gspec_loop_generator(zeiss, sigma):
    all_spec = np.zeros((zeiss.shape[1],zeiss.shape[1]))
    for i in range(zeiss.shape[1]):
        all_spec[i,:] = gaussian_spectrum_generator(zeiss, zeiss[0,i], 10*sigma)
    return all_spec


def gspec_generator(zeiss, sigma):
    zeiss_zero = zeiss.ravel().astype('float64')
    all_spec = zeiss_zero[np.newaxis,:] - zeiss_zero[:,np.newaxis]
    all_spec = np.exp((-np.power(all_spec, 2) / (2*sigma**2)))
    all_spec = all_spec / np.sum(all_spec, axis=-1)[:,np.newaxis]
    return all_spec


def spectrum_FT(harmonic_number,spectrum,deltalambda):
    # if harmonic has not been inserted
    spectral_channels = spectrum.shape[-1]

    if harmonic_number == 0:
        print("no harmonic_number.")
        return

    freq = harmonic_number / spectral_channels
    omega = 2 * math.pi * freq

    # initializing matrices
    spec = np.arange(spectral_channels, dtype='float64')

    Gn = np.cos(omega * (spec + 1 - 0.5))
    Gn = np.tensordot(Gn, spectrum*deltalambda, axes=([-1], [-1]))
    Sn = np.sin(omega * (spec + 1 - 0.5))
    Sn = np.tensordot(Sn, spectrum*deltalambda, axes=([-1], [-1]))

    # area = (spectrum*deltalambda).sum()
    area = (spectrum*deltalambda).sum(axis=-1)
    # + 0.5 * spectrum[(0, last_channel), :, :].sum

    G = Gn / area
    S = Sn / area
    return G,S


def spectrum_FTv2(harmonic_number,spectrum):
    # if harmonic has not been inserted
    spectral_channels = spectrum.shape[-1]

    if harmonic_number == 0:
        print("no harmonic_number.")
        return

    freq = harmonic_number / spectral_channels
    omega = 2 * math.pi * freq

    # initializing matrices
    spec = np.arange(spectral_channels, dtype='float64')

    Gn = np.cos(omega * (spec + 1 - 0.5))
    Gn = np.tensordot(Gn, spectrum, axes=([-1], [-1]))
    Sn = np.sin(omega * (spec + 1 - 0.5))
    Sn = np.tensordot(Sn, spectrum, axes=([-1], [-1]))

    # area = (spectrum*deltalambda).sum()
    area = spectrum.sum(axis=-1)
    # + 0.5 * spectrum[(0, last_channel), :, :].sum

    G = Gn / area
    S = Sn / area
    return G,S


def test_wavelength(harmonic_number, cbox_index):
    zeiss = channel2wavelength()

    fig, phasor_axes = pre_plot()
    spectra_axes = fig.add_subplot(222)
    colorbar_axes = fig.add_subplot(223)

    deltalambda = zeiss[0,1]-zeiss[0,0]

    # cmap = plt.get_cmap('jet')
    cmap = mpl.cm.get_cmap('jet')

    # phasor_axes.set_color_cycle([cmap(i) for i in np.linspace(0, 1, zeiss.shape[1])])
    # phasor_axes.set_color_cycle([cmap(1. * i / zeiss.shape[1]) for i in range(zeiss.shape[1])])

    colors = [cmap(1. * i / zeiss.shape[1]) for i in range(zeiss.shape[1])]
    for sigma in range(3,4):
        G, S = [], []
        for wavelength in range(0, zeiss.shape[1]):
        # for wavelength in range(14, 15):

            spectrum = gaussian_spectrum_generator(zeiss, zeiss[0, wavelength], 10*sigma)

            g,s = spectrum_FT(harmonic_number,spectrum,deltalambda)

        # if wavelength % 4 == 0 and sigma%2 == 0:
            #     spectra_axes.plot(zeiss[0, :],spectrum,color= colors[wavelength],linestyle='solid')
            spectra_axes.plot(zeiss[0, :],spectrum, color=colors[wavelength], linestyle='solid')

            phasor_axes.plot(g, s, color=colors[wavelength], linestyle='solid', marker='o', markersize=3)

            G = np.append(G, g)
            S = np.append(S, s)

        phasor_axes.plot(G, S, 'k--', linewidth=0.5)

        # for wavelength in range(0, zeiss.shape[1]):
        #     phasor_axes.plot(G, S, color=colors[wavelength], linestyle='solid', marker='o', markersize=3)
    
    G_bin, S_bin = convert_gs_binind(G, S, 512)
    SEER_G = convert_gs_rgb(G_bin, S_bin, cbox_index)
    
    SEER_G = SEER_G[:,np.newaxis,:].repeat(3,axis=1)
    colorbar_axes.imshow(SEER_G)
    
    # colorbar_axes.imshow(SEER_G.reshape(SEER_G.shape[0],1,SEER_G.shape[1]))

    phasor_axes.set_title('Phasor')
    phasor_axes.set_aspect(1)
    spectra_axes.set_title('Spectra')
    colorbar_axes.set_title('Colorbar')
    return fig
    # plt.title('harmonic' + str(harmonic_number))


def colorimpaired_cmap(rad):
    myCmap = np.ones((rad.shape + (3,)), dtype='float')
    sideLen = np.float16(1/11)
    myCmap[np.logical_and(rad >= 0, rad < sideLen),:] = (127, 59, 8)
    myCmap[np.logical_and(rad >= sideLen, rad < 2 * sideLen), :] = (179, 88, 6)
    myCmap[np.logical_and(rad >= 2 * sideLen, rad < 3 * sideLen), :] = (224, 130, 20)
    myCmap[np.logical_and(rad >= 3 * sideLen, rad < 4 * sideLen), :] = (253, 184, 99)
    myCmap[np.logical_and(rad >= 4 * sideLen, rad < 5 * sideLen), :] = (254, 224, 182)
    myCmap[np.logical_and(rad >= 5 * sideLen, rad < 6 * sideLen), :] = (247, 247, 247)
    myCmap[np.logical_and(rad >= 6 * sideLen, rad < 7 * sideLen), :] = (216, 218, 235)
    myCmap[np.logical_and(rad >= 7 * sideLen, rad < 8 * sideLen), :] = (178, 171, 210)
    myCmap[np.logical_and(rad >= 8 * sideLen, rad < 9 * sideLen), :] = (128, 115, 172)
    myCmap[np.logical_and(rad >= 9 * sideLen, rad < 10 * sideLen), :] = (84, 39, 136)
    myCmap[np.logical_and(rad >= 10 * sideLen, rad < 11 * sideLen), :] = (45, 0, 75)

    myCmap = myCmap/255
    return myCmap



def convert_gs_binind(g,s,bins):
    sq_bin_side = (1 - -1)/bins
    ind_last = int(bins - 1)
    g_binind = np.floor(g / sq_bin_side) + np.ceil(bins/2)
    g_binind = g_binind.astype('int')
    s_binind = np.floor(s / sq_bin_side) + np.ceil(bins/2)
    s_binind = s_binind.astype('int')
    g_binind[g_binind == int(bins)] = ind_last
    s_binind[s_binind == int(bins)] = ind_last
    s_binind *= -1
    s_binind += ind_last
    
    return g_binind, s_binind


def convert_gs_rgb(g,s, cbox_index=3):
    bins = 512  # VM.bins
    rows = s
    cols = g
    rows = bins / 2 - rows
    cols = cols - bins / 2
    
    rad = np.minimum(np.sqrt(np.power(rows, 2) + np.power(cols, 2)) / (bins / 2), 1)
    theta = np.arctan2(rows, cols)
    theta[theta < 0] += 2 * np.pi
    theta /= 2 * np.pi

    if cbox_index == 3:
        hue = theta
        val = 1 - 0.85 * rad
        sat = np.ones_like(val)

    elif cbox_index == 4:
        hue = theta
        val = rad
        sat = np.ones_like(val)

    elif cbox_index == 5:
        # radius
        tmp = mpl.cm.jet(rad)
        tmp = tmp[:, 0:3]

    elif cbox_index == 6:
        # angle
        hue = theta
        val = np.ones_like(hue)
        sat = np.ones_like(val)
    
    elif cbox_index == 7:
        # colorimpaired
        tmp = colorimpaired_cmap(theta)

    if not (cbox_index == 5 or cbox_index == 7):
        tmp = np.stack((hue, sat, val), axis=-1)
        # tmp = tmp[:,np.newaxis,:].repeat(3,axis=1)
        tmp = hsv_to_rgb(tmp)

    G = tmp

    return G


def create_SEERphasor(cbox_index=3):
    
    G = convert_gs_rgb()



def generate_colorbar(ax, harmonic_number, cbox_index):
    zeiss = channel2wavelength()
    spectrum = gspec_generator(zeiss, 30)
    G, S = spectrum_FTv2(harmonic_number, spectrum)
    G_bin, S_bin = convert_gs_binind(G,S,512)
    rgbarray = convert_gs_rgb(G_bin, S_bin, cbox_index=cbox_index)
    cmap = mpl.colors.ListedColormap(rgbarray, name='SEER'+str(cbox_index))
    norm = mpl.colors.Normalize(vmin=zeiss[0,0], vmax=zeiss[0,-1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical',
                                    ticklocation='left')
    cb1.set_label('Wavelength')
    ax.yaxis.set_label_position('left')
    ax.set_aspect(3)
    return cb1


def generate_colorbarv2(ax, wavelength_range, harmonic_number, cbox_index):
    zeiss = np.linspace(wavelength_range[0], wavelength_range[1],100)
    spectrum = gspec_generator(zeiss, 30)
    G, S = spectrum_FTv2(harmonic_number, spectrum)
    G_bin, S_bin = convert_gs_binind(G,S,512)
    rgbarray = convert_gs_rgb(G_bin, S_bin, cbox_index=cbox_index)
    cmap = mpl.colors.ListedColormap(rgbarray, name='SEER'+str(cbox_index))
    norm = mpl.colors.Normalize(vmin=wavelength_range[0], vmax=wavelength_range[-1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical',
                                    ticklocation='left')
    cb1.set_label('Wavelength')
    ax.yaxis.set_label_position('left')
    ax.set_aspect(3)
    return cb1


def pre_plot():

    fig = plt.figure()
    phasor_axes = fig.add_subplot(221)

    circ = plt.Circle((0, 0), radius=1, axes = phasor_axes,facecolor='none', edgecolor='black')
    circ.set_linestyle('dashed')

    phasor_axes.add_artist(circ)

    # plot the lines
    phasor_axes.plot([-1, 1], [-1, 1], 'k--')
    phasor_axes.plot([-1, 1], [1, -1], 'k--')
    phasor_axes.plot([-1, 1], [0, 0], 'k--')
    phasor_axes.plot([0, 0], [-1, 1], 'k--')

    return fig,phasor_axes

if __name__ == '__main__':
    # test_wavelength(1)
    tis = ['Gradient Descent', 'Gradient Ascent', 'Radius', 'Angle']
    for ind in [3,4,5,6]:
        fig = test_wavelength(2,ind)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        colorbar = generate_colorbarv2(ax1, [400, 700], 2, ind)
        ax1.set_title(tis[ind-3])
        fig.suptitle(tis[ind-3])
        fig.savefig(tis[ind-3]+'_spectrum.png')
        fig1.savefig(tis[ind-3]+'_colorbar.png')
    
    plt.show()
