import numpy
import math
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib


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
        G = numpy.zeros(gs_shape, dtype='float')
        S = numpy.zeros(gs_shape, dtype='float')

        spec = numpy.arange(last_channel + 1, dtype='float64')
        for time_point in range(0, data_shape[0]):
            for z_position in range(0, data_shape[1]):

                array_data = array_data[time_point, z_position, 0:last_channel + 1, :, :].astype('float64')

                Gn = numpy.cos(omega_delta_t * (spec + 1 - 0.5))
                Gn = numpy.tensordot(Gn, array_data, axes=([0], [0]))
                # Gn = Gn.sum(axis=0)
                Sn = numpy.sin(omega_delta_t * (spec + 1 - 0.5))
                Sn = numpy.tensordot(Sn, array_data, axes=([0], [0]))
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
        G = numpy.zeros(
            (1, 1, array_data.shape[1], array_data.shape[2]), dtype='float64')
        S = numpy.zeros(
            (1, 1, array_data.shape[1], array_data.shape[2]), dtype='float64')

        spec = numpy.arange(last_channel + 1, dtype='float64')

        array_data = array_data[0:last_channel + 1, :, :].astype('float64')

        Gn = numpy.cos(omega_delta_t * (spec + 1 - 0.5))
        Gn = numpy.tensordot(Gn, array_data, axes=([0], [0]))
        Sn = numpy.sin(omega_delta_t * (spec + 1 - 0.5))
        Sn = numpy.tensordot(Sn, array_data, axes=([0], [0]))
        area = array_data[1:last_channel].sum(axis=0) + 0.5 * array_data[(0, last_channel), :, :].sum(axis=0)

        G[time_point, z_position, :, :] = Gn / area
        S[time_point, z_position, :, :] = Sn / area


    # put all the NaN and g,s > 1 and g,s < -1 to -2,-2
    G[G > 1] = numpy.nan
    G[G < -1] = numpy.nan
    S[S > 1] = numpy.nan
    S[S < -1] = numpy.nan
    G[numpy.isnan(G)] = -2
    S[numpy.isnan(S)] = -2

    ##put all the NaN to 2,2
    # Variables_Module.G[numpy.isnan(Variables_Module.G)] = 2
    # Variables_Module.S[numpy.isnan(Variables_Module.S)] = 2
    return G,S


def channel2wavelength():
    # mat_contents = sio.loadmat(
    #     '/Users/Wen/Desktop/Fraser lab/PROJECTS/PhasorMaps/Wen_checkerboard/Franco Simulator Checker/fluorescein_literature_LSM780.mat')
    dir = 'E:/ImageProcessing/TestData'

    mat_contents = sio.loadmat(dir + '/fluorescein_literature_LSM780.mat')
    zeiss = mat_contents['zeiss_780_wav']
    # zeiss shape: (1,32)
    return zeiss


def gaussian_spectrum_generator(zeiss, peak_lambda, sigma):
    # w shape: (32,)
    spectrum = numpy.exp((-numpy.power(numpy.linalg.norm(zeiss - peak_lambda, axis=0), 2)) / (2 * sigma ** 2))
    spectrum = spectrum / numpy.sum(spectrum)
    return spectrum


def spectrum_FT(harmonic_number,spectrum,deltalambda):
    # if harmonic has not been inserted
    spectral_channels = spectrum.shape[0]

    if harmonic_number == 0:
        print("no harmonic_number.")
        return

    freq = harmonic_number / spectral_channels
    omega = 2 * math.pi * freq

    # initializing matrices
    spec = numpy.arange(spectral_channels, dtype='float64')

    Gn = numpy.cos(omega * (spec + 1 - 0.5))
    Gn = numpy.tensordot(Gn, spectrum*deltalambda, axes=([0], [0]))
    Sn = numpy.sin(omega * (spec + 1 - 0.5))
    Sn = numpy.tensordot(Sn, spectrum*deltalambda, axes=([0], [0]))

    area = (spectrum*deltalambda).sum()
    # + 0.5 * spectrum[(0, last_channel), :, :].sum

    G = Gn / area
    S = Sn / area
    return G,S


def test_wavelength(harmonic_number):
    zeiss = channel2wavelength()


    fig, phasor_axes = pre_plot()
    spectra_axes = fig.add_subplot(222)
    colorbar_axes = fig.add_subplot(223)

    deltalambda = zeiss[0,1]-zeiss[0,0]

    # cmap = plt.get_cmap('jet')
    cmap = matplotlib.cm.get_cmap('jet')

    # phasor_axes.set_color_cycle([cmap(i) for i in numpy.linspace(0, 1, zeiss.shape[1])])
    # phasor_axes.set_color_cycle([cmap(1. * i / zeiss.shape[1]) for i in range(zeiss.shape[1])])


    colors = [cmap(1. * i / zeiss.shape[1]) for i in range(zeiss.shape[1])]
    for sigma in range(3,4):
        G ,S = [], []
        for wavelength in range(0, zeiss.shape[1]):
        # for wavelength in range(14, 15):

            spectrum= gaussian_spectrum_generator(zeiss, zeiss[0, wavelength], 10*sigma)

            g,s = spectrum_FT(harmonic_number,spectrum,deltalambda)

        # if wavelength % 4 == 0 and sigma%2 == 0:
            #     spectra_axes.plot(zeiss[0, :],spectrum,color= colors[wavelength],linestyle='solid')
            spectra_axes.plot(zeiss[0, :],spectrum,color= colors[wavelength],linestyle='solid')

            phasor_axes.plot(g, s,color= colors[wavelength],linestyle='solid', marker='o',markersize=3)

            G = numpy.append(G, g)
            S = numpy.append(S, s)

        phasor_axes.plot(G, S, 'k--', linewidth=0.5)

        # for wavelength in range(0, zeiss.shape[1]):
        #     phasor_axes.plot(G, S, color=colors[wavelength], linestyle='solid', marker='o', markersize=3)

    G = convert_gs_rgb(G,S)
    colorbar_axes.imshow(G.reshape(G.shape[0],1,G.shape[1]))

    phasor_axes.set_title('phasor')
    spectra_axes.set_title('spectra')
    # plt.title('harmonic' + str(harmonic_number))


def convert_gs_rgb(g,s, cbox_index=3):
    bins = 512  # VM.bins

    rows = s
    cols = g

    if cbox_index == 3:
        rad = numpy.minimum(numpy.sqrt(numpy.power(rows, 2) + numpy.power(cols, 2)) / (bins / 2), 1)
        theta = numpy.arctan2(rows, cols)
        theta[theta < 0] += 2 * numpy.pi
        theta /= 2 * numpy.pi

        hue = theta
        val = 1 - 0.85 * rad
        sat = numpy.ones_like(val)

        G = numpy.stack((hue, sat, val), axis=-1)
        G = hsv_to_rgb(G)

    return G

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


test_wavelength(1)
test_wavelength(2)
plt.show()
