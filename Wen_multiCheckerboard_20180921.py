import numpy
import tifffile as tf
import scipy.io as sio
from matplotlib import pyplot as plt



class Wen_checkerboard:

    def directory_check(self, ifWorkstation=True):
        if ifWorkstation == True:
            dir = 'E:/ImageProcessing/TestData'
        else:
            dir = '/Users/Wen/Desktop/Fraser lab/PROJECTS/PhasorMaps/Wen_checkerboard/Franco Simulator Checker'
        return dir



    def read_files_lsm(self, dir, filename):
        # read lsm files
        # dim_str = 'TZCYX'
        G_hyper = tf.TiffFile(dir+filename).asarray()
        self.G_hyper = G_hyper[:, :, 0:32, :, :]
        self.G_hyper = G_hyper[0:1, 0:1, 0:32, :, :]
        print('data_array shape: ' + str(self.G_hyper.shape))  # (1, 1, 32, 40, 179)
        mat_contents = sio.loadmat(dir + '/fluorescein_literature_LSM780.mat')
        self.zeiss = mat_contents['zeiss_780_wav']
        print(self.zeiss)


    def read_files_checkerboard(self, dir):
        # read lsm files
        filename1 = dir + '/140917CFPa_Subset_fingerprinting_Subset2.lsm'
        filename2 = dir + '/140917YFPb_Subset_Subset2.lsm'
        filename3 = dir + '/140917RFPa_Subset_Subset2.lsm'
        mat_contents = sio.loadmat(dir+'/fluorescein_literature_LSM780.mat')

        # dim_str = 'TZCYX'
        self.data_CFP = tf.TiffFile(filename1).asarray()
        self.data_YFP = tf.TiffFile(filename2).asarray()
        self.data_RFP = tf.TiffFile(filename3).asarray()
        self.zeiss = mat_contents['zeiss_780_wav']
        print(self.zeiss)




    def save_2_tiff(self, output_data, filename):
        if output_data.ndim ==3:
            output_data = output_data[numpy.newaxis, numpy.newaxis, :, :, :]

        for t in range(output_data.shape[0]):
            for z in range(output_data.shape[1]):
                for ch in range(0, output_data.shape[2]):
                    newfilename = filename + '-t'+(str(t)) + '-z'+(str(z)) + '-ch' + (str(ch)) + '.tif'
                    tf.imsave(newfilename, numpy.uint16(output_data[t, z, ch, :, :]))



    def generate_1_pixel_spectrum_avg(self, number_of_pixels, data):
        spectra = numpy.zeros((number_of_pixels,32),dtype = 'float')
        for i in range(0, number_of_pixels):
            x, y = numpy.random.randint(data.shape[4]),numpy.random.randint(data.shape[3])
            spectra[i,:] = data[:,:,:,y,x]

        avg_spectrum = numpy.average(spectra,axis =0)
        # spectrum shape: (32,)
        return avg_spectrum



    def generate_1_pixel_spectrum_rand(self, data):
        spectrum = numpy.zeros((32,),dtype = 'float')
        x, y = numpy.random.randint(data.shape[4]),numpy.random.randint(data.shape[3])
        spectrum[:]= data[:,:,:,y,x]
        # spectrum shape: (32,)
        return spectrum



    ########################################################################
    def Gaussian_weight(self, mean_rgb):
        # w shape: (32,)
        w = numpy.exp((-numpy.power(numpy.linalg.norm(self.zeiss - mean_rgb, axis = 0), 2))/(2*self.sigma**2)).astype('float')
        w = w/numpy.sum(w)
        return w


    def Gaussian_kernel(self, spectrum):
        # spectrum shape: (32,)
        # zeiss shape: (1,32)
        # weight shape: (32,)
        # original indistinguishable checkerboard rgb
        # mean_r = 650
        # mean_g = 520
        # mean_b = 470
        # all transmitted peak rgb
        # mean_r = 650
        # mean_g = 510
        # mean_b = 440
        # testing here #########
        # mean_r = 630
        # mean_g = 545
        # mean_b = 470
     
        R_weight = self.Gaussian_weight(self.mean_r)
        G_weight = self.Gaussian_weight(self.mean_g)
        B_weight = self.Gaussian_weight(self.mean_b)

        R = numpy.sum(R_weight * spectrum)
        G = numpy.sum(G_weight * spectrum)
        B = numpy.sum(B_weight * spectrum)
        rgb_max = numpy.maximum(numpy.maximum(R,G),B)

        R = R/rgb_max * self.gamma
        G = G/rgb_max * self.gamma
        B = B/rgb_max * self.gamma

        return numpy.asarray([R,G,B])



    def Gaussian_kernel_data(self, G_hyper):
        # zeiss shape: (1,32)
        # weight shape: (32,)
        # dim_str = 'TZCYX'
        R_weight = self.Gaussian_weight(self.mean_r)
        G_weight = self.Gaussian_weight(self.mean_g)
        B_weight = self.Gaussian_weight(self.mean_b)

        G_rgb = numpy.zeros((G_hyper.shape[:2]+(3,)+G_hyper.shape[-2:]), dtype ='float')
        G_rgb[:,:,0,:,:] = numpy.tensordot(G_hyper, R_weight, axes = (2,0))
        G_rgb[:,:,1,:,:] = numpy.tensordot(G_hyper, G_weight, axes = (2,0))
        G_rgb[:,:,2,:,:] = numpy.tensordot(G_hyper, B_weight, axes = (2,0))

        rgb_max = numpy.amax(G_rgb, axis=2)

        print("check here: ")
        print(G_rgb.shape)
        print(rgb_max.shape)

        G_rgb = numpy.divide(G_rgb, rgb_max) * self.gamma
        newG_rgb = numpy.transpose(G_rgb, (0,1,3,4,2))

        return newG_rgb



    ####################################################################
    #!/usr/bin/env python
    # vim:set ft=python fileencoding=utf-8 sr et ts=4 sw=4 : See help 'modeline'

    '''
        == A few notes about color ==

        Color   Wavelength(nm) Frequency(THz)
        Red     620-750        484-400
        Orange  590-620        508-484
        Yellow  570-590        526-508
        Green   495-570        606-526
        Blue    450-495        668-606
        Violet  380-450        789-668

        f is frequency (cycles per second)
        l (lambda) is wavelength (meters per cycle)
        e is energy (Joules)
        h (Plank's constant) = 6.6260695729 x 10^-34 Joule*seconds
                             = 6.6260695729 x 10^-34 m^2*kg/seconds
        c = 299792458 meters per second
        f = c/l
        l = c/f
        e = h*f
        e = c*h/l

        List of peak frequency responses for each type of 
        photoreceptor cell in the human eye:
            S cone: 437 nm
            M cone: 533 nm
            L cone: 564 nm
            rod:    550 nm in bright daylight, 498 nm when dark adapted. 
                    Rods adapt to low light conditions by becoming more sensitive.
                    Peak frequency response shifts to 498 nm.

    '''


    def wavelength_to_rgb(self, spectrum):

        '''This converts a given wavelength of light to an 
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).

        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        '''


        #### Find maxima wavelength from a 32-ch spectrum#######
        ind = numpy.argmax(spectrum)
        wavelength = self.zeiss[0,ind]

        ##########################################
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** self.gamma
            G = 0.0
            B = (1.0 * attenuation) ** self.gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** self.gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** self.gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** self.gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** self.gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** self.gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        # R *= self.gamma
        # G *= self.gamma
        # B *= self.gamma
        return numpy.asarray([R,G,B])



    def wavelength_to_rgb_data(self,  G_hyper):

        vector = self.zeiss[0]

        ind = numpy.argmax(G_hyper, axis=2) #(1, 1, 196, 365)
        print('ind shape: '+str(ind.shape))
        tmp = numpy.zeros((ind.shape + (vector.shape[0],)),dtype=vector.dtype)+vector #(1, 1, 32, 196, 365)
        # x = tmp[ind]

        x = vector[ind]
        print(x.shape)

        G_rgb = numpy.zeros((G_hyper.shape[:2] +G_hyper.shape[-2:] + (3,)), dtype='float')
        mask = numpy.logical_and(x>=380, x<=440)
        attenuation = 0.3 + 0.7 * (x[mask] - 380) / (440 - 380)
        G_rgb[mask, 0] = numpy.multiply((-(x[mask] - 440) / (440 - 380)), attenuation) ** self.gamma
        G_rgb[mask, 1] = 0
        G_rgb[mask, 2] = (1.0 * attenuation) ** self.gamma

        mask = numpy.logical_and(x>440, x<=490)
        G_rgb[mask, 0] = 0
        G_rgb[mask, 1] = ((x[mask] - 440) / (440 - 380)) ** self.gamma
        G_rgb[mask, 2] = 1

        mask = numpy.logical_and(x>490, x<=510)
        G_rgb[mask, 0] = 0
        G_rgb[mask, 1] = 1
        G_rgb[mask, 2] = (-(x[mask] - 510) / (510 - 490)) ** self.gamma

        mask = numpy.logical_and(x>510, x<=580)
        G_rgb[mask, 0] = ((x[mask] - 510) / (580 - 510)) ** self.gamma
        G_rgb[mask, 1] = 1
        G_rgb[mask, 2] = 0

        mask = numpy.logical_and(x>580, x<=645)
        G_rgb[mask, 0] = 1
        G_rgb[mask, 1] = (-(x[mask] - 645) / (645 - 580)) ** self.gamma
        G_rgb[mask, 2] = 0

        mask = numpy.logical_and(x>645, x<=750)
        attenuation = 0.3 + 0.7 * (750 - x[mask]) / (750 - 645)
        G_rgb[mask, 0] = (1.0 * attenuation) ** self.gamma
        G_rgb[mask, 1] = 0
        G_rgb[mask, 2] = 0

        mask = numpy.logical_or(x < 380, x > 750)
        G_rgb[mask, 0] = 0
        G_rgb[mask, 1] = 0
        G_rgb[mask, 2] = 0
        # G_rgb = numpy.transpose(G_rgb, (0, 1, 4, 2, 3))

        print("test")


        # ##########################################
        # if wavelength >= 380 and wavelength <= 440:
        #     attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        #     R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** self.gamma
        #     G = 0.0
        #     B = (1.0 * attenuation) ** self.gamma
        # elif wavelength >= 440 and wavelength <= 490:
        #     R = 0.0
        #     G = ((wavelength - 440) / (490 - 440)) ** self.gamma
        #     B = 1.0
        # elif wavelength >= 490 and wavelength <= 510:
        #     R = 0.0
        #     G = 1.0
        #     B = (-(wavelength - 510) / (510 - 490)) ** self.gamma
        # elif wavelength >= 510 and wavelength <= 580:
        #     R = ((wavelength - 510) / (580 - 510)) ** self.gamma
        #     G = 1.0
        #     B = 0.0
        # elif wavelength >= 580 and wavelength <= 645:
        #     R = 1.0
        #     G = (-(wavelength - 645) / (645 - 580)) ** self.gamma
        #     B = 0.0
        # elif wavelength >= 645 and wavelength <= 750:
        #     attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        #     R = (1.0 * attenuation) ** self.gamma
        #     G = 0.0
        #     B = 0.0
        # else:
        #     R = 0.0
        #     G = 0.0
        #     B = 0.0
        # # R *= self.gamma
        # # G *= self.gamma
        # # B *= self.gamma
        # return numpy.asarray([R,G,B])
        return G_rgb



    # here spectrum is average
    def generate_1_cell_on_checkerboard_avg(self, CFP_spectrum, YFP_spectrum,RFP_spectrum,ii,jj,CFP_rate, YFP_rate, RFP_rate):

        di,dj = ii*self.deltaD, jj*self.deltaD
        print('di: '+ str(di)+' dj: ' + str(dj))
        
        CFP_spectrum = numpy.roll(CFP_spectrum, -di)
        CFP_spectrum = CFP_spectrum * CFP_rate 
        
        YFP_spectrum = YFP_spectrum * YFP_rate 

        RFP_spectrum = numpy.roll(RFP_spectrum, dj)
        RFP_spectrum = RFP_spectrum * RFP_rate 

        G_rgb = numpy.zeros((3, self.cell_size,self.cell_size),dtype='float')
        G_hyper = numpy.zeros((32, self.cell_size, self.cell_size), dtype='float')

        step = int(self.cell_size/5)
        G_rgb[:, :, :] = self.Gaussian_kernel(CFP_spectrum)
        G_rgb[:, step:4*step, step:4*step] = self.Gaussian_kernel(YFP_spectrum)
        G_rgb[:, 2*step:3*step, 2*step:3*step] = self.Gaussian_kernel(RFP_spectrum)

        G_hyper[:, :, :] = CFP_spectrum
        G_hyper[:, step:4 * step, step:4 * step] = YFP_spectrum
        G_hyper[:, 2 * step:3 * step, 2 * step:3 * step] = RFP_spectrum

        fig = plt.figure()
        spec_axes = fig.add_subplot(111)
        spec_axes.plot(self.zeiss[0,:],CFP_spectrum,'c')
        spec_axes.plot(self.zeiss[0,:],YFP_spectrum,'y')
        spec_axes.plot(self.zeiss[0,:],RFP_spectrum,'r')

        return G_rgb, G_hyper,fig



    # here spectrum is random
    def generate_1_cell_on_checkerboard_rand(self,ii,jj,CFP_rate, YFP_rate, RFP_rate):

        di,dj = ii*self.deltaD, jj*self.deltaD
        print('di: '+ str(di)+' dj: ' + str(dj))

        G_rgb = numpy.zeros((3, self.cell_size, self.cell_size),dtype = 'float')
        G_hyper = numpy.zeros((32, self.cell_size, self.cell_size), dtype='float')

        step = int(self.cell_size / 5)
        for i in range(0,self.cell_size):
            for j in range(0,self.cell_size):
                if i in range(2*step, 3* step) and j in range(2*step, 3 * step):

                    RFP_spectrum = self.generate_1_pixel_spectrum_rand(self.data_RFP)
                    RFP_spectrum = numpy.roll(RFP_spectrum, dj)
                    RFP_spectrum = RFP_spectrum * RFP_rate 
                    # G_rgb[:, j, i] = self.Gaussian_kernel(RFP_spectrum)
                    G_rgb[:,j,i] = self.wavelength_to_rgb(RFP_spectrum)
                    G_hyper[:, j, i] = RFP_spectrum

                elif i in range(step, 4* step) and j in range(step, 4 * step):
                    YFP_spectrum = self.generate_1_pixel_spectrum_rand(self.data_YFP)
                    YFP_spectrum = YFP_spectrum * YFP_rate 
                    # G_rgb[:, j, i] = self.Gaussian_kernel(YFP_spectrum)
                    G_rgb[:,j,i] = self.wavelength_to_rgb(YFP_spectrum)
                    G_hyper[:, j, i] = YFP_spectrum

                else:
                    CFP_spectrum = self.generate_1_pixel_spectrum_rand(self.data_CFP)
                    CFP_spectrum = numpy.roll(CFP_spectrum, -di)
                    CFP_spectrum = CFP_spectrum * CFP_rate 
                    # G_rgb[:,j,i] = self.Gaussian_kernel(CFP_spectrum)
                    G_rgb[:, j,i] = self.wavelength_to_rgb(CFP_spectrum)
                    G_hyper[:, j,i] = CFP_spectrum

        return G_rgb,G_hyper



    def generate_checkerboard_new(self, board_size):
        ######
        CFP_spectrum = self.generate_1_pixel_spectrum_avg(1000, self.data_CFP)
        YFP_spectrum = self.generate_1_pixel_spectrum_avg(1000, self.data_YFP)
        RFP_spectrum = self.generate_1_pixel_spectrum_avg(1000, self.data_RFP)

        ########## generate rates for normalization
        CFP_max = numpy.max(CFP_spectrum)
        YFP_max = numpy.max(YFP_spectrum)
        RFP_max = numpy.max(RFP_spectrum)

        spectrum_max = numpy.maximum(numpy.maximum(CFP_max, YFP_max), RFP_max)
        CFP_rate = spectrum_max / CFP_max
        YFP_rate = spectrum_max / YFP_max
        RFP_rate = spectrum_max / RFP_max
        
        # CFP_int = numpy.trapz(CFP_spectrum)
        # YFP_int = numpy.trapz(YFP_spectrum)
        # RFP_int = numpy.trapz(RFP_spectrum)
        # spectrum_max = numpy.maximum(numpy.maximum(CFP_int, YFP_int), RFP_int)
        # CFP_rate = spectrum_max / CFP_int
        # YFP_rate = spectrum_max / YFP_int
        # RFP_rate = spectrum_max / RFP_int

        ########## generate step distance for each shift
        CFP_indmax = numpy.argmax(CFP_spectrum)
        YFP_indmax = numpy.argmax(YFP_spectrum)
        RFP_indmax = numpy.argmax(RFP_spectrum)

        d1 = YFP_indmax - CFP_indmax
        d2 = RFP_indmax - YFP_indmax

        RFP_spectrum = numpy.roll(RFP_spectrum, -d2)
        CFP_spectrum = numpy.roll(CFP_spectrum, d1)
        
        self.data_RFP = numpy.roll(self.data_RFP,-d2,axis=2) 
        self.data_CFP = numpy.roll(self.data_CFP,d1,axis=2)
        print("generate_checkerboard new:")
        for i in range(0, 3):
            for j in range(0, 3):

                # G_cell_rgb, G_cell_hyper, fig = self.generate_1_cell_on_checkerboard_avg(CFP_spectrum, YFP_spectrum, RFP_spectrum, i,j,CFP_rate,YFP_rate,RFP_rate)
                G_cell_rgb, G_cell_hyper = self.generate_1_cell_on_checkerboard_rand(i,j,CFP_rate,YFP_rate,RFP_rate)

                if j == 0:
                    G_r_rgb = G_cell_rgb
                    G_r_hyper = G_cell_hyper
                else:
                    G_r_rgb = numpy.concatenate((G_r_rgb, G_cell_rgb),axis = -2)
                    G_r_hyper = numpy.concatenate((G_r_hyper, G_cell_hyper), axis=-2)
            if i == 0:
                G_rgb = G_r_rgb
                G_hyper = G_r_hyper
            else:
                G_rgb = numpy.concatenate((G_rgb, G_r_rgb), axis=-1)
                G_hyper = numpy.concatenate((G_hyper, G_r_hyper), axis=-1)
        # plt.show()
        G_hyper = G_hyper[numpy.newaxis, numpy.newaxis, :, :, :]

        return G_rgb,G_hyper



    ###################################################
    def __init__(self,  mean_r=630, mean_g=545, mean_b=470, sigma=20.0, gamma=0.8, deltaD=2, cell_size=100):
        self.data_CFP = []
        self.data_YFP = []
        self.data_RFP = []
        self.zeiss = []

        self.mean_r = mean_r
        self.mean_g = mean_g
        self.mean_b = mean_b

        self.sigma = sigma
        self.gamma = gamma
        self.deltaD = deltaD
        self.cell_size = cell_size

        self.G_hyper = []
        self.G_rgb = []


    def checkerboard_generator(self, dir, mode_num=0):
        self.read_files_checkerboard(dir)
        # zeiss shape: (1,32)
        _, self.G_hyper = self.generate_checkerboard_new(board_size=3)
        # self.G_rgb = numpy.transpose(G_rgb[numpy.newaxis, numpy.newaxis, :, :, :], (0,1,3,4,2))
        self.G_rgb= self.Gaussian_kernel_data(self.G_hyper)
        self.G_rgb2 =  self.wavelength_to_rgb_data(self.G_hyper)

    def realdata_rgb_generator(self,dir, filename):
        self.read_files_lsm(dir,filename)
        self.G_rgb = self.Gaussian_kernel_data(self.G_hyper)
        self.G_rgb2 = self.wavelength_to_rgb_data(self.G_hyper)


    def plot_RGB_data(self):
        fig = plt.figure()
        G_RGB_axes = fig.add_subplot(211)
        G_RGB_axes.imshow(self.G_rgb[0,0])
        G_RGB_axes2 = fig.add_subplot(212)
        G_RGB_axes2.imshow(self.G_rgb2[0,0])
        #
        # fig2 = plt.figure()
        # G_gray_axes = fig2.add_subplot(111)
        # G_gray_axes.imshow(numpy.mean(self.G_hyper[0,0], axis=0), cmap='gray')

        plt.show()
        plt.ion()

    def plot_GaussianKerenl(self):
        CFP_spectrum = self.generate_1_pixel_spectrum_avg(1000, self.G_hyper)

        R_weight = self.Gaussian_weight(self.mean_r)
        G_weight = self.Gaussian_weight(self.mean_g)
        B_weight = self.Gaussian_weight(self.mean_b)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.zeiss[0,:],R_weight/numpy.max(R_weight),'r')
        ax.plot(self.zeiss[0,:],G_weight/numpy.max(G_weight),'g')
        ax.plot(self.zeiss[0,:],B_weight/numpy.max(B_weight),'b')

        ax.plot(self.zeiss[0,:], CFP_spectrum/numpy.max(CFP_spectrum),'y')
        plt.show()

    def plot_avg_data_sampled(self):
        spectrum = self.generate_1_pixel_spectrum_avg(1000, self.G_hyper)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.zeiss[0,:], spectrum)
        plt.show()


#mean_r=630, mean_g=545, mean_b=470, sigma=20.0, gamma=0.8, deltaD=2, cell_size=100
x = Wen_checkerboard(630,545,470,10)
dir = x.directory_check(True)

###########  checkerboard  ###############################
# x.checkerboard_generator(dir)

###########  real data  ###############################


# filename = '/fig4_HySP_1frameHD_740nm_pos2.lsm'
# filename = '/fig5_02-mko2-wholefish-FL_Subset.lsm'
# filename = '/fig5_02-mko2-wholefish-FL.lsm'
# filename = '/fig6_27-fish3-ct122a-kdrl-mcherry-cerulean-458nm-tile-zoom-forcomparison.lsm'
filename = '/fig7_fish1_tileHD_Brain_zoomout2x.lsm'
x.realdata_rgb_generator(dir, filename)
x.plot_RGB_data()

x.plot_GaussianKerenl()
x.plot_avg_data_sampled()
# output_filename = 'E:/ImageProcessing/Projects/HySP_NLM/Checkerboard/Wen_checkerboard_20180924'
# output_filename = '/Users/Wen/PycharmProjects/Wen_HySP_test/Checkerboard/Checkerboard_tif_normalized/Wen_checkerboard_20180428'
# x.save_2_tiff(self.G_hyper, output_filename)



