#!/usr/bin/env python
# Dillon Brout 3/10/2015
# dbrout@physics.upenn.edu

"""
Usage:
import mcmc
a = mcmc.metropolis_hastings( model, data, psfs, weights, substamp, , Nimage )
a.run_d_mc()

1D arrays (all of same size)
model                 : contains all model parameters

2D Stamps (all of same size) 
data                  : data stamps (1 for each epoch)
psfs                  : psf stamps (1 for each epoch)
weights               : uncertainty stamps (1 for each epoch)

Integers
substamp              : size of one edge of a stamp
Nimage                : Number of epochs


To do list: 
only test for convergence with supernova/star parameters
check autocorr with galsim_iter.py and figure out when to stop mcmc
figure out how to calculate mean and uncertainty properly
calculate covariance

Geweke is slow for lots of iter

GEWEKE COVARIANCE

"""






import numpy as np
import scipy.ndimage
#from . import Matplot, flib
#from .utils import autocorr, autocov
from copy import copy
import pdb
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
#from pylab import pcolor, show, colorbar, xticks, yticks
#import pylab as plt
import time
import pyfits as pf
import os
import math
import pyfftw


class metropolis_hastings():

    def __init__(self
                , model = None
                , stdev = None
                , data = None
                , psfs = None
                , weights = None
                , substamp = 0
                , Nimage = 1
                , maxiter = 100000
                , gain = 1.0
                , model_errors = False
                , readnoise = 5.
                , analytical = 'No'
                , mask = None
                , fix = None
                , sky=None
                ):

        if model is None:
            raise AttributeError('Must provide model array!')
        if stdev is None:
            self.stdev = np.sqrt(model)
        else:
            self.stdev = stdev
        if data is None:
            raise AttributeError('Must provide real data for comparison!')
        if psfs is None:
            raise AttributeError('Must provide psfs for each epoch!')
        if weights is None:
            raise AttributeError('Must provide weights for each epoch!')
        if substamp == 0:
            if len(model) > 1:
                raise AttributeError('Must provide substamp size!')
            else:
                if len(model) == 1:
                    print 'Warning : Substamp size is zero, assuming calibration star.' 
                else:
                    raise AttributeError('Model length is zero')
        #else:
        #    if Nimage == 1:
        #        raise AttributeError('Must provide Nimage (number of epochs)!')      

        #oktogo = False
        #if len( model[ substamp**2+1: ] ) == len( data ):
        #    if len( stdev[ substamp**2+1: ] ) == len( data ):
        #        if len( data ) == Nimage:
        #            if len( data ) == len( psfs ):
        #                if len( data ) == len( weights ):
        #                    oktogo = True
        #if not oktogo:
        #    raise AttributeError('Require that the dimensions of the following are all '+str(Nimage)+': \
        #    \n\tmodel\n\tdata\n\tpsfs')

        self.model = model
        self.deltas = copy(self.stdev) #this vec will change for each iter
        self.substamp = substamp
        self.Nimage = Nimage
        self.maxiter = maxiter
        self.gain = gain
        self.model_errors = model_errors
        self.readnoise = readnoise
        self.sky = sky

        self.didtimeout = False
        if Nimage == 1:
            self.psfs = np.zeros((1,substamp,substamp))
            self.psfs[0,:,:] = psfs
            self.weights = np.zeros((1,substamp,substamp))
            self.weights[0,:,:] = weights
            self.data = np.zeros((1,substamp,substamp))
            self.data[0,:,:] = data
        else:
            self.data = data
            self.psfs = psfs
            self.weights = weights

        if mask == None:
            self.mask = np.zeros(self.data.shape)+1.
        else:
            self.mask = mask

        if fix == None:
            self.fix = (np.zeros(len(self.model)+1)+1.)
        else:
            self.fix = fix

        #plt.imshow(self.data[0,:,:])
        #plt.show()
        self.galaxy_model = copy(self.model[ 0 : self.substamp**2.]).reshape(self.substamp,self.substamp)


        self.z_scores_say_keep_going = True

        self.sims = np.zeros([Nimage,substamp,substamp])
        self.run_d_mc()


    def run_d_mc( self ):
        self.lastchisq = 9999999999.9
        self.chisq = []
        self.chisq.append(self.lastchisq)
        self.history = []
        self.accepted_history = 0
        self.accepted_int = 0
        self.t1 = time.time()
        self.counter = 0
        #plt.imshow(self.data)
        #plt.show()
        #self.t2 = time.time()

        while self.z_scores_say_keep_going:
            #self.t2 = time.time()
            self.counter += 1
            self.accepted_int += 1
            self.mcmc_func()
            
            #Check Geweke Convergence Diagnostic every 5000 iterations
            if (self.counter % 50) == 49: 
                self.check_geweke()
                self.last_geweke = self.counter
            if self.counter > self.maxiter:
                self.z_scores_say_keep_going = False#GETOUT
                self.didtimeout = True
            #plt.imshow(self.data[20,self.substamp/2.-14.:self.substamp/2.+14.,self.substamp/2.-14.:self.substamp/2.+14.])
            #plt.show()

        self.summarize_run()
        self.model_params()

        self.t2 = time.time()

    def summarize_run( self ):
        self.t2 = time.time()
        print 'Total Time: ' + str( self.t2 - self.t1 )
        print 'Num Iterations: ' + str( self.counter )
        print 'Accepted Percentage: ' + str( self.accepted_history )
        print 'Seconds per iteration: '+str(float(( self.t2 - self.t1 )/self.counter))
        #np.savez(self.results_npz, pixel_history = self.pixel_history
        #                        , simulated_stamps = self.simulated_images
        #                        , data_stamps = self.real_data_stamps_trimmed
        #                        , sn_flux_history  = self.sn_flux_history
        #                        )


    def mcmc_func( self ):

        #t1 = time.time()
        self.adjust_model()
        #t2 = time.time()

        # Contains the convolution
        self.kernel()
        #t3 = time.time()

        #Calculate Chisq over all epochs
        self.thischisq = self.chisq_sim_and_real()
        #t4 = time.time()

        #decide whether to accept new values
        accept_bool = self.accept(self.lastchisq,self.thischisq)
        #t5 = time.time()


        if accept_bool:
            print 'accepted'
            self.lastchisq = self.thischisq
            self.accepted_history = ( self.accepted_history * self.accepted_int + 1.0 ) / ( self.accepted_int + 1 )
            self.copy_adjusted_image_to_model()
            self.update_history()
            self.chisq.append( self.thischisq )
        else:
            self.accepted_history = ( self.accepted_history * self.accepted_int ) / ( self.accepted_int + 1 )
            self.update_unaccepted_history()

        #t6 = time.time()
        #print 'adjust model '+str(t2-t1)
        #print 'kernel '+str(t3-t2)
        #print 'chisq '+str(t4-t3)
        #print 'accept bool '+str(t5-t4)
        #print 'history update '+str(t6-t5)
        #raw_input()

    def adjust_model( self ):
        for i in np.arange( len( self.deltas ) ):
            if self.stdev[ i ] > 0:
                self.deltas[ i ] = np.random.normal( scale= self.stdev[ i ] )
            else:
                self.deltas[ i ] = 0.0

        self.kicked_model = self.model + self.deltas
        self.kicked_galaxy_model = self.kicked_model[ 0 : self.substamp**2. ].reshape( self.substamp, self.substamp )

        return

    def kernel( self ):

        if self.Nimage == 1:
                self.sims[ 0, : , : ] = (self.galaxy_model + self.sky + self.psfs[ 0, : , : ]*self.kicked_model[self.substamp**2.])*self.mask
        else:
            for epoch in np.arange( self.Nimage ):
                t1 = time.time()
                custom_fft_conv = CustomFFTConvolution(self.kicked_galaxy_model + self.sky[epoch], self.psfs[ epoch,:,:])
                galaxy_conv = custom_fft_conv(self.kicked_galaxy_model + self.sky[epoch], self.psfs[ epoch,:,:])
                #t2 = time.time()
                #galaxy_conv = scipy.ndimage.convolve( self.kicked_galaxy_model + self.sky[epoch], self.psfs[ epoch,:,:] )
                #t3 = time.time()
                #print 'FFTW: '+str(t2-t1)
                #print 'scipy: '+str(t3-t2)
                #raw_input()
                star_conv = self.kicked_model[self.substamp**2. + epoch ] * self.psfs[ epoch,:,:]
                self.sims[ epoch,:,:] =  (star_conv + galaxy_conv)*self.mask

    def get_final_sim( self ):

        if self.Nimage == 1:
                self.sims[ 0, : , : ] = (self.model_params[:self.substamp**2] + self.sky + self.psfs[ 0, : , : ]*self.kicked_model[self.substamp**2.])*self.mask
        else:
            for epoch in np.arange( self.Nimage ):
                galaxy_conv = scipy.ndimage.convolve( self.kicked_galaxy_model + self.sky[epoch], self.psfs[ epoch,:,:] )
                star_conv = self.kicked_model[self.substamp**2. + epoch ] * self.psfs[ epoch,:,:]
                self.sims[ epoch,:,:] =  (star_conv + galaxy_conv)*self.mask

    def chisq_sim_and_real( self, model_errors = False ):
        chisq = 0.0
        if self.Nimage == 1:
            if model_errors:
                chisq += np.sum( ( (self.sims[ 0, :,:] - self.data[ 0, :,:])**2 / (self.sims[ 0,:,:]/self.gain + self.readnoise/self.gain**2) ).ravel() )
            else:
                chisq += np.sum( ( (self.sims[ 0, :,:] - self.data[ 0, :,:])**2 * (self.weights[ 0,:,:])).ravel() )
        else:
            for epoch in np.arange( self.Nimage ):
                if model_errors:
                    chisq += np.sum( ( (self.sims[ epoch, :,:] - self.data[ epoch, :,:])**2 / (self.sims[ epoch,:,:]/self.gain + self.readnoise/self.gain**2) ).ravel() )
                else:
                    chisq += np.sum( ( (self.sims[ epoch, :,:] - self.data[ epoch, :,:])**2 / (self.weights[ epoch,:,:] )**2).ravel() )
            

        #save_fits_image(( (self.sims[ 0, :,:] - self.data[ 0, :,:])**2 * (self.weights[ 0,:,:])),'./chisq.fits')
        #save_fits_image(self.sims[ 0, :,:],'./sim.fits')
        #save_fits_image(self.data[ 0, :,:],'./data.fits')
        #save_fits_image(self.weights[ 0,:,:],'./weights.fits')
        #save_fits_image((self.sims[ 0,:,:]/self.gain + self.readnoise/self.gain**2),'./modelerrors.fits')

        print 'Chisquare: '+str(chisq)
        #raw_input()

        return chisq

    def accept( self, last_chisq, this_chisq ):
        alpha = np.exp( last_chisq - this_chisq ) / 2.0
        return_bool = False
        if alpha >= 1:
            return_bool = True
        else:
            if np.random.rand() < alpha:
                return_bool = True
        return return_bool

    def copy_adjusted_image_to_model( self ):
        self.model = copy( self.kicked_model )
        return

    def update_history( self ):
        self.history.append( self.kicked_model )
        return

    def update_unaccepted_history( self ):
        self.history.append( self.model )
        return

    def model_params( self ):
        self.make_history()
        burn_in = int(self.nphistory.shape[0]*.5)
        self.model_params = copy( self.model )
        self.model_uncertainty = copy( self.model )
        for i in np.arange( len( self.model ) ):
            self.model_params[ i ] = np.mean( self.nphistory[ burn_in : , i ] )
            self.model_uncertainty[ i ] = np.std( self.nphistory[ burn_in : , i ] )

    def autocorr( self, x ):
        result = np.correlate( x, x, mode='full' )
        return result[ result.size / 2 : ]

    def get_params( self ):
        if self.didtimeout:
            filelist = [ f for f in os.listdir("./out")]
            for f in filelist:
                os.remove('./out/'+f)
            for i in np.arange(self.Nimage):            
                save_fits_image(self.sims[i,:,:],'./out/'+str(self.nphistory[-1,self.substamp**2+i])+'_flux.fits')
                save_fits_image(self.data[i,:,:],'./out/'+str(self.nphistory[-1,self.substamp**2+i])+'_fluxdata.fits')
                save_fits_image(self.weights[i,:,:],'./out/'+str(self.nphistory[-1,self.substamp**2+i])+'_fluxnoise.fits')

            return np.zeros(len(self.model_params))+1e8,np.zeros(len(self.model_params))+1e9,self.nphistory
        return self.model_params, self.model_uncertainty, self.nphistory # size: self.history[num_iter,len(self.model_params)]

    def get_params_analytical_weighted( self ):
        burn_in = int(self.nphistory.shape[0]*.5)
        model_params = copy( self.model )
        model_uncertainty = copy( self.model )
        
        for i in np.arange( len( self.model ) ):
            model_params[ i ] = np.mean( self.nphistory[ burn_in : , i ] )
            model_uncertainty[ i ] = np.std( self.nphistory[ burn_in : , i ] )

        sim = self.model_params[:self.substamp**2] + self.psfs[0,:,:].ravel()*model_params[self.substamp**2]

        sum_numer = np.sum(sim.ravel()*self.psfs[0,:,:].ravel()*self.weights[0,:,:].ravel())
        sum_denom = np.sum(self.psfs[0,:,:].ravel()*self.psfs[0,:,:].ravel()*self.weights[0,:,:].ravel())

        scale = sum_numer/sum_denom

        #compute an image of modle params and then compute sum.
        
        return scale
        
    def get_params_analytical_simple( self ):
        burn_in = int(self.nphistory.shape[0]*.5)
        model_params = copy( self.model )
        model_uncertainty = copy( self.model )
        
        for i in np.arange( len( self.model ) ):
            model_params[ i ] = np.mean( self.nphistory[ burn_in : , i ] )
            model_uncertainty[ i ] = np.std( self.nphistory[ burn_in : , i ] )

        sim = self.model_params[:self.substamp**2] + self.psfs[0,:,:].ravel()*model_params[self.substamp**2]

        sum_numer = np.sum(sim.ravel())
        sum_denom = np.sum(self.psfs[0,:,:].ravel())

        scale = sum_numer/sum_denom

        return scale

    def make_history( self ):
        num_iter = len( self.history )
        self.nphistory = np.zeros( (num_iter , len( self.model )) ) 
        for i in np.arange( num_iter ):
            self.nphistory[ i , : ] = self.history[ i ]


    #DIAGNOSTICS
    def check_geweke( self, zscore_mean_crit=1, zscore_std_crit=1.0 ):
        #print 'making history'
        self.make_history()
        #print 'geweke'
        zscores = self.geweke( self.nphistory[:, self.substamp**2 : ] )
        #print 'done'
        #If abs(mean) of zscores is less than .5 and if stdev lt 1.0 then stop and calculate values and cov
        means = np.mean(zscores[1,:,:], axis=0)
        print means
        stdevs = np.std(zscores[1,:,:], axis=0)
        print stdevs
        alltrue = True
        for mean in means:
            if alltrue:
                if (abs(mean) > zscore_mean_crit) or (math.isnan(mean)):
                    alltrue = False
        if alltrue:
            for std in stdevs:
                if alltrue:
                    if (std > zscore_std_crit) or (math.isnan(std)):
                        alltrue = False
        if alltrue:
            self.z_scores_say_keep_going = False
            print 'Zscores computed and convergence criteria has been met'
        else:
            print 'Zscores computed and convergence criteria have not been met, mcmc will continue...'

        return

    def geweke( self, x_in, first = .1, last = .5, intervals = 20, maxlag = 20):
        """Return z-scores for convergence diagnostics.
        Compare the mean of the first percent of series with the mean of the last percent of
        series. x is divided into a number of segments for which this difference is
        computed. If the series is converged, this score should oscillate between
        -1 and 1.
        Parameters
        ----------
        x : array-like, size x[num_params,num_iter]
          The trace of some stochastic parameter.
        first : float
          The fraction of series at the beginning of the trace.
        last : float
          The fraction of series at the end to be compared with the section
          at the beginning.
        intervals : int
          The number of segments.
        maxlag : int
          Maximum autocorrelation lag for estimation of spectral variance
        Returns
        -------

        """
    
        # Filter out invalid intervals
        if first + last >= 1:
            raise ValueError(
                "Invalid intervals for Geweke convergence analysis",
                (first, last))

        #if its 1d make it 2d so all code can be the same
        ndim = np.ndim(x_in)
        if ndim == 1:
            x = np.array(x_in.shape[0],1)
            x[:,0] = x_in
        else:
            x = x_in
        starts = np.linspace(0, int(x[:,0].shape[0]*(1.-last)), intervals).astype(int)


        # Initialize list of z-scores
        zscores = [None] * intervals 
        zscores = np.zeros((2,len(starts),x.shape[1]))


        # Loop over start indices
        #print len(starts)
        for i,s in enumerate(starts):

            # Size of remaining array
            x_trunc = x[s:,:]
            #print x_trunc.shape
            n = x_trunc.shape[0]

            # Calculate slices
            first_slice = x_trunc[ :int(first * n),:]
            last_slice = x_trunc[ int(last * n):,:]

            z = (first_slice.mean(axis=0) - last_slice.mean(axis=0))
            
            #spectral density
            z /= np.sqrt(np.fft.rfft(first_slice,axis=0)[0]/first_slice.shape[0] +
                     np.fft.rfft(last_slice,axis=0)[0]/last_slice.shape[0])
            
            #print zscores.shape
            #print x.shape[0]
            zscores[0,i,:] = np.ones(x.shape[1])*x.shape[0] - n
            #print z.shape
            zscores[1,i,:] = z
            #print zscores[1,:,:]

        #print zscores[1,:,:]
        #raw_input()
        return zscores



    def plot_covar( self, data ):


        # generating some uncorrelated data
        data = rand(10,100) # each row of represents a variable

        # creating correlation between the variables
        # variable 2 is correlated with all the other variables
        data[2,:] = sum(data,0)
        # variable 4 is correlated with variable 8
        data[4,:] = log(data[8,:])*0.5

        # plotting the correlation matrix
        R = corrcoef(data)
        pcolor(R)
        colorbar()
        yticks(arange(0.5,10.5),range(0,10))
        xticks(arange(0.5,10.5),range(0,10))
        show()


class CustomFFTConvolution(object):

    def __init__(self, A, B, threads=1):

        #shape = (np.array(A.shape) + np.array(B.shape))-1
        shape = np.array(A.shape)
        if np.iscomplexobj(A) and np.iscomplexobj(B):
            self.fft_A_obj = pyfftw.builders.fftn(
                    A, s=shape, threads=threads)
            self.fft_B_obj = pyfftw.builders.fftn(
                    B, s=shape, threads=threads)
            self.ifft_obj = pyfftw.builders.ifftn(
                    self.fft_A_obj.get_output_array(), s=shape,
                    threads=threads)

        else:
            self.fft_A_obj = pyfftw.builders.rfftn(
                    A, s=shape, threads=threads)
            self.fft_B_obj = pyfftw.builders.rfftn(
                    B, s=shape, threads=threads)
            self.ifft_obj = pyfftw.builders.irfftn(
                    self.fft_A_obj.get_output_array(), s=shape,
                    threads=threads)

    def __call__(self, A, B):

        fft_padded_A = self.fft_A_obj(A)
        fft_padded_B = self.fft_B_obj(B)

        return self.ifft_obj(fft_padded_A * fft_padded_B)


def save_fits_image(image,filename):
    hdu = pf.PrimaryHDU(image)
    if os.path.exists(filename):
        os.remove(filename)
    hdu.writeto(filename)
    return

if __name__ == "__main__":

    #TEST DATA
    # 4 by for image with 4 supernova epochs initalized to 1
    Nepochs = 4
    substamp = 5
    model = np.array([250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,0,119900,160000,200000])
    stdev = np.array([20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,20.,0.,25.,25.,25.])
    

    data = np.zeros((4,5,5))
    a = [250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250]
    ina = np.asarray(a)
    x = ina.reshape(5,5)
    data[0,:,:] = x
    a = [250,250,250,250,250,250,250,250,250,250,250,250,3250,250,250,250,250,250,250,250,250,250,250,250,250]
    ina = np.asarray(a)
    x = ina.reshape(5,5)
    data[1,:,:] = x
    a = [250,250,250,250,250,250,250,250,250,250,250,250,4250,250,250,250,250,250,250,250,250,250,250,250,250]
    ina = np.asarray(a)
    x = ina.reshape(5,5)
    data[2,:,:] = x
    a = [250,250,250,250,250,250,250,250,250,250,250,250,5250,250,250,250,250,250,250,250,250,250,250,250,250]
    ina = np.asarray(a)
    x = ina.reshape(5,5)
    data[3,:,:] = x

    psfs = np.ones((4,5,5))/1000.

    #psf = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
    #for epoch in np.arange(Nepochs):
    #    psfs[epoch,:,:] = psf

    weights = 1/(np.ones((4,5,5))+10)
    #weight = 1./(np.ones(25).reshape(5,5)+4.)
    #for epoch in np.arange(Nepochs):
    #    weights[epoch,:,:] = weight


    a = metropolis_hastings( model = model
        , stdev = stdev
        , data = data
        , psfs = psfs
        , weights = weights
        , substamp = substamp
        , Nimage = Nepochs
        )

    model, uncertainty, history = a.get_params()

    print 'FINAL MODEL'
    print model
    print 'MODEL Uncertainty'
    print uncertainty



