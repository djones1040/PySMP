#!/usr/bin/env python
# Dillon Brout 3/10/2015
# dbrout@physics.upenn.edu

"""
Usage:
import mcmc
mcmc.run( model, stdev, data, psfs, weights, substamp, , Nimage )

1D arrays (all of same size)
model                 : contains all model parameters
stdev                 : contains stdev for each model parameter to be used as a random kick

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
from pylab import pcolor, show, colorbar, xticks, yticks
import pylab as p
import time



class metropolis_hastings():

    def __init__(self
                , model = None
                , stdev = None
                , data = None
                , psfs = None
                , weights = None
                , substamp = 0
                , Nimage = 1
                ):

        if model is None:
            raise AttributeError('Must provide model array!')
        if stdev is None:
            raise AttributeError('Must provide stdev for each model parameter!')
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
        else:
            if Nimage == 1:
                raise AttributeError('Must provide Nimage (number of epochs)!')      

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
        self.stdev = stdev
        self.deltas = copy(self.stdev) #this vec will change for each iter
        self.data = data
        self.psfs = psfs
        self.weights = weights
        self.substamp = substamp
        self.Nimage = Nimage

        self.galaxy_model = self.model[ 0 : self.substamp**2.].reshape(self.substamp,self.substamp)



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
        #self.t2 = time.time()

        while self.z_scores_say_keep_going:
            #self.t2 = time.time()
            self.counter += 1
            self.accepted_int += 1
            self.mcmc_func()
            
            #Check Geweke Convergence Diagnostic every 5000 iterations
            if (self.counter % 10000) == 9000: 
                self.check_geweke()
                self.last_geweke = self.counter

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
        #print 'inside mcmc func'
        #raw_input()

        #print self.model
        self.adjust_model()
        #print 'adjusted model'
        # Contains the convolution
        self.kernel()
        #print 'finished kernel'
       
        #Calculate Chisq over all epochs
        self.thischisq = self.chisq_sim_and_real()

        #decide whether to accept new values
        accept_bool = self.accept(self.lastchisq,self.thischisq)


        if accept_bool:
            #print 'this chi sq '+str(self.thischisq)+' last chi sq '+str(self.lastchisq)+ ' ACCEPTED'
            self.lastchisq = self.thischisq
            self.accepted_history = ( self.accepted_history * self.accepted_int + 1.0 ) / ( self.accepted_int + 1 )
            self.copy_adjusted_image_to_model()
            self.update_history()
            self.chisq.append( self.thischisq )
        else:
            #print 'this chi sq '+str(self.thischisq)+' last chi sq '+str(self.lastchisq)+ ' NOPE'
            self.accepted_history = ( self.accepted_history * self.accepted_int ) / ( self.accepted_int + 1 )
            self.update_unaccepted_history()

    def adjust_model( self ):
        for i in np.arange( len( self.deltas ) ):
            if self.stdev[ i ] > 0:
                self.deltas[ i ] = np.random.normal( scale= self.stdev[ i ] )
            else:
                self.deltas[ i ] = 0.0
        #print self.deltas
        self.kicked_model = self.model + self.deltas
        self.kicked_galaxy_model = self.kicked_model[ 0 : self.substamp**2. ].reshape( self.substamp, self.substamp )
        #print self.model
        #print self.kicked_galaxy_model
        #raw_input()
        return

    def kernel( self ):
        for epoch in np.arange( self.Nimage ):
            #print 'model and psf'
            #print self.kicked_galaxy_model
            #print self.psfs[ epoch, : , : ]
            #raw_input()
            galaxy_conv = scipy.ndimage.convolve( self.kicked_galaxy_model, self.psfs[ epoch, : , : ] )
            star_conv = self.kicked_model[self.substamp**2. + epoch ] * self.psfs[ epoch, : , : ]
            #print 'convoluations'
            #print galaxy_conv
            #print star_conv
            self.sims[ epoch, : , : ] =  star_conv + galaxy_conv
            #print  star_conv + galaxy_conv

            #raw_input()

    def chisq_sim_and_real( self ):
        chisq = 0.0
        for epoch in np.arange( self.Nimage ):
            chisq += np.sum( ( (self.sims[ epoch, : , : ] - self.data[ epoch, : , : ])**2 * self.weights[ epoch, : , : ] ).ravel() )

        return chisq

    def accept( self, last_chisq, this_chisq ):
        alpha = np.exp( last_chisq - this_chisq ) / 2.0
        print alpha
        raw_input()
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

    def get_model( self ):
        return self.model_params, self.model_uncertainty, self.nphistory # size: self.history[num_iter,len(self.model_params)]

    def make_history( self ):
        num_iter = len( self.history )
        self.nphistory = np.zeros( (num_iter , len( self.model )) ) 
        for i in np.arange( num_iter ):
            self.nphistory[ i , : ] = self.history[ i ]

        #self.nphistory
        #p.plot(self.nphistory[:,-4])
        #p.plot(self.nphistory[:,-3])
        #p.plot(self.nphistory[:,-2])
        #p.plot(self.nphistory[:,-1])
        #p.show()

    #DIAGNOSTICS

    def check_geweke( self, zscore_mean_crit=.7, zscore_std_crit=1.0 ):
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
                if abs(mean) > zscore_mean_crit:
                    alltrue = False
        if alltrue:
            for std in stdevs:
                if alltrue:
                    if std > zscore_std_crit:
                        alltrue = False
        if alltrue:
            self.z_scores_say_keep_going = False
            print 'Zscores computed and convergence criteria has been met'
        else:
            print 'Zscores computed and convergence criteria have not been met, mcmc will continue...'

        #print means
        #raw_input()
        #if zscores arent improving then up the stdevs by a percentage. if they do improve decrease them by a percentage.

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

    model, uncertainty, history = a.get_model()

    print 'FINAL MODEL'
    print model
    print 'MODEL Uncertainty'
    print uncertainty



