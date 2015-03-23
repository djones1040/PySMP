#!/usr/bin/env python
# D. Jones - 1/10/14
"""This code is from the IDL Astronomy Users Library
with modifications from Dan Scolnic.  

This code is from the IDL Astronomy Users Library
(adapted for IDL from DAOPHOT, then translated from IDL to Python).

Subroutine of GETPSF to perform a one-star least-squares fit, 
part of the DAOPHOT PSF photometry sequence.  This version requires
input noise and mask images and DOES NOT recenter the PSF.  The fitting
is done by a Levenberg-Marquardy least-squares algorithm
using mpfit - http://www.physics.wisc.edu/~craigm/idl/fitting.html

CALLING SEQUENCE:
     from PythonPhot import pkfit_norecent_noise_smp as pkfit
     pk = pkfit.pkfit_class(f, gauss, psf,
                            ronois, phpadu )
     errmag,chi,niter,scale,xnew,ynew = pk.pkfit(scale,x,y,sky,radius)

PKFIT CLASS INPUTS:
     f           - NX by NY array containing actual picture data.
     ronois      - readout noise per pixel, scalar
     phpadu      - photons per analog digital unit, scalar
     psf         - an array containing the PSF
     psfcenter   - a tuple containing the image x and y coordinates that correspond 
                   to the center of the psf
     noise_image - the noise image corresponding to f
     mask_image  - the mask image corresponding to f.  Masked pixels are not used.

PKFIT FUNCTION INPUTS:
     x, y    - the initial estimates of the centroid of the star relative
                to the corner (0,0) of the subarray.  Upon return, the
                final computed values of X and Y will be passed back to the
                calling routine.
     sky     - the local sky brightness value, as obtained from APER
     radius  - the fitting radius-- only pixels within RADIUS of the
                instantaneous estimate of the star's centroid will be
                included in the fit, scalar

OPTIONAL PKFIT FUNCTION INPUTS:
     maxiter - maximum iterations (default = 25)

INPUT-OUTPUT:
     scale  - the initial estimate of the brightness of the star,
               expressed as a fraction of the brightness of the PSF.
               Upon return, the final computed value of SCALE will be
               passed back to the calling routine.

RETURNS:
     errmag - the estimated standard error of the value of SCALE
               returned by this routine.
     chi    - the estimated goodness-of-fit statistic:  the ratio
               of the observed pixel-to-pixel mean absolute deviation from
               the profile fit, to the value expected on the basis of the
               noise as determined from Poisson statistics and the
               readout noise.
     niter  - the number of iterations the solution required to achieve
               convergence.  If NITER = 25, the solution did not converge.
               If for some reason a singular matrix occurs during the least-
               squares solution, this will be flagged by setting NITER = -1.
     image_stamp
     noise_stamp
     mask_stamp
     psf_stamp

EXAMPLE:
     import pyfits
     from PyIDLPhot import pkfit_norecent_noise as pkfit

     # read in the FITS images
     image = pyfits.getdata(fits_filename)
     noiseim = pyfits.getdata(fits_noise_filename)
     maskim = pyfits.getdata(fits__mask_filename)

     # read in the PSF image
     psf = pyfits.getdata(psf_filename)
     hpsf = pyfits.getheader(psf_filename)
     gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]

     # x and y points for PSF fitting
     xpos,ypos = np.array([1450,1400]),np.array([1550,1600])

     # run 'aper' on x,y coords to get sky values
     mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
              aper.aper(image,xpos,ypos,phpadu=1,apr=5,zeropoint=25,
              skyrad=[40,50],badpix=[-12000,60000],exact=True)

     # load the pkfit class
     pk = pkfit.pkfit_class(image,gauss,psf,1,1,noiseim,maskim)

     # do the PSF fitting
     for x,y,s in zip(xpos,ypos,sky):
          errmag,chi,sharp,niter,scale = \
              pk.pkfit_norecent_noise(1,x,y,s,5)
          flux = scale*10**(0.4*(25.-hpsf['PSFMAG']))
          dflux = errmag*10**(0.4*(25.-hpsf['PSFMAG']))
          print('PSF fit to coords %.2f,%.2f gives flux %s +/- %s'%(x,y,flux,dflux))
               
RESTRICTIONS:
     No parameter checking is performed

REVISON HISTORY:
     Adapted from the official DAO version of 1985 January 25
     Version 2.0                              W. Landsman STX             November,  1988
     Converted to IDL V5.0                    W. Landsman                 September, 1997
     Converted to Python                      D. Jones                    January,   2014
"""

import numpy as np
from scipy import linalg
from PythonPhot import dao_value
import mpfit
import mpfitexpr
sqrt,where,abs,shape,zeros,array,isnan,\
    arange,matrix,exp,sum,isinf,median,ones,bool = \
    np.sqrt,np.where,np.abs,np.shape,\
    np.zeros,np.array,np.isnan,\
    np.arange,np.matrix,np.exp,\
    np.sum,np.isinf,np.median,np.ones,np.bool

class pkfit_class:

    def __init__(self,image,psf, psfcenter, 
                 ronois,phpadu,
                 noise_image,mask_image):
        self.f = image
        self.psf = psf
        self.psfcenter = psfcenter
        self.fnoise = noise_image
        self.fmask = mask_image
        self.ronois = ronois
        self.phpadu = phpadu

    def pkfit_norecent_noise_smp(self,scale,x,y,sky,skyerr,radius,
                                 maxiter=25,stampsize=100,
                                 debug=False,returnStamps=False):
        f = self.f; psf = self.psf;
        fnoise = self.fnoise; fmask = self.fmask

        if f.dtype != 'float64': f = f.astype('float64')
#        psf1d = psf.reshape(shape(psf)[0]**2.)

        s = shape(f) #Get array dimensions
        nx = s[1] ; ny = s[0] #Initialize a few things for the solution

        redo = 0
        clamp = zeros(3) + 1.
        dtold = zeros(3)
        niter = 0
        chiold = 1.

        if debug:
            print('PKFIT: ITER  X      Y      SCALE    ERRMAG   CHI     SHARP')
            
        if isnan(x) or isnan(y):
            scale=1000000.0;
            errmag=100000
            chi=100000
            #sharp=100000
            #return(errmag,chi,sharp,niter,scale)
            return errmag, chi, niter, scale

#        loop=True
#        while loop:                        #Begin the big least-squares loop
#            niter = niter+1
            
        ixlo = int(x+0.5-radius)
        if ixlo < 0: ixlo = 0       #Choose boundaries of subarray containing
        iylo = int(y+0.5-radius)
        if iylo < 0: iylo = 0       # 3points inside the fitting radius
        ixhi = int(x+0.5+radius) #+1 
        if ixhi > (nx-1): ixhi = nx-1
        iyhi = int(y+0.5+radius) #+1
        if iyhi > ny-1: iyhi = ny-1
        ixx  = int(2*radius + 1) #ixhi-ixlo+1
        iyy  = int(2*radius + 1) #iyhi-iylo+1
        dy   = arange(iyy) + iylo - y    #X distance vector from stellar centroid
        dysq = dy**2
        dx   = arange(ixx) + ixlo - x
        dxsq = dx**2
        rsq  = zeros([iyy,ixx])  #RSQ - array of squared
            
        radsq = radius**2

        for j in range(iyy): rsq[j,:] = (dxsq+dysq[j])/radsq
            
            # The fitting equation is of the form
            #
            # Observed brightness =
            #      SCALE + delta(SCALE)  *  PSF + delta(Xcen)*d(PSF)/d(Xcen) +
            #                                           delta(Ycen)*d(PSF)/d(Ycen)
            #
            # and is solved for the unknowns delta(SCALE) ( = the correction to
            # the brightness ratio between the program star and the PSF) and
            # delta(Xcen) and delta(Ycen) ( = corrections to the program star's
            # centroid).
            #
            # The point-spread function is equal to the sum of the integral under
            # a two-dimensional Gaussian profile plus a value interpolated from
            # a look-up table.
            
            # D. Jones - noise edit from Scolnic
            #        good = where(rsq.reshape(shape(rsq)[0]*shape(rsq)[1]) < 1)[0]
            #        rsqshape = shape(rsq)
            #        fnoise_sub = fnoise[iylo:iyhi+1,ixlo:ixhi+1]#.reshape(rsqshape[0]*rsqshape[1])
            #        fmask_sub = fmask[iylo:iyhi+1,ixlo:ixhi+1]#.reshape(rsqshape[0]*rsqshape[1])
            #        good = where((rsq.reshape(rsqshape[0]*rsqshape[1]) < 1.) & 
            #                        (fnoise_sub > 0) &
                #                        (fmask_sub == 0))[0]
        good = where((rsq < 1.) & 
                     (fnoise[iylo:iyhi+1,ixlo:ixhi+1] > 0) &
                     (fmask[iylo:iyhi+1,ixlo:ixhi+1] == 0))

        
        ngood = len(good[0])
        if ngood < 1: ngood = 1

        bad_psf = where(rsq*radius**2. >= (radius-1)**2.)
        good_psf = where(rsq*radius**2. <= (radius-1)**2.)
        bad_pix = where((fnoise[iylo:iyhi+1,ixlo:ixhi+1] == 0) | 
                        (fmask[iylo:iyhi+1,ixlo:ixhi+1] != 0))
        
        good_local = where((rsq*radius**2. < ((radius-1)/2.)**2.) & 
                           (fnoise[iylo:iyhi+1,ixlo:ixhi+1] != 0) & 
                           (fmask[iylo:iyhi+1,ixlo:ixhi+1] == 0))
        
        t = zeros([3,ngood])
            #        sbd=shape(badpix)
            #        sdf=shape(df)

        if not len(good[0]) or not len(good_local[0]):
            # D. Jones - modified from Scolnic
            print 'good', good[0]
            print 'good_local', good_local[0]
            print 'Return1'
            scale=1000000.0;
            errmag=100000
            chi=100000
            sharp=100000
            if returnStamps: return (errmag,chi,niter,scale, np.zeros([stampsize,stampsize]), np.zeros([stampsize,stampsize])+1e8, np.zeros([stampsize,stampsize]), np.zeros([stampsize,stampsize])) 
            else: return(errmag,chi,niter,scale)
        if y < 50 or x < 50 or x > ny-50 or x > nx-50:
            # D. Jones - modified from Scolnic
            print('Star too close to the edge!')
            print('Returning...')
            scale=1000000.0;
            errmag=100000
            chi=100000
            sharp=100000
            return(errmag,chi,niter,scale)

#            dx = dx[good[1]]# % ixx]
#            dy = dy[good[0]]#/ixx]
#            model,dvdx,dvdy = dao_value.dao_value(dx, dy, gauss, 
#                                                  psf, #psf1d=psf1d, 
#                                                  deriv=True)#,ps1d=True)

            #        mshape = shape(model)
            #        if len(mshape) > 2:
            #            model = model.reshape(mshape[0]*mshape[1])

        # D. Jones - modified from Scolnic
#            if len(dvdx) == 0:
#                print 'Return2'
#                scale=1000000.0
#                errmag=100000
#                chi=100000
#                sharp=100000
#                return(errmag,chi,sharp,niter,scale)

        if debug: print('model created '); return(errmag,chi,niter,scale)

#            t[0,:] = model

            # D. Jones - modified from Scolnic
            #        sa=shape(dvdx)
            #        if sa[0] > ngood or len(sa) == 0:
            #            scale=0
            #            return(errmag,chi,sharp,niter)

#            t[1,:] = -scale*dvdx
#            t[2,:] = -scale*dvdy

        psf_stamp = np.zeros([stampsize,stampsize])
        noise_stamp = np.zeros([stampsize,stampsize])+1e8
        mask_stamp = np.zeros([stampsize,stampsize])
        image_stamp = np.zeros([stampsize,stampsize])

            #define what cen1, cen2 is
        imlen=(ixhi-ixlo)/2.0
        cen=stampsize/2.0

        model2=f[iylo:iyhi+1,ixlo:ixhi+1]*0.0

        model2[good_psf]=psf[np.shape(psf)[0]/2-radius:np.shape(psf)[0]/2+radius+1,
                             np.shape(psf)[1]/2-radius:np.shape(psf)[1]/2+radius+1][good_psf]
        
        psf_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1]=model2

        temp=psf_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1]
        if len(bad_psf[0]): temp[bad_psf]=0.0
        psf_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1]=temp
            
        ntemp=sqrt(f[iylo:iyhi+1,ixlo:ixhi+1]-sky)
        ntemp[np.where(ntemp < 0)] = 0
        ntemp += skyerr**2.
        if len(bad_pix[0]): ntemp[bad_pix]=10000000000.0

        noise_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1] = ntemp
            
        mask_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1] = fmask[iylo:iyhi+1,ixlo:ixhi+1]
        image_stamp[cen-imlen:cen+imlen+1,cen-imlen:cen+imlen+1] = f[iylo:iyhi+1,ixlo:ixhi+1]
        
        fsub = f[iylo:iyhi+1,ixlo:ixhi+1]
        
        fsub_full = f[iylo:iyhi+1,ixlo:ixhi+1]
        if  np.abs(f[ self.psfcenter[1], self.psfcenter[0]] - fsub[15,15]) > 0.2:
            raise Exception("Check PSF Center for rounding")
            # D. Jones - reshape arrays, python is less flexible than IDL here
            #        subshape = shape(fsub)
            #        fsub = fsub.reshape(subshape[0]*subshape[1])
            #        rsq = rsq.reshape(subshape[0]*subshape[1])
        fsub = fsub[good[0],good[1]]
            # D. Jones - added for noise version from Scolnic
        fsubnoise=fnoise[iylo:iyhi+1,ixlo:ixhi+1]
        fsubnoise_full=fnoise[iylo:iyhi+1,ixlo:ixhi+1]
            #        fsubnoise = fsubnoise.reshape(subshape[0]*subshape[1])
            # D. Jones - noise addition from Scolnic
#            fsubnoise = fsubnoise[good]
#            sig=fsubnoise
#            sigsq = fsubnoise**2.

        sig = np.zeros(len(good_local[0]))
        sig[:] = skyerr
        
        # model, image, err, sky
        vals = mpfitexpr.mpfitexpr(" p[0]*x+p[1] ",model2[good_local],fsub_full[good_local],sig, [1,sky], full_output=True)[0]
        
        errv=np.zeros(51)
        for h in range(51):
            try:
                errv[h]=np.sum((fsub_full[good_local]-sky-(vals.params[0]+h/10.0*vals.perror[0])*model2[good_local])**2./(fsubnoise[good_local]*0+skyerr)**2.)
            except:
                print "Output of mpfitexpr below. Likely this failure was due to a mask/weight file being all zeros near a bright star/galaxy."
                print "Returning infinite error and chi2"
                print vals
                scale=1000000.0;
                errmag=100000
                chi=100000
                sharp=100000
                if returnStamps: return (errmag,chi,niter,scale, np.zeros([stampsize,stampsize]), np.zeros([stampsize,stampsize])+1e8, np.zeros([stampsize,stampsize]), np.zeros([stampsize,stampsize]))
                else: return(errmag,chi,niter,scale)
        err23=np.min(np.abs(errv-errv[0]-2.3))
        ij = np.where(np.abs(errv-errv[0]-2.3) == np.min(np.abs(errv-errv[0]-2.3)))[0][0]
        errmag=ij/10.0*vals.perror[0]
        chi=vals.fnorm/vals.dof
        chi=ij/10.0
        scale=vals.params[0]

#            rsq = rsq[good[0],good[1]]
            # D. Jones - Scolnic lines removed by Scolnic
            # Scolnic Added!!!
            #
            #        yx=zeros(1)
            #        yx[0]=sky
            #        skys=yx[0]
            #        sky=skys

        if returnStamps:
            return(errmag,chi,vals.niter,scale,image_stamp,noise_stamp,mask_stamp,psf_stamp)
        else:
            return(errmag,chi,vals.niter,scale)
