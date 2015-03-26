#!/usr/bin/env python
# D. Jones - 11/24/14
"""
Scene modeling pipeline for DES and PanSTARRS.

Usage:

smp.py -s supernova_file -p parameter_file\
 --nomask --nodiff --nozpt -r root_dir -f filter 

-s/--supernova_file  : Filename with all the information about
                       the supernova.
-p/--params          : Parameter file
-r/--root_dir        : images root directory
#-f/--filter          : observation filter, use 'all' for all filters
--nomask             : set if no mask image exists (one will be 
                       created).
--nodiff             : set if no difference image exists
--nozpt              : set if zeropoints have not been measured
--debug              : debug flag saves intermediate products
                       and prints additional information

TEST - Dillon
"""


import numpy as np
import exceptions
import os
import scipy.ndimage
import mcmc
import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt

#from matplotlib.backends.backend_pdf import PdfPages

snkeywordlist = {'SURVEY':'string','SNID':'string','FILTERS':'string',
                 'PIXSIZE':'float','NXPIX':'float','NYPIX':'float',
                 'ZPFLUX':'float','RA':'string', 
                 'DECL':'string','PEAKMJD':'float','WEIGHT_BADPIXEL':'string',
                 'CATALOG_FILE':'string', 'IMAGE_DIR':'string',
                 'PSF_UNIT':'string', 
                 'NOBS':'float','PLATESCALE':'float','PSF_MODEL':'string'}
snvarnameslist = {'ID_OBS':'string','MJD':'float','BAND':'string',
                  'IMAGE_NAME_SEARCH':'string','IMAGE_NAME_WEIGHT':'string',
                  'IMAGE_NAME_PSF':'string','CATALOG_FILE':'string'}
paramkeywordlist = {'STAMPSIZE':'float','RADIUS1':'float',
                    'RADIUS2':'float','SUBSTAMP':'float',
                    'MAX_MASKNUM':'float','RDNOISE_NAME':'string',
                    'GAIN_NAME':'string','FWHM_MAX':'float',
                    'PSF_MAX':'float','NOISE_TYPE':'string',
                    'MASK_TYPE':'string','MJDPLUS':'float','MJDMINUS':'float',
                    'BUILD_PSF':'string','CNTRD_FWHM':'float','FITRAD':'float',
                    'FIND_ZPT':'string'}

class get_snfile:
    def __init__(self,snfile, rootdir):
        varnames = ''
        fin = open(snfile,'r')
        for line in fin:
            line = line.replace('\n','')
            if not line.startswith('#') and line.replace(' ',''):
                if not line.replace(' ','').startswith('OBS:') and \
                        not line.replace(' ','').startswith('VARNAMES:'):
                    key,val = line.split('#')[0].split(':')
                    key = key.replace(' ','')
                    if key.upper() != 'WEIGHT_BADPIXEL' and (key.upper() != 'CATALOG_FILE' or not 'des' in snfile):             
                        val = val.split()[0]
                        val = val.replace(' ','')
                        self.__dict__[key.lower()] = val
                    elif key.upper() == 'CATALOG_FILE' and 'des' in snfile:
                        catfilter = val.split()[0]                        
                        if filt.lower() == catfilter.lower():
                            print val
                            self.__dict__["catalog_file"] = {catfilter.lower(): rootdir + val.split()[1]}
                        elif filt.lower() == 'all':
                            if "catalog_file" in self.__dict__:
                                self.__dict__["catalog_file"][val.split()[0]] = rootdir + val.split()[1]
                            else:
                                self.__dict__["catalog_file"] = {}
                                self.__dict__["catalog_file"][val.split()[0]] = rootdir + val.split()[1]
                    else:
                        try:
                            self.__dict__[key.lower()] = np.array(val.split()).astype('float')
                        except:
                            raise exceptions.RuntimeError("Error : WEIGHT_BADPIXEL cannot be parsed!")
                
                #elif line.replace(' ','').startswith('VARLIST:'):
                elif line.split(' ')[0] == 'VARNAMES:':
                    varnames = filter(None,line.split('VARNAMES:')[-1].split(' '))
                    for v in varnames:
                        self.__dict__[v.lower()] = np.array([])
                elif line.replace(' ','').startswith('OBS:'):
                    vals = filter(None,line.split(' '))[1:]
                    if not varnames:
                        raise exceptions.RuntimeError("Error : Variable names are not defined!!")
                    elif len(varnames) != len(vals):
                        raise exceptions.RuntimeError("Error : Number of variables provided is different than the number defined!!!")
                    for var,val in zip(varnames,vals):
                        self.__dict__[var.lower()] = np.append(self.__dict__[var.lower()],val)

        catalog_exists = True
        for p in snkeywordlist.keys():
            if not self.__dict__.has_key(p.lower()):
                if p.lower() != 'catalog_file':
                    raise exceptions.RuntimeError("Error : keyword %s doesn't exist in supernova file!!!"%p)
                else:
                    catalog_exists = False
            if snkeywordlist[p] == 'float':
                try:
                    self.__dict__[p.lower()] = float(self.__dict__[p.lower()])
                except:
                    raise exceptions.RuntimeError('Error : keyword %s should be set to a number!'%p)

        for p in snvarnameslist.keys():
            if not self.__dict__.has_key(p.lower()):
                if p.lower() != 'catalog_file':
                    raise exceptions.RuntimeError("Error : field %s doesn't exist in supernova file!!!"%p)
                elif catalog_exists == False:
                    raise exceptions.RuntimeError("Error : field %s doesn't exist in supernova file!!!"%p)
            if snvarnameslist[p] == 'float':
                try:
                    self.__dict__[p.lower()] = self.__dict__[p.lower()].astype('float')
                except:
                    raise exceptions.RuntimeError('Error : keyword %s should be set to a number!'%p)
        #print self.__dict__['catalog_file']
        #raw_input()

class get_params:
    def __init__(self,paramfile):
        fin = open(paramfile,'r')
        for line in fin:
            line = line.replace('\n','')
            if not line.startswith('#') and line.replace(' ',''):
                try:
                    key,val = line.split('#')[0].split(':')
                except:
                    raise exceptions.RuntimeError('Invalid format!  Should be key: value')
                key = key.replace(' ','')
                val = val.replace(' ','')
                self.__dict__[key.lower()] = val
        for p in paramkeywordlist.keys():
            if not self.__dict__.has_key(p.lower()):
                raise exceptions.RuntimeError("Error : keyword %s doesn't exist in parameter file!!!"%p)
            if paramkeywordlist[p] == 'float':
                try:
                    self.__dict__[p.lower()] = float(self.__dict__[p.lower()])
                except:
                    raise exceptions.RuntimeError('Error : keyword %s should be set to a number!'%p)

class smp:
    def __init__(self,snparams,params,zpt_fits='./zpts/zpt_plots.txt',big_zpt_compare='./zpts/big_zpt',clear_zpt=False):
        self.snparams = snparams
        self.params = params
        self.zpt_fits = zpt_fits
        self.big_zpt = big_zpt_compare
        a = open(zpt_fits,'w')
        a.write('ZPT FILE LOCATIONS\n')
        a.close()
        if clear_zpt:
            big = open(self.big_zpt+'.txt','w')
            big.write('RA\tDEC\tZpt\tCat Mag\tMP Fit Mag\tMCMC Fit Mag\n')
            big.close()

    def main(self,nodiff=False,getzpt=False,
             nomask=False,outfile='',debug=False,
             verbose=False):
        from txtobj import txtobj
        from astLib import astWCS
        from PythonPhot import cntrd,aper,getpsf,rdpsf
        from mpfit import mpfit
        import astropy.io.fits as pyfits
        import pkfit_norecent_noise_smp

        self.verbose = verbose

        params,snparams = self.params,self.snparams

        if snparams.psf_model == 'psfex' and not snparams.has_key('psf_fwhm'):
            raise exceptions.RuntimeError('Error : PSF_FWHM must be provided in supernova file!!!')

        smp_im = np.zeros([snparams.nobs,params.substamp,params.substamp])
        smp_noise = np.zeros([snparams.nobs,params.substamp,params.substamp])
        smp_psf = np.zeros([snparams.nobs,params.substamp,params.substamp])
#        smp_bigim = np.zeros([snparams.nobs,params.stampsize,params.stampsize])
#        smp_bignoise = np.zeros([snparams.nobs,params.stampsize,params.stampsize])
#        smp_bigpsf = np.zeros([snparams.nobs,params.stampsize,params.stampsize])


        smp_dict = {'scale':np.zeros(snparams.nobs),
                    'scale_err':np.zeros(snparams.nobs),
                    'image_scalefactor':np.zeros(snparams.nobs),
                    'snx':np.zeros(snparams.nobs),
                    'sny':np.zeros(snparams.nobs),
                    'fwhm_arcsec':np.zeros(snparams.nobs),
                    'sky':np.zeros(snparams.nobs),
                    'flag':np.ones(snparams.nobs),
                    'psf':np.zeros(snparams.nobs),
                    'zpt':np.zeros(snparams.nobs),
                    'mjd':np.zeros(snparams.nobs),
                    'mjd_flag':np.zeros(snparams.nobs)}
        smp_scale = np.zeros(snparams.nobs)
        smp_sky = np.zeros(snparams.nobs)
        smp_flag = np.zeros(snparams.nobs)

#        if not nodiff:
#            smp_diff = smp_im[:,:,:]

        #print snparams.catalog_file.keys()
        #raw_input()

        for imfile,noisefile,psffile,band,i in \
                zip(snparams.image_name_search,snparams.image_name_weight,snparams.image_name_psf,snparams.band,
                    range(len(snparams.image_name_search))):
                
            if filt != 'all' and band not in filt:
                if verbose: print('filter %s not in filter list for image file %s'%(band,filt,imfile))
                continue
            imfile,noisefile,psffile = '%s/%s'%(snparams.image_dir,imfile),\
                '%s/%s'%(snparams.image_dir,noisefile),'%s/%s'%(snparams.image_dir,psffile)
            if not os.path.exists(imfile):
                raise exceptions.RuntimeError('Error : file %s does not exist'%imfile)
            if not os.path.exists(noisefile):
                os.system('gunzip %s.gz'%noisefile)
                if not os.path.exists(noisefile):
                    raise exceptions.RuntimeError('Error : file %s does not exist'%noisefile)
            if not os.path.exists(psffile):
                raise exceptions.RuntimeError('Error : file %s does not exist'%psffile)

            if not nomask:
                maskfile = '%s/%s'%(snparams.image_dir,snparams.image_name_mask[i])

                if not os.path.exists(maskfile):
                    os.system('gunzip %s.gz'%maskfile)
                    if not os.path.exists(maskfile):
                        raise exceptions.RuntimeError('Error : file %s does not exist'%maskfile)

            # read in the files
            im = pyfits.getdata(imfile)
            hdr = pyfits.getheader(imfile)
            noise = pyfits.getdata(noisefile)
            psf = pyfits.getdata(psffile)

            if params.weight_type.lower() == 'ivar':
                noise = np.sqrt(1/noise)
            elif params.weight_type.lower() != 'noise':
                raise exceptions.RuntimeError('Error : WEIGHT_TYPE value %s is not a valid option'%params.WEIGHT_TYPE)
            if nomask:
                mask = np.zeros(np.shape(noise))
                maskcols = np.where((noise < 0) |
                                    (np.isfinite(noise) == False))
                mask[maskcols] = 100.0
            else:
                mask = pyfits.getdata(maskfile)

            wcs = astWCS.WCS(imfile)
            ra1,dec1 = wcs.pix2wcs(0,0)
            ra2,dec2 = wcs.pix2wcs(snparams.nxpix-1,
                                   snparams.nypix-1)
            ra_high = np.max([ra1,ra2])
            ra_low = np.min([ra1,ra2])
            dec_high = np.max([dec1,dec2])
            dec_low = np.min([dec1,dec2])
            try:
                snparams.RA = float(snparams.ra)
                snparams.DECL = float(snparams.decl)
            except:
                try:
                    snparams.RA = astCoords.hms2decimal(snparams.ra,':')
                    snparams.DECL = astCoords.dms2decimal(snparams.decl,':')
                except:
                    raise exceptions.RuntimeError('Error : RA/Dec format unrecognized!!')

            xsn,ysn = wcs.wcs2pix(snparams.RA,snparams.DECL)
            if xsn < 0 or ysn < 0 or xsn > snparams.nxpix-1 or ysn > snparams.nypix-1:
                raise exceptions.RuntimeError("Error : SN Coordinates %s,%s are not within image"%(snparams.ra,snparams.decl))

            #print snparams.catalog_file
            #raw_input()
            if type(snparams.catalog_file) == np.array:
                if os.path.exists(snparams.catalog_file[i]):
                    starcat = txtobj(snparams.catalog_file[i],useloadtxt=True)
                    if not starcat.__dict__.has_key('mag'):
                        try:
                            starcat.mag = starcat.__dict__[band]
                            starcat.dmag = starcat.__dict__['d%s'%band]
                        except:
                            raise exceptions.RuntimeError('Error : catalog file %s has no mag column!!'%snparams.catalog_file[i])
                else: 
                    raise exceptions.RuntimeError('Error : catalog file %s does not exist!!'%snparams.catalog_file[i])
            elif type(snparams.catalog_file) == dict and 'des' in snfile:
                if os.path.exists(snparams.catalog_file[band]):
                    starcat = txtobj(snparams.catalog_file[band],useloadtxt=True, des=True)
                    if not starcat.__dict__.has_key('mag_%s'%band):
                        try:
                            print starcat.__dict__
                            starcat.mag = starcat.__dict__[band]
                            starcat.dmag = starcat.__dict__['d%s'%band]
                        except:
                            raise exceptions.RuntimeError('Error : catalog file %s has no mag column!!'%snparams.catalog_file[band])
                else: 
                    raise exceptions.RuntimeError('Error : catalog file %s does not exist!!'%snparams.catalog_file[band])
            else:
                if os.path.exists(snparams.catalog_file[i]):
                    starcat = txtobj(snparams.catalog_file[i],useloadtxt=True)
                    if not starcat.__dict__.has_key('mag'):
                        try:
                            starcat.mag = starcat.__dict__[band]
                            starcat.dmag = starcat.__dict__['d%s'%band]
                        except:
                            print snparams.catalog_file
                            raise exceptions.RuntimeError('Error : catalog file %s has no mag column!!'%snparams.catalog_file[i])

                else: 
                    raise exceptions.RuntimeError('Error : catalog file %s does not exist!!'%snparams.catalog_file)
 
            if snparams.psf_model.lower() == 'daophot':
                if params.build_psf == 'yes':
                    cols = np.where((starcat.ra > ra_low) & 
                                    (starcat.ra < ra_high) & 
                                    (starcat.dec > dec_low) & 
                                    (starcat.dec < dec_high))[0]
                    if not len(cols):
                        raise exceptions.RuntimeError("Error : No stars in image!!")
                    
                    mag_star = starcat.mag[cols]
                    x_star,y_star = wcs.wcs2pix(starcat.ra[cols],starcat.dec[cols])
                    x_star,y_star = cntrd.cntrd(im,x_star,y_star,params.cntrd_fwhm)
                    mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
                        aper.aper(im,x_star,y_star,apr = params.fitrad)

                    self.rdnoise = hdr[params.rdnoise_name]
                    self.gain = hdr[params.gain_name]
                    if not os.path.exists(psffile) or params.clobber_psf == 'yes':
                        gauss,psf,magzpt = getpsf.getpsf(im,x_star,y_star,mag,sky,
                                                         hdr[params.rdnoise_name],hdr[params.gain_name],
                                                         range(len(x_star)),params.fitrad,
                                                         psffile)
                        hpsf = pyfits.getheader(psffile)
                        #self.gauss = gauss
                    else:
                        print('PSF file exists.  Not clobbering...')
                        hpsf = pyfits.getheader(psffile)
                        magzpt = hpsf['PSFMAG']
                        #self.gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]
                elif getzpt:
                    self.rdnoise = hdr[params.rdnoise_name]
                    self.gain = hdr[params.gain_name]

                    cols = np.where((starcat.ra > ra_low) & 
                                    (starcat.ra < ra_high) & 
                                    (starcat.dec > dec_low) & 
                                    (starcat.dec < dec_high))[0]

                    if not len(cols):
                        raise exceptions.RuntimeError("Error : No stars in image!!")
                    
                    mag_star = starcat.mag[cols]
                    coords = wcs.wcs2pix(starcat.ra[cols],starcat.dec[cols])
                    x_star,y_star = [],[]
                    for c in coords:
                        x_star += [c[0]]
                        y_star += [c[1]]
                    x_star,y_star = np.array(x_star),np.array(y_star)
                    x_star,y_star = cntrd.cntrd(im,x_star,y_star,params.cntrd_fwhm)
                    mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
                        aper.aper(im,x_star,y_star,apr = params.fitrad)

                    hpsf = pyfits.getheader(psffile)
                    magzpt = hpsf['PSFMAG']
                    #self.gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]
                else:
                    hpsf = pyfits.getheader(psffile)
                    magzpt = hpsf['PSFMAG']
                    #self.gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]
                    self.rdnoise = hdr[params.rdnoise_name]
                    self.gain = hdr[params.gain_name]


                #fwhm = 2.355*self.gauss[3]

            # begin taking PSF stamps
            if snparams.psf_model.lower() == 'psfex':
                
                self.psf, self.psfcenter= self.build_psfex(psffile,xsn,ysn)
            elif snparams.psf_model.lower() == 'daophot':
                self.psf = rdpsf.rdpsf(psffile)[0]/10.**(0.4*(25.-magzpt))
            else:
                raise exceptions.RuntimeError("Error : PSF_MODEL not recognized!")

            if getzpt:
                zpt,zpterr = self.getzpt(x_star,y_star,starcat.ra[cols], starcat.dec[cols],starcat,mag,sky,skyerr,badflag,mag_star,im,noise,mask,psffile,imfile,psf=self.psf)
            else:
                try:
                    zpt = float(snparams.image_zpt[i])
                except:
                    print('Warning : IMAGE_ZPT field does not exist!  Calculating')
                    self.rdnoise = hdr[params.rdnoise_name]
                    self.gain = hdr[params.gain_name]

                    cols = np.where((starcat.ra > ra_low) & 
                                    (starcat.ra < ra_high) & 
                                    (starcat.dec > dec_low) & 
                                    (starcat.dec < dec_high))[0]

                    if not len(cols):
                        raise exceptions.RuntimeError("Error : No stars in image!!")
                    try:
                        #print starcat.mag_i
                        #raw_input()
                        if band.lower() == 'g':
                            mag_star = starcat.mag_g[cols]
                        elif band.lower() == 'r':
                            mag_star = starcat.mag_r[cols]
                        elif band.lower() == 'i':
                            mag_star = starcat.mag_i[cols]
                        elif band.lower() == 'z':
                            mag_star = starcat.mag_z[cols]
                        else:
                            raise Exception("Throwing all instances where mag_%band fails to mag. Should not appear to user.")
                    except:
                        mag_star = starcat.mag[cols]
                    coords = wcs.wcs2pix(starcat.ra[cols],starcat.dec[cols])
                    x_star,y_star = [],[]
                    for c in coords:
                        x_star += [c[0]]
                        y_star += [c[1]]
                    x_star,y_star = np.array(x_star),np.array(y_star)
                    x_star,y_star = cntrd.cntrd(im,x_star,y_star,params.cntrd_fwhm)
                    mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
                        aper.aper(im,x_star,y_star,apr = params.fitrad)
                    zpt,zpterr = self.getzpt(x_star,y_star,starcat.ra[cols], starcat.dec[cols],starcat,mag,sky,skyerr,badflag,mag_star,im,noise,mask,psffile,imfile,psf=self.psf)    
            if i == 0: firstzpt = zpt
            if zpt != 0.0 and np.min(self.psf) > -10000:
                scalefactor = 10.**(-0.4*(zpt-firstzpt))
            im *= scalefactor
            im[np.where(mask != 0)] =-999999.0

            if xsn > 25 and ysn > 25 and xsn < snparams.nxpix-25 and ysn < snparams.nypix-25:
                magsn,magerrsn,fluxsn,fluxerrsn,skysn,skyerrsn,badflag,outstr = \
                        aper.aper(im,xsn,ysn,apr = params.fitrad)
                if np.sum(mask[ysn-params.fitrad:ysn+params.fitrad+1,xsn-params.fitrad:xsn+params.fitrad+1]) != 0:
                    badflag = 1
                if skysn < -1e5: badflag = 1
                if not badflag:
                    pk = pkfit_norecent_noise_smp.pkfit_class(im,self.psf,self.psfcenter,self.rdnoise,self.gain,noise,mask)
                    #pk = pkfit_norecent_noise_smp.pkfit_class(im,self.gauss,self.psf,self.rdnoise,self.gain,noise,mask)
                    errmag,chi,niter,scale,image_stamp,noise_stamp,mask_stamp,psf_stamp = \
                        pk.pkfit_norecent_noise_smp(1,xsn,ysn,skysn,skyerrsn,params.fitrad,returnStamps=True,
                                                    stampsize=params.substamp)

                if snparams.psf_model.lower() == 'psfex':
                    fwhm = float(snparams.psf_fwhm[i])
                if snparams.psf_unit.lower() == 'arcsec':
                    fwhm_arcsec = fwhm
                elif snparams.psf_unit.lower().startswith('pix'):
                    print snparams.psf_model.lower()
                    fwhm_arcsec = fwhm*snparams.platescale
                else:
                    raise exceptions.RuntimeError('Error : FWHM units not recognized!!')

                if not badflag and fwhm_arcsec < params.fwhm_max and \
                        np.min(im[ysn-2:ysn+3,xsn-2:xsn+3]) != np.max(im[ysn-2:ysn+3,xsn-2:xsn+3]) and \
                        len(np.where(mask[ysn-25:ysn+26,xsn-25:xsn+26] != 0)[0]) < params.max_masknum and \
                        np.max(psf_stamp[params.substamp/2+1-3:params.substamp/2+1+4,params.substamp/2+1-3:params.substamp/2+1+4]) == np.max(psf_stamp[:,:]):
                    smp_im[i,:,:] = image_stamp
                    smp_noise[i,:,:] = noise_stamp
                    smp_psf[i,:,:] = psf_stamp
                    #smp_bigim[i,:,:] = bigimage_stamp
                    #smp_bignoise[i,:,:] = bignoise_stamp
                    #smp_bigpsf[i,:,:] = bigpsf_stamp

                    smp_dict['scale'][i] = scale
                    smp_dict['scale_err'][i] = errmag
                    smp_dict['sky'][i] = skysn
                    smp_dict['flag'][i] = 0
                    smp_dict['zpt'][i] = zpt
                    smp_dict['mjd'][i] = float(snparams.mjd[i])
                    smp_dict['image_scalefactor'][i] = scalefactor
                    smp_dict['snx'][i] = xsn
                    smp_dict['sny'][i] = ysn
                    smp_dict['fwhm_arcsec'][i] = fwhm_arcsec
                    if smp_dict['mjd'][i] < snparams.peakmjd - params.mjdminus or \
                                      smp_dict['mjd'][i] > snparams.peakmjd + params.mjdplus:
                        smp_dict['mjd_flag'][i] = 1

        # Now all the images are in the arrays
        # Begin the fitting
        badnoisecols = np.where(smp_noise <= 1)
        smp_noise[badnoisecols] = 1e10
        badpsfcols = np.where(smp_psf < 0)
        smp_noise[badpsfcols] = 1e10
        smp_psf[badpsfcols] = 0.0

#        badnoisecols = np.where(smp_bignoise <= 1)
#        smp_bignoise[badnoisecols] = 1e10
#        badpsfcols = np.where(smp_bigpsf < 0)
#        smp_bignoise[badpsfcols] = 1e10
#        smp_bigpsf[badpsfcols] = 0.0

        # data can't be sky subtracted with this cut in place
        infinitecols = np.where((smp_im == 0) | (np.isfinite(smp_im) == 0))
        smp_noise[infinitecols] = 1e10
        smp_im[infinitecols] = 0

        mpparams = np.concatenate((np.zeros(float(params.substamp)**2.),smp_dict['scale'],smp_dict['sky']))
        mpdict = [{'value':'','step':0,
                  'relstep':0,'fixed':0} for i in range(len(mpparams))]
        # provide an initial guess - CHECK

        maxcol = np.where(smp_im[0,:,:].reshape(params.substamp**2.) == np.max(smp_im[0,:,:]))[0][0]

        mpparams[maxcol+1] = np.max(smp_im[0,:,:])/np.max(smp_psf[0,:,:])
        for i in range(len(mpparams)):
            mpdict[i]['value'] = mpparams[i]
        for col in np.where((smp_dict['mjd_flag'] == 1) | (smp_dict['flag'] == 1))[0]+int(params.substamp)**2:
            mpdict[col]['fixed'] = 1
            mpdict[col]['value'] = 0
        for col in range(int(params.substamp)**2+len(smp_dict['scale'])):
            mpdict[col]['step']=np.max(smp_dict['scale'])
        #Fixing Galaxy Model at zero. Only for time testing purposes. Remove before use.
        for col in range(1, int(float(params.substamp)**2)):
            mpdict[col]['fixed'] = 1
            mpdict[col]['value'] = 0
        mpargs = {'x':smp_psf,'y':smp_im,'err':smp_noise,'params':params}
        
        if verbose: print('Creating Initial Scene Model')
        first_result = mpfit(scene,parinfo=mpdict,functkw=mpargs)
        for i in range(len(first_result.params)):
            mpdict[i]['value'] = first_result.params[i]
        if verbose: print('Creating Final Scene Model')
        second_result = mpfit(scene,parinfo=mpdict,functkw=mpargs)

        chi2 = scene_check(second_result.params,x=smp_psf,y=smp_im,err=smp_noise,params=params)

        # write the results to file
        fout = open(outfile,'w')
        print >> fout, '# MJD ZPT Flux Fluxerr pkflux pkfluxerr xpos ypos chi2 mjd_flag flux_firstiter fluxerr_firstoter'
        for i in range(len(smp_dict['x'])):
            print >> fout, '%.1f %.3f %3f %.3f %.3f %.3f %.2f %.2f %.2f %i %.3f %.3f'%(smp_dict['mjd'][i],smp_dict['zpt'][i],
                                                                                       second_result.params[params.substamp**2.+i],
                                                                                       second_result.perror[params.substamp**2.+i],
                                                                                       smp_dict['scale'][i],smp_dict['scale_err'][i],
                                                                                       smp_dict['x'],smp_dict['y'],chi2[i],
                                                                                       smp_dict['mjd_flag'][i],
                                                                                       first_result.params[params.substamp**2.+i],
                                                                                       first_result.perror[params.substamp**2.+i])
        fout.close()
        #self.big_zpt_plot()
        print('SMP was successful!!!')


    def getzpt(self,xstar,ystar,ras, decs,starcat,mags,sky,skyerr,badflag,mag_cat,im,noise,mask,psffile,imfile,psf='',mpfit_or_mcmc='mpfit'):
        """Measure the zeropoints for the images"""
        import pkfit_norecent_noise_smp
        from PythonPhot import iterstat
        import astropy.io.fits as pyfits
        #from PythonPhot import pkfit_norecent_noise


        flux_star = np.array([-999.]*len(xstar))
        flux_star_mcmc = np.array([-999.]*len(xstar))
        for x,y,m,s,se,i in zip(xstar,ystar,mags,sky,skyerr,range(len(xstar))):
            if x > 51 and y > 51 and x < self.snparams.nxpix-51 and y < self.snparams.nypix-51:
                if self.snparams.psf_model.lower() == 'psfex':
                    psf, psfcenter = self.build_psfex(psffile,x,y)
                elif psf == '':
                    raise exceptions.RuntimeError("Error : PSF array is required!")
                #Initialize
                pk = pkfit_norecent_noise_smp.pkfit_class(im,psf,psfcenter,self.rdnoise,self.gain,noise,mask)
                #Run for MPFIT
                errmag,chi,niter,scale = pk.pkfit_norecent_noise_smp(1,x,y,s,se,self.params.fitrad,mpfit_or_mcmc='mpfit')
                flux_star[i] = scale #write file mag,magerr,pkfitmag,pkfitmagerr and makeplots
                ##Run for MCMC
                #errmag_mcmc,chi_mcmc,niter_mcmc,scale_mcmc = pk.pkfit_norecent_noise_smp(1,x,y,s,se,self.params.fitrad,mpfit_or_mcmc='mcmc')
                #flux_star_mcmc[i] = scale_mcmc

        badflag = badflag.reshape(np.shape(badflag)[0])
        
        #check for only good fits MPFIT        
        goodstarcols = np.where((mag_cat != 0) & 
                                (flux_star != 1) & 
                                (flux_star < 1e7) &
                                (np.isfinite(mag_cat)) &
                                (np.isfinite(flux_star)) &
                                (flux_star > 0) &
                                (badflag == 0))[0]

        ##check for only good fits MCMC      
        #goodstarcols_mcmc = np.where((mag_cat != 0) & 
        #                        (flux_star_mcmc != 1) & 
        #                        (flux_star_mcmc < 1e7) &
        #                        (np.isfinite(mag_cat)) &
        #                        (np.isfinite(flux_star_mcmc)) &
        #                        (flux_star_mcmc > 0) &
        #                        (badflag == 0))[0]        

        #Writing mags out to file .zpt in same location as image
        mag_compare_out = imfile.split('.')[-2] + '.zpt'
        f = open(mag_compare_out,'w')
        f.write('RA\tDEC\tCat Mag\tMP Fit Mag\tMCMC Fit Mag\n')
        for i in goodstarcols:
            f.write(str(ras[i])+'\t'+str(decs[i])+'\t'+str(mag_cat[i])+'\t'+str(-2.5*np.log10(flux_star[i]))+'\t0.0\n')  
        f.close()


        #NEED TO MAKE A PLOT HERE!
        if len(goodstarcols) > 10:
            md,std = iterstat.iterstat(mag_cat[goodstarcols]+2.5*np.log10(flux_star[goodstarcols]),
                                       startMedian=True,sigmaclip=3.0,iter=10)
            zpt_plots_out = mag_compare_out = imfile.split('.')[-2] + '_mpfit_zptPlots'
            self.make_zpt_plots(zpt_plots_out,goodstarcols,mag_cat,flux_star,md,starcat)
            b = open(self.big_zpt+'.txt','a')
            for i in goodstarcols:
                b.write(str(ras[i])+'\t'+str(decs[i])+'\t'+str(md)+'\t'+str(mag_cat[i])+'\t'+str(-2.5*np.log10(flux_star[i]))+'\t0.0\n')
            b.close()
        else:
            raise exceptions.RuntimeError('Error : not enough good stars to compute zeropoint!!!')

        if self.verbose:
            print('measured ZPT: %.3f +/- %.3f'%(md,std))

        return(md,std)

    def make_zpt_plots(self,filename,goodstarcols,mag_cat,flux_star,zpt,starcat):
        #print starcat.mag_i
        #print starcat.__dict__.keys()
        #raw_input()
        plt.subplot2grid((5, 2), (2, 0), rowspan=1, colspan=1)
        grid_size = (5, 2)
        plt.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=2)

        #Fit Mag vs Catalog Mag
        plt.scatter(mag_cat[goodstarcols],2.5*np.log10(flux_star[goodstarcols]))
        plt.plot(np.arange(0.,100.,1.),-1*np.arange(0.,100.,1.)+zpt)
        plt.ylabel('2.5*log10(counts), zpt='+str(round(zpt,3)))
        plt.xlabel('Catalog Mag')
        plt.xlim(xmax = 23, xmin = 16)
        plt.ylim(ymax = 20, ymin = 2)
        
        #Catalog Mag - Fit Mag vs Fit Mag
        plt.subplot2grid(grid_size, (2, 0), rowspan=3, colspan=1)
        cut_bad_fits = [abs(mag_cat[goodstarcols]+2.5*np.log10(flux_star[goodstarcols])-zpt) < .50]
        plt.scatter(-2.5*np.log10(flux_star[goodstarcols])[cut_bad_fits]+zpt,mag_cat[goodstarcols][cut_bad_fits]+2.5*np.log10(flux_star[goodstarcols])[cut_bad_fits]-zpt)
        plt.ylabel('Catalog Mag - Fit Mag')
        plt.xlabel('Fit Mag')
        
        #Catalog Mag - Fit Mag vs g-i mag color
        plt.subplot2grid(grid_size, (2, 1), rowspan=3, colspan=1)
        #plt.scatter(mag_cat[goodstarcols]+2.5*np.log10(flux_star[goodstarcols])-zpt,starcat.mag_g[goodstarcols]-starcat.mag_i[goodstarcols])
        plt.ylabel('Catalog Mag - Fit Mag')
        plt.xlabel('Mag g-i')
        plt.tight_layout()
        plt.savefig(filename+'.pdf')
        
        a = open(self.zpt_fits,'a')
        a.write(filename+'.pdf\n')
        a.close()
        #plt.show()
        #raw_input()
        return

    #def big_zpt_plot( self, filename = './zpts/bigzpt.pdf' ):
    #    import rdcol
    #    data = rdcol.read(self.big_zpt,0,1,'\t')
    #    plt.scatter(data['MP Fit Mag']+data['Zpt'],data['Cat Mag']-data['MP Fit Mag']-zpt)
    #    plt.ylabel('Catalog Mag - Fit Mag')
    #    plt.xlabel('Fit Mag')
    #    plt.savefig(self.big_zpt+'.pdf')
        

    def build_psfex(self, psffile,x,y):
        '''
        Inputs from dump_psfex output file:

        PSF: Xcoord, Ycoord, dx, dy, psfval 
        
        Returns psfout such that psfout[Ycoord+5, Xcoord+5] = psfval

        e.g. if a line reads "PSF: 17 18 1.266 0.341 1.649823e-02"

        print psfout[23, 22] will return .01649823
        

        PSF_CENTER: 17 17        # PSF coords
        PSF_MAX:    17 18        # PSF coords 
        IMAGE_CENTER: 554 3914   # IMAGE coords (PSF_CENTER here)
        IMAGE_CORNER: 1 1      # pixel index at corner 
        
        Center of PSF in image coordinates is 553 3913
        This corresponds to psfout[22,22]
        
        '''

        ### psf = os.popen("dump_psfex -inFile_psf %s -xpix %s -ypix %s -gridSize %s"%(psffile,x,y,
        ###                                                                           self.params.substamp)).read()
        psf = os.popen("dump_psfex -inFile_psf %s -xpix %s -ypix %s -gridSize %s"%(psffile,x,y,
                                                                                   35)).readlines()
        #ix, iy, psfval = np.genfromtxt(psffile, usecols = (1,2,5), skip_footer = 4)
        readdata,readheader = False,True
        ix,iy,psfval = [],[],[]
        IMAGE_CORNERX = 0
        IMAGE_CORNERY = 0
        for line in psf:
            line = line.replace('\n','')
            if line.startswith('PSF:'):
                #linelist = filter(None,line.split(' '))
                linelist = line.split()
                ix += [int(linelist[1])]; iy += [int(linelist[2])]; psfval += [float(linelist[5])]
            elif line.startswith("IMAGE_CENTER"):
                linelist = line.split()
                IMAGE_CENTERX = float(linelist[1]); IMAGE_CENTERY = float(linelist[2])
            #elif line.startswith("IMAGE_CORNER"):
            #    linelist = line.split()
            #    IMAGE_CORNERX = float(linelist[1]); IMAGE_CORNERY = float(linelist[2])

        #IMAGE_CENTERX -= IMAGE_CORNERX; IMAGE_CENTERY -= IMAGE_CORNERY
        ix,iy,psfval = np.array(ix),np.array(iy),np.array(psfval)
        psfout = np.zeros((2*self.params.fitrad + 1,2*self.params.fitrad + 1))
        for x,y,p in zip(ix,iy,psfval):
            if x >= (35 - 2*self.params.fitrad -1)/2 and y >= (35 - 2*self.params.fitrad -1)/2 and x < (2*self.params.fitrad +1) and y < (2*self.params.fitrad + 1):
                psfout[y-(35 - 2*self.params.fitrad - 1)/2,x-(35 - 2*self.params.fitrad -1)/2] = p 
            #psfout[y,x] = p

        return(psfout), (IMAGE_CENTERX, IMAGE_CENTERY)

def scene_check(p,x=None,y=None,fjac=None,params=None,err=None):
    """Scene modeling function, but with error 
    measurements and optionally saves stamps"""
    status = 0

    Nimage = len(x[:,0,0])
    substamp = float(params.substamp)
        
    model = np.zeros([Nimage,substamp,substamp])
    galaxy = p[0:substamp**2.].reshape(substamp,substamp)

    chi2 = np.zeros(Nimage)
    for i in range(len(Nimage)):
        conv_prod = scipy.ndimage.convolve(sp,x[i,:,:])
        # model = scale + convolution + sky
        model[i,:,:] = p[substamp**2.+1]*x[i,:,:] + conv_prod + p[substamp**2+Nimage+i]

        xx = np.where(err < 10000.0)
        chi2[i]=np.sqrt(np.sum((model[xx]-y[xx])**2/err[xx]**2.)/float(len(xx)))
        
    return(chi2)


def scene(p,x=None,y=None,fjac=None,params=None,err=None):
    """Scene modeling function given to mpfit"""
    status = 0

    Nimage = len(x[:,0,0])
    substamp = float(params.substamp)
        
    model = np.zeros([Nimage,substamp,substamp])
    galaxy = p[0:substamp**2.].reshape(substamp,substamp)

    for i in range(Nimage):
        conv_prod = scipy.ndimage.convolve(galaxy,x[i,:,:])
        # model = scale + convolution + sky
        model[i,:,:] = p[substamp**2.+i]*x[i,:,:] + conv_prod + p[substamp**2+Nimage+i]

    return(status, (y.reshape(Nimage*substamp*substamp)-model.reshape(Nimage*substamp*substamp))/err.reshape(Nimage*substamp*substamp))

if __name__ == "__main__":

    import sys,getopt
    
     # read in arguments and options
    try:
        print sys.argv[1:]
        opt,arg = getopt.getopt(
            sys.argv[1:],"hs:p:r:f:v",
            longopts=["help","snfile","params","rootdir",
                      "filter","nomask","nodiff","nozpt",
                      "debug","verbose"])
    except getopt.GetoptError as err:
        print str(err)
        print "Error : incorrect option or missing argument."
        print __doc__
        sys.exit(1)

    verbose,nodiff,debug,clear_zpt = False,False,False,False

    snfile,param_file,root_dir,filt = '','','',''
    nomask,getzpt = 'none',False
    for o,a in opt:
        if o in ["-h","--help"]:
            print __doc__
            sys.exit(0)
        elif o in ["-s","--snfile"]:
            snfile = a
        elif o in ["-p","--params"]:
            param_file = a
        elif o in ["-r","--rootdir"]:
            root_dir = a
        elif o in ["-f","--filter"]:
            filt = a
        elif o in ["-v","--verbose"]:
            verbose = True
        elif o == "--nomask":
            nomask = True
        elif o == "--nodiff":
            nodiff = True
        elif o == "--getzpt":
            getzpt = True
        elif o == "--debug":
            debug = True
        elif 0 == "--clear_zpt"
            clear_zpt = True

    if not snfile or not param_file:
        print("Error : snfile and params  must be provided")
        print(__doc__)
        sys.exit(1)
    snparams = get_snfile(snfile, root_dir)
    params = get_params(param_file)

    if nomask == 'none':
        if params.mask_type.lower() == 'none':
            nomask = True
        else: nomask = False
    if getzpt == 'none':
        if params.find_zpt.lower() == 'yes':
            getzpt = True
        else: getzpt = False

    if not filt:
        print("Filt not defined.  Using all...")
        filt = snparams.filters
    if not root_dir:
        root_dir = snparams.image_dir


    scenemodel = smp(snparams,params)
    scenemodel.main(nodiff=nodiff,getzpt=getzpt,nomask=nomask,debug=debug,verbose=verbose,clear_zpt=clear_zpt)
