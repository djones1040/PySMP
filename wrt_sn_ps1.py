#!/usr/bin/env python
# D. Jones - 2/4/14
"""Write scene modeling supernova file for a PS1 supernova

wrt_sn_ps1.py -i <id> -f <filters>

"""

import os
import glob
import pyfits
from txtobj import txtobj
import numpy as np

def main(id,outfile='',filters='griz'):

    import pipeclasses
    params = pipeclasses.paramfileclass()
    params.loadfile(os.environ['PIPE_PARAMS'])
    params.loadfile(os.environ['EXTRAPARAMFILE'],addflag=True)

    if not outfile:
        fout = open('%s.supernova'%id,'w')
    else:
        fout = open(outfile,'w')

    nfiles = 0
    nxpix = 0
    for f in filters:
        impath = '%s/%s/%s'%(params.get('WORKSPACE_DIR'),id,f)
        files = glob.glob('%s/*%s'%(impath,params.get('SWARPSUFFIX')))
        nfiles += len(files)
        if not nxpix:
            nxpix = pyfits.getval(files[0],'NAXIS1')
            nypix = pyfits.getval(files[0],'NAXIS2')

    objlist = txtobj(params.get('C2E_EVENTLIST'))
    col = np.where(objlist.PS1id == id)
    ra,dec,peakmjd = objlist.ra[col][0],\
        objlist.dec[col][0],objlist.mjd_disc[col][0]

    header = """SURVEY:    PS1
SNID:      %s
FILTERS:   griz
PIXSIZE:   0.25    # arcsec
NXPIX:     %i
NYPIX:     %i
ZPFLUX:    30.0     # PS1 FLUX zero point
PSF_FORMAT: DAOPHOT
PEAKMJD:  %.1f
WEIGHT_BADPIXEL:  3  99999
RA: %.7f
DECL: %.7f

IMAGE_DIR: 
PSF_UNIT: pixels
PSF_SIZEPARAM: sigma
PSF_MODEL: daophot
PLATESCALE: %s

NVAR: 12 
NOBS: %i
VARNAMES: ID_OBS ID_COADD MJD BAND GAIN FLUX FLUXERR IMAGE_NAME_SEARCH IMAGE_NAME_WEIGHT IMAGE_NAME_PSF IMAGE_NAME_MASK CATALOG_FILE IMAGE_ZPT"""%(id,nxpix,nypix,peakmjd,ra,dec,params.get('SW_PLATESCALE'),nfiles)
    print >> fout, header

    count = 1
    swarpsuffix = params.get('SWARPSUFFIX')
    icmpsuffix = params.get('IDLDAOPHOTCMPSUFFIX')
    
    for f in filters:
        impath = '%s/%s/%s'%(params.get('WORKSPACE_DIR'),id,f)
        files = glob.glob('%s/*%s'%(impath,params.get('IDLDAOPHOTCMPSUFFIX')))
        for fl in files:
            line = 'OBS: %i %i %.1f %s %s -99 -99 '%(count,count,pyfits.getval(fl,'MJD-OBS'),
                                                     f,pyfits.getval(fl,'GAIN'))
            line += '%s %s %s %s %s %s'%(fl.replace('.icmp','.fits'),
                                         fl.replace('.icmp','.noise.fits'),
                                         fl.replace('.icmp','.dao.psf.fits'),
                                         fl.replace('.icmp','.mask.fits'),
                                         fl.replace('.icmp','.absphotcat.wcs'),
                                         pyfits.getval(fl,'ZPTMAG'))
            print >> fout, line

    fout.close()

if __name__ == "__main__":

    import sys,getopt
    
     # read in arguments and options
    try:
        opt,arg = getopt.getopt(
            sys.argv[1:],"hi:o:f:",
            longopts=["help","id=","outfile=","filters="])
    except getopt.GetoptError:
        print "Error : incorrect option or missing argument."
        print __doc__
        sys.exit(1)

    id,outfile,filters = '','',''
    for o,a in opt:
        if o in ["-h","--help"]:
            print __doc__
            sys.exit(0)
        elif o in ["-i","--id"]:
            id = a
        elif o in ["-o","--outfile"]:
            outfile = a
        elif o in ["-f","--filters"]:
            filters = a

    if not id:
        print("Error : id must be provided")
        print(__doc__)
        sys.exit(1)

    main(id,outfile=outfile,filters=filters)
