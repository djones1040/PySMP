#!/usr/bin/env python
# D. Jones - 5/14/14
# diffimmagstats.py --cmpfile=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052/tmpl/g/PSc360052.md04s047.g.stack_44.sw.icmp --psffile=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052/g/PSc360052.md04s047.g.ut091126f.648816_44.sw.dao.psf.fits --diffim=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052_tmpl/g/PSc360052.md04s047.g.ut091126f.648816_44_md04s047.g.stack_44.diff.fits
"""Calculate increase in uncertainty due
to bright host galaxies

Usage: diffimmagstats.py --cmpfile=cmpfile --psffile=psffile --diffim=diffimfile

"""
import glob
import os
import numpy as np
import exceptions
#import pyfits
import astropy.io.fits as pyfits
class txtobj:
    def __init__(self,filename,allstring=False,
                 cmpheader=False,sexheader=False,
                 useloadtxt=True, des=False,
                 delimiter=''):
        if cmpheader: hdr = pyfits.getheader(filename)

        coldefs = np.array([])
        if cmpheader:
            for k,v in zip(hdr.keys(),hdr.values()):
                if 'COLTBL' in k and k != 'NCOLTBL':
                    coldefs = np.append(coldefs,v)
        elif sexheader:
            fin = open(filename,'r')
            lines = fin.readlines()
            for l in lines:
                if l.startswith('#'):
                    coldefs = np.append(coldefs,filter(None,l.split(' '))[2])
        elif des:
            fin = open(filename, 'r')
            lines = fin.readlines()
            for l in lines:
                if l.startswith('# VARNAMES:'):
                    coldefs = np.array(l.split()[2:])
                    coldefs = map(lambda x: x.lower(), coldefs)
                    break
            
        else:
            fin = open(filename,'r')
            lines = fin.readlines()
            coldefs = np.array(filter(None,lines[0].split(' ')))
            coldefs = coldefs[np.where(coldefs != '#')]
        for c in coldefs:
            c = c.replace('\n','')
            if c:
                self.__dict__[c] = np.array([])
        
        self.filename = np.array([])
        if useloadtxt:
            for c,i in zip(coldefs,range(len(coldefs))):
                c = c.replace('\n','')
                if c:
                    if not delimiter:
                        try:
                            self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i])
                        except:
                            self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],dtype='string')
                    else:
                        try:
                            self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],delimiter=',')
                        except:
                            self.__dict__[c] = np.loadtxt(filename,unpack=True,usecols=[i],dtype='string',delimiter=',')

            self.filename = np.array([filename]*len(self.__dict__[coldefs[0]]))

        else:
            fin = open(filename,'r')
            count = 0
            for line in fin:
                if count >= 1 and not line.startswith('#'):
                    entries = filter(None,line.split(' '))
                    for e,c in zip(entries,coldefs):
                        e = e.replace('\n','')
                        c = c.replace('\n','')
                        if not allstring:
                            try:
                                self.__dict__[c] = np.append(self.__dict__[c],float(e))
                            except:
                                self.__dict__[c] = np.append(self.__dict__[c],e)
                        else:
                            self.__dict__[c] = np.append(self.__dict__[c],e)
                        self.filename = np.append(self.filename,filename)
                else: count += 1
            fin.close()

    def addcol(self,col,data):
        self.__dict__[col] = data
    def cut_inrange(self,col,minval,maxval,rows=[]):
        if not len(rows):
            rows = np.where((self.__dict__[col] > minval) &
                            (self.__dict__[col] < maxval))[0]
            return(rows)
        else:
            rows2 = np.where((self.__dict__[col][rows] > minval) &
                            (self.__dict__[col][rows] < maxval))[0]
            return(rows[rows2])
    def appendfile(self,filename,useloadtxt=False):
        if useloadtxt:
            fin = open(filename,'r')
            for line in fin:
                if line.startswith('#'):
                    coldefs = filter(None,line.split('#')[1].split('\n')[0].split(' '))
                    break
            fin.close()
            for c,i in zip(coldefs,range(len(coldefs))):
                try:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.loadtxt(filename,unpack=True,usecols=[i])))
                except:
                    self.__dict__[c] = np.concatenate((self.__dict__[c],np.loadtxt(filename,unpack=True,
                                                                                   usecols=[i],dtype='string')))
            self.filename = np.append(self.filename,np.array([filename]*len(np.loadtxt(filename,unpack=True,usecols=[i],dtype='string'))))
            
            return()
        fin = open(filename,'r')
        for line in fin:
            if line.startswith('#'):
                coldefs = filter(None,line.split('#')[1].split('\n')[0].split(' '))
            else:
                entries = filter(None,line.split(' '))
                for e,c in zip(entries,coldefs):
                    e = e.replace('\n','')
                    c = c.replace('\n','')
                    try:
                        self.__dict__[c] = np.append(self.__dict__[c],float(e))
                    except:
                        self.__dict__[c] = np.append(self.__dict__[c],e)
                self.filename = np.append(self.filename,filename)
