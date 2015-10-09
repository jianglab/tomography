#!/usr/bin/env python

#
# Author: Rui Yan <yan49@purdue.edu>, Sep 2015
# Copyright (c) 2012 Purdue University
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
#
from EMAN2 import *
import os, sys, math, itertools
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import collections
from itertools import chain
from scipy import stats
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

def main():
	progname = os.path.basename(sys.argv[0])
	usage = """
        Determine the thickness, sample tilt and mean free path of tomographic tilt series
	
	Example:

	python tomoThickness.py --tiltseries 6hSINVc1s2_17.ali --tiltangles 6hSINVc1s2_17.tlt --boxsize 200 --gain 8 --MFP 200  --B 1600 --d0 200 --theta0 5 --alpha0 0 --I0 40000 --niter 200 --interval 50  --x0 1200,1400,1000,2400,2900,2600,1400,800 --y0 1100,1400,2000,3600,2900,600,2800,2400 
        
	python tomoThickness.py --tiltseries virusJAN15000.ali --tiltangles virusJAN15000.tlt  --boxsize 100 --MFP 350 --gain 10 --B 480 --I0 2820 --d0 100 --alpha0 10 --theta0 0 --niter 200  --x0 1100,1200,1150,1050 --y0 200,300,250,350
	"""
                
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	parser.add_argument("--tiltseries", type=str, default='', help="tilt series with tilt axis along Y")
        parser.add_argument('--tiltangles',type=str,default='',help='File in .tlt format containing the tilt angle of each image in the tiltseries.')
        parser.add_argument("--boxsize", type=int, default=200, help="perform grid boxing using given box size. default to 200")
        parser.add_argument("--x0", type=str, default=0, help="for test on some regions, multiple regions are allowed, --x0 100,200,300")
        parser.add_argument("--y0", type=str, default=0, help="for test on some regions, multiple regions are allowed, --y0 100,200,300")
        parser.add_argument("--adaptiveBox", action="store_true", default=False, help="squeeze the x side of boxsize by cos(theta(tlt))")
        parser.add_argument("--writeClippedRegions", action="store_true", default=False, help="write out the clipped region of interest, test only")
        
	parser.add_argument("--I0", type=float, default=2000, help="whole spectrum I0")
        parser.add_argument("--d0", type=float, default=100, help="initial thickness")
        parser.add_argument("--theta0", type=float, default=0, help="offset of angle theta (the initial offset angle around y-axis)")
	parser.add_argument("--alpha0", type=float, default=0, help="offset of angle alpha (the initial offset angle around x-axis)")
	parser.add_argument("--gain", type=float, default=20, help="# of electrons = gain * pixel_value")
        parser.add_argument("--B", type=float, default=0, help="# of electrons = gain * pixel_value + B")
        parser.add_argument("--MFP", type=float, default=350, help="mean free path, for vitreous ice, 350nm@300kV, 300nm@200kV")
        parser.add_argument("--k", type=float, default=0, help="I0(theta) = I0/(cos(theta)**k), and 0=<k<=1")	
	
        parser.add_argument("--addOffset", type=str, default='-32000', help="Add options.addOffset to pixel values")
        #parser.add_argument("--inversePixel", action="store_true", default=False, help="inverse pixel values")
	
	parser.add_argument("--plotData", action="store_true", default=False, help="plot the original data, including curvilinear mode and linear mode")
	parser.add_argument("--plotResults", action="store_true", default=False, help="plot the original data and fitted results, including curvilinear mode and linear mode")

        parser.add_argument("--mode", type=int, default=0, help="")
        parser.add_argument("--niter", type=int, default=200, help="niter in basinhopping")
        parser.add_argument("--interval", type=int, default=50, help="interval in basinhopping")
        parser.add_argument("--T", type=float, default=1e-4, help="T in basinhopping")
	
	parser.add_argument("--modifyTiltFile", action="store_true", default=False, help="modify the .tlt file by returned theta0")
	parser.add_argument('--modifiedTiltName',type=str,default='',help='the filename of modified tilt angles')
	#parser.add_argument("--refineRegion", action="store_true", default=False, help="use returned theta0 to re-clip region and re-do optimization")
        
        parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level, higner number means higher level of verboseness")
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)
        
        global options
	(options, args) = parser.parse_args()
        logger = E2init(sys.argv, options.ppid)
	
	serieshdr = EMData(options.tiltseries,0,True)
        
	global nslices
	nslices = serieshdr['nz']
	nx = serieshdr['nx']
	ny = serieshdr['ny']
	print "tiltseries %s: %d*%d*%d"%(options.tiltseries, nx, ny, nslices)
        
	#read in tilt angles file, *.tlt
        anglesfile = open(options.tiltangles,'r') #Open tilt angles file
	alines = anglesfile.readlines()		  #Read its lines
	anglesfile.close()			  #Close the file
	
        #global tiltangles
	tiltangles = [ alines[i].replace('\n','') for i in range(len(alines)) ]	#Eliminate trailing return character, '\n', for each line in the tiltangles file
	ntiltangles = len(tiltangles)
        tiltanglesArray = np.array(tiltangles)
        if (options.verbose>=10): print tiltangles
	
	blocks = []
        boxsize = options.boxsize
	
	if (options.x0 and options.y0):
		x0 = [int(x) for x in options.x0.split(',')]
		y0 = [int(y) for y in options.y0.split(',')]
	else:
		print "Please provide the X/Y coordinates of selected regions using --x0 --y0\n"
		sys.exit(0)
	
	origDictionary = collections.OrderedDict()
        for k in range(nslices):
		angle = float(tiltangles[k])
		r0 = Region(0, 0, k, nx, ny, 1)
		tiltedImg = EMData(options.tiltseries, 0, 0, r0)	
		blockMeanList = []
            
		for i in range(len(x0)):
			testname = options.tiltseries.split('.')[0]+'_x0%g_y0%g_clip.hdf'%(x0[i], y0[i])
			
			xp = (x0[i] - nx/2.0) * math.cos(math.radians(angle)) + nx/2.0
			yp = y0[i]		
			
			if (options.adaptiveBox):
			    boxsizeX = int(boxsize * math.cos(math.radians(angle)))
			else:
			    boxsizeX = boxsize
						
			#extract the whole image at each tilt
			xp = xp-boxsizeX/2
			yp = yp-boxsize/2
			r = Region(xp, yp, boxsizeX, boxsize) 
			img = tiltedImg.get_clip(r)
					
			if (options.writeClippedRegions): img.write_image(testname, k)
					
			blockMeanValues = blockMean(img, boxsizeX, boxsize)
			blockMeanList.append(blockMeanValues)
		origDictionary[tiltangles[k]] = flattenList(blockMeanList)
		
	#if (options.verbose>=10): print origDictionary
	assert(len(origDictionary)==len(tiltangles))
	startZ = 0
        endZ = nslices
	stepZ = 1 
	
	dictionary0 = collections.OrderedDict()
	n=0
	for key, value in origDictionary.items()[startZ:endZ]:
		if (math.fmod(n, stepZ) == 0): dictionary0[key] = value	
		n+=1
	#print "len(dictionary)=", len(dictionary0)

        #check if the tilt angles are from negative to positive, if not, reverse the order of dictionary
	if (float(tiltangles[0]) > 0):
	    print "Reversing the order of tilt angles since we usually start from negative tilts to positive tilts"
	    items = dictionary0.items()
	    items.reverse()
	    dictionary0 = collections.OrderedDict(items)
            
        if (options.verbose>=10): print dictionary0
        if (options.plotData): plotOriginalData(dictionary0, options)
       
        global dictionary
        #dictionary = averageRescaledResultDict(rescaledResultDict, options)
        dictionary = dictionary0
        
        global maxVal, minVal
        maxKey, maxVal = max(dictionary.iteritems(), key=lambda x:x[1])
        maxVal = maxVal[0]
        minKey, minVal = min(dictionary.iteritems(), key=lambda x:x[1])
        minVal = minVal[0]
        print "max: max average pixel value = %g @ tilt angles =%s"%(maxVal, maxKey)
        print "min: min average pixel value = %g @ tilt angles =%s"%(minVal, minKey)
	
	if (options.mode == 0): #use complete model, use multiple regions
            print "Using complete model and %g boxes!"%len(x0)
            I0 = options.I0
	    d0 = options.d0
	    theta0 = options.theta0
	    alpha0 = options.alpha0
	    A = options.gain
	    B = options.B
	    MFP = options.MFP
	    niter = options.niter
	    interval = options.interval
	    p0 = [I0, d0, theta0, alpha0, A, B, MFP]
	    x0 = p0
            boundsList = [(maxVal, None),(10, 250), (-10, 10), (-10, 10), (0.01, None), (None, int(minVal)), (1, None)]
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=boundsList)
            mybounds = MyBounds()
            mybounds.xmax=[float('inf'), 250.0, 10.0, 10.0, float('inf'), int(minVal), float('inf')]
	    mybounds.xmin=[maxVal, 10.0, -10.0, -10.0, 0.01, (-1)*(float('inf')), 1.0]
	    mytakestep = MyTakeStep3()
	    res = scipy.optimize.basinhopping(optimizationFuncFullModel0, x0, T=options.T, stepsize=0.01, minimizer_kwargs=minimizer_kwargs, niter=niter, take_step=mytakestep, accept_test=mybounds, \
					   callback=None, interval=interval, disp=False, niter_success=None)
	    #print res
	    tmp = res.x.tolist()
            #tmp[1] = tmp[1]+100
            I0, d0, theta0, alpha0, A, B, MFP = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]
            gamma0 = calculateGamma0(theta0, alpha0)
	    #print "[I0, d0, theta0, alpha0, A, B, MFP, gamma0] =", I0, d0, theta0, alpha0, A, B, MFP, gamma0
	    print "***************************************************"
	    print "Tilt series: %s"%options.tiltseries
	    print "Fitting results:"
	    print "Thickness = %g nm"%d0
	    print "Sample tilt: theta0 = %g degree, alpha0 = %g degree, gamma0 = %g degree"%(theta0, alpha0, gamma0)
	    print "Mean free path = %g nm"%MFP
            if (options.plotResults):
		compareFitData(dictionary, tmp, options)
	    
	if (options.modifyTiltFile):
		if (options.modifiedTiltName):
			tiltFile = options.modifiedTiltName
		else:
			tiltFile = options.tiltseries.split(".")[0] + "_modifiedFullModel.tlt"
		fp = open(tiltFile, 'w')
		for i in tiltangles:
			tlt = float(i) + theta0
			#print float(tlt)
			line = "%g\n"%(tlt)
			fp.write(line)
		fp.close()


def plotOriginalData(dictionary, options):
	#plot the curve mode and log-ratio mode of original data
	thetaLst = []
	xlinearLst=[]
        intensityLst = []

        for theta, intensity in dictionary.iteritems():
                thetaLst.append(float(theta))
                intensityLst.append(intensity)
		cosAngle = math.cos((float(theta)/360.)*math.pi*2)
		x = (1./(cosAngle))
		xlinearLst.append(x)
                
        xdata = np.asarray(thetaLst)
        ydata = np.asarray(intensityLst)
        ydataInv = ydata[::-1]
	#print xdata, ydata
        
        x0 = [int(x) for x in options.x0.split(',')]
        y0 = [int(x) for x in options.y0.split(',')]
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']
        markers = ['s', 'o', '^', 'v', 'x', '*', '+', 'd', 'D', '<', '>', 'p', '8', 'H']
        
        plt.figure(figsize=(12.5, 10))
        #plt.subplot(221)
        for i in range(len(x0)):
            boxPosition = '%g,%g'%(x0[i], y0[i])
            if (i<len(colors)):
                plt.plot(xdata, ydata[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
            else:
                i = i-len(colors)
                plt.plot(xdata, ydata[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
        
        plt.axvline(0, linestyle='--', color='k', linewidth=2.0)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)
            
        plt.xlabel(r'$\theta$ ($^\circ$)', fontsize = 24, labelpad = 10)
        plt.ylabel('Intensity', fontsize = 24, labelpad = 10)
        #plt.xlim(-70, 70)
        plt.grid(True, linestyle = '--', alpha = 0.5)

	#plot the linear format (log-ratio mode) of original data
	xlinear = np.asarray(xlinearLst)
	ylinear = np.log(ydata)
        plt.figure(figsize=(12.5, 10))
        #plt.subplot(222)
        for i in range(len(x0)):
            boxPosition = '%g,%g'%(x0[i], y0[i])
            if (i<len(colors)):
                plt.plot(xlinear, ylinear[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
            else:
                i = i-len(colors)
                plt.plot(xlinear, ylinear[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
                
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)
        plt.xlabel(r'1/cos($\theta$)', fontsize = 24, labelpad = 10)
        plt.ylabel('ln(Intensity)', fontsize = 24, labelpad = 10)
        plt.grid(True, linestyle = '--', alpha = 0.5)

        plt.show()

def compareFitData(dictionary, tmp, options):
	thetaLst = []
	xlinearLst=[]
        intensityLst = []

        for theta, intensity in dictionary.iteritems():
                thetaLst.append(float(theta))
                intensityLst.append(intensity)
		cosAngle = math.cos((float(theta)/360.)*math.pi*2)
		x = (1./(cosAngle))
		xlinearLst.append(x)
                
        xdata = np.asarray(thetaLst)
        ydata = np.asarray(intensityLst)
        ydataInv = ydata[::-1]
	#print xdata, ydata
        
        x0 = [int(x) for x in options.x0.split(',')]
        y0 = [int(x) for x in options.y0.split(',')]
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']
        markers = ['s', 'o', '^', 'v', 'x', '*', '+', 'd', 'D', '<', '>', 'p', '8', 'H']
        
        plt.figure(figsize=(25, 20))
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
	
	#plot the curvilinear format of original data
        plt.subplot(221)
        for i in range(len(x0)):
            boxPosition = '%g,%g'%(x0[i], y0[i])
            if (i<len(colors)):
                plt.plot(xdata, ydata[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
            else:
                i = i-len(colors)
                plt.plot(xdata, ydata[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
        
        plt.axvline(0, linestyle='--', color='k', linewidth=2.0)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)
            
        plt.xlabel(r'$\theta$ ($^\circ$)', fontsize = 24, labelpad = 10)
        plt.ylabel('Intensity', fontsize = 24, labelpad = 10)
	plt.title('Original: %s'%options.tiltseries)
        #plt.xlim(-70, 70)
        plt.grid(True, linestyle = '--', alpha = 0.5)

	#plot the linear format (log-ratio mode) of original data
	xlinear = np.asarray(xlinearLst)
	ylinear = np.log(ydata)
        #plt.figure(figsize=(12.5, 10))
        plt.subplot(222)
        for i in range(len(x0)):
            boxPosition = '%g,%g'%(x0[i], y0[i])
            if (i<len(colors)):
                plt.plot(xlinear, ylinear[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
            else:
                i = i-len(colors)
                plt.plot(xlinear, ylinear[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
                
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)
        plt.xlabel(r'1/cos($\theta$)', fontsize = 24, labelpad = 10)
        plt.ylabel('ln(Intensity)', fontsize = 24, labelpad = 10)
	plt.title('Original: %s'%options.tiltseries)
        plt.grid(True, linestyle = '--', alpha = 0.5)
	

        I0, d0, theta0, alpha0, A, B, MFP = tmp
        x0 = [int(x) for x in options.x0.split(',')]
        y0 = [int(x) for x in options.y0.split(',')]
        
	xfit = []
	yfit = []
	xdata = []
        xModified=[]
        ydata = []
        ydataLinear = []
        I0Lst = []
        
	for theta, intensity in dictionary.iteritems():
		for i in range(len(intensity)):
			theta_i = float(theta) + theta0
                        xModified.append(theta_i)
			#angle.append(theta_i)
			cosAngle = math.cos((float(theta)/360.)*math.pi*2)
			cosTheta = math.cos((theta_i/360.)*math.pi*2)
			cosAlpha = math.cos((alpha0/360.)*math.pi*2)
                        intensityIn = math.log(I0)
                        
			y = intensityIn - (1./(MFP * cosTheta * cosAlpha)) * d0
			yfit.append(y)
			#print intensity
                        ########which one is used as ydata in corrected plots
                        y2 = math.log(intensity[i])
			#y2 = math.log(A * (intensity[i] - B))
			ydataLinear.append(y2)
                        ydata.append(intensity[i])
			
			#x = (-1) * (1./(MFP * cosTheta * cosAlpha))
			x = (1./(cosTheta))
			xfit.append(x)
			#x2 = (-1) * (1./(MFP * cosAngle))
			x2 = (1./(cosAngle))
			xdata.append(x2)
                        
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']
        markers = ['s', 'o', '^', 'v', 'x', '*', '+', 'd', 'D', '<', '>', 'p', '8', 'H']
        
	#plot the linear format (log-ratio mode) of fitted data after determination of parameters
        xfit = np.asarray(xfit)
        xfit2 = np.reshape(xfit, (nslices, len(x0)))
        yfit = np.asarray(yfit)
        yfit2 = np.reshape(yfit, (nslices, len(x0)))
        xdata = np.asarray(xdata)
        xdata2 = np.reshape(xdata, (nslices, len(x0)))
        ydataLinear = np.asarray(ydataLinear)
        ydataLinear2 = np.reshape(ydataLinear, (nslices, len(x0)))
        residuals = ydataLinear - yfit
        fres = sum(residuals**2)
        text_str = 'I0=%g\nd0=%g\ntheta0=%g\nalpha0=%g\ngain=%g\nB=%g\nMFP=%g\nres=%g'%(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], fres)
	
	plt.subplot(224)
        #plt.figure(figsize=(12.5, 10))
        for i in range(len(x0)):
                boxPosition = '%g,%g'%(x0[i], y0[i])
                plt.plot(xfit2[:, i], ydataLinear2[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
               
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)
                
	#plt.plot(xfit, ydata, 'g^')
        plt.title('Least-squares fitting: %s'%options.tiltseries)
        plt.xlabel(r'1/cos($\theta$+$\theta_0$)', fontsize = 24, labelpad = 10)
        plt.ylabel('ln(Intensity)', fontsize = 24, labelpad = 10)
        plt.grid(True, linestyle = '--', alpha = 0.5)
        #plt.show()
                
	#plot the curvilinear format of fitted data after determination of parameters
	#xdata, xModified, ydata, yfit  = fitDataCurve(dictionary, tmp)
        xdata = np.asarray(xdata)
        xdata2 = np.reshape(xdata, (nslices, len(x0)))
        xModified = np.asarray(xModified)
        xModified2 = np.reshape(xModified, (nslices, len(x0)))
        ydata = np.asarray(ydata)
        ydata2 = np.reshape(ydata, (nslices, len(x0)))
        ydata2Inv = ydata2[::-1]
	yfit = np.asarray(yfit)
        yfit2 = np.reshape(yfit, (nslices, len(x0)))
		
	residuals = ydata - yfit
	fres = sum(residuals**2)
	text_str = 'I0=%g\nd0=%g\ntheta0=%g\nalpha0=%g\ngain=%g\nB=%g\nMFP=%g\nres=%g'%(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], fres)
                
        #plt.plot(xModified2, yfit2, 'r--', linewidth=2.0)
        #plt.figure(figsize=(12.5, 10))
	plt.subplot(223)
        for i in range(len(x0)):
                boxPosition = '%g,%g'%(x0[i], y0[i])
                plt.plot(xModified2[:, i], ydata2[:, i], markers[i], label = boxPosition, markersize=5, color = colors[i])
                
        plt.axvline(0, linestyle='--', color='k', linewidth=2.0)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(fontsize = 18)
        ax = plt.gca()
	ax.tick_params(pad = 10)

        plt.title('Least-squares fitting: %s'%options.tiltseries)
        plt.xlabel(r'$\theta$+$\theta_0$ ($^\circ$)', fontsize = 24, labelpad = 10)
        plt.ylabel('Intensity', fontsize = 24, labelpad = 10)
        #plt.xlim(-70, 70)
        plt.grid(True, linestyle = '--', alpha = 0.5)
	
	pdfName = options.tiltseries.split('.')[0]+'_results.pdf'
	print pdfName
	with PdfPages(pdfName) as pdf:
		pdf.savefig()
		plt.close()
        
	#plt.show()


def calculateGamma0(theta0, alpha0):
        
        cosTheta0 = math.cos((theta0/360.)*math.pi*2)
        cosAlpha0 = math.cos((alpha0/360.)*math.pi*2)
        tanTheta0 = math.tan((theta0/360.)*math.pi*2)
        tanAlpha0 = math.tan((alpha0/360.)*math.pi*2)
        #tmp = 1./(cosTheta0 * cosTheta0 * cosAlpha0 * cosAlpha0) - tanTheta0 * tanTheta0 * tanAlpha0 * tanAlpha0
        
        tmp = tanTheta0 * tanTheta0 + tanAlpha0 * tanAlpha0 + 1
        cosGamma0 = math.pow(tmp, -0.5)
        gamma0 = math.acos(cosGamma0)*360./(math.pi*2)

        return gamma0

def optimizationFuncFullModel0(x): # use complete model
    I0, d0, theta0, alpha0, A, B, MFP = x
    
    cosTheta0 = math.cos((theta0/360.)*math.pi*2)
    cosAlpha0 = math.cos((alpha0/360.)*math.pi*2)
    tanTheta0 = math.tan((theta0/360.)*math.pi*2)
    tanAlpha0 = math.tan((alpha0/360.)*math.pi*2)
    #tmp = 1./(cosTheta0 * cosTheta0 * cosAlpha0 * cosAlpha0) - tanTheta0 * tanTheta0 * tanAlpha0 * tanAlpha0
    
    tmp = tanTheta0 * tanTheta0 + tanAlpha0 * tanAlpha0 + 1
    cosGamma0 = math.pow(tmp, -0.5)
    
    func = 0
    n = 0

    for theta, intensity in dictionary.iteritems():
        for i in range(len(intensity)):
            
            A = math.fabs(A)
            I0 = math.fabs(I0)
            intensityExit = math.log(A * (intensity[i] - B)) 
            intensityIn = math.log(I0)

            theta_i = float(theta) + theta0
            cosTheta = math.cos((theta_i/360.)*math.pi*2)
            #cosAlpha = math.cos((alpha0/360.)*math.pi*2)
            #err =  intensityIn  - (1./(MFP * cosTheta * cosAlpha)) * d0 - intensityExit
	    err =  intensityIn  - (1./(MFP * cosTheta * cosGamma0)) * d0 * cosTheta0 - intensityExit
            func += err * err
            n+=1
    
    func = func/n  
  
    return func

class MyBounds(object):
    def __init__(self, xmax = [], xmin = []):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        
        return tmax and tmin 

class MyTakeStep3(object):
    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize
        #p0 = [I0, d0, theta0, alpha0, A, B, MFP]
        x = np.float64(x)
        x[0] += np.random.uniform(-1000.*s, 1000.*s)
        x[1] += np.random.uniform(-10.*s, 10.*s)
        x[2] += np.random.uniform(-s, s)
        x[3] += np.random.uniform(-s, s)
        x[4] += np.random.uniform(-10.*s, 10.*s)
        x[5] += np.random.uniform(-100.*s, 100.*s)
        x[6] += np.random.uniform(-10.*s, 10.*s)

        return x	


def flattenList(nestedLst):
	flattenLst = list(chain.from_iterable(nestedLst))
	return flattenLst        

def blockMean(img, boxsizeX, boxsize):
        nx, ny = img.get_xsize(), img.get_ysize()
        nxBlock = int(nx/boxsizeX)
        nyBlock = int(ny/boxsize)
	#print nxBlock, nyBlock
    
        blockMeanValues = []
        for i in range(nxBlock):
            x0 = i*boxsizeX
            for j in range(nyBlock):
                y0 = j*boxsize
                r = Region(x0, y0, boxsizeX, boxsize)
                blkImg = img.get_clip(r)
                blockMeanValue = oneBlockMean(blkImg)
                blockMeanValues.append(blockMeanValue)
                
        return blockMeanValues       

def oneBlockMean(img):
	nx, ny = img.get_xsize(), img.get_ysize()
    
	ary=EMNumPy.em2numpy(img)
	ary = reject_outliers(ary, m = 3)

	blkMean = np.mean(ary)
	blkSigma = np.std(ary)

	if (blkMean < 0):
		blkMean = blkMean * (-1)   #average pixel values must be positive
                
        if (blkMean > 30000):
                offset = float(options.addOffset)
                blkMean = blkMean + offset

	return blkMean
    
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

if __name__ == "__main__":
    main()