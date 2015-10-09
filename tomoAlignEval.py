#!/usr/bin/env python

#
# Author: Rui Yan <yan49@purdue.edu>, Dec 2014
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
import os, sys, math, itertools, csv
import numpy as np
from scipy import *
from scipy.interpolate import splrep, sproot, splev
from scipy.interpolate import UnivariateSpline
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import collections
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import chain
from scipy.stats import norm
import matplotlib.mlab as mlab
from numpy.polynomial import polynomial
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

def main():
	progname = os.path.basename(sys.argv[0])
	usage = """
	A indicator of the alignment of tomography tilt series, it can process *.st, *.preali and *.ali simultaneously and compare them.
	This program calculates the mean of all regions first, and then fit the linear relationship.
	Example:
	python tomoAlignment7.py --alltiltseries 6hSINV-sec4_50_rotIMOD.st,6hSINV-sec4_50_rotIMOD.preali,6hSINV-sec4_50_2.ali --alltiltangles 6hSINV-sec4_50.rawtlt,6hSINV-sec4_50.rawtlt,6hSINV-sec4_50_2.tlt --startZ 0 --endZ 60 --x0 450,650,1050,1250,1650,2250,2650,2850,3050,3450 --y0 1250,450,1650,3450,1450,3050,1050,450,850,450 --boxsize 100 --regionsize 100 --verbose 0 --fitLinear --errorsDistribution --FSCfile FSCeo_bin2_6hSINV-sec4_50.csv 
	
	"""
                
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	#parser.add_argument("--tiltseries", type=str, default='', help="tilt series with tilt axis along Y")
	parser.add_argument("--alltiltseries", type=str, default='', help="All tilt series in *.st, *.preali and *.ali, spearated by comma(,)")
        #parser.add_argument('--tiltangles',type=str,default='',help='File in .tlt format containing the tilt angle of each image in the tiltseries.')
	parser.add_argument('--alltiltangles',type=str,default='',help='File in *.rawtlt, *.rawtlt and *.tlt, corresponding to alltiltseries.')
        #parser.add_argument("--tiltAxisAngle", type=float, default=0, help="Tilt axis angle")
        parser.add_argument("--boxsize", type=int, default=256, help="perform grid boxing using given box size. default to 256")
        parser.add_argument("--x0", type=str, default=0, help="for test on some regions, multiple regions are allowed, --x0 100,200,300")
        parser.add_argument("--y0", type=str, default=0, help="for test on some regions, multiple regions are allowed, --y0 100,200,300")
	parser.add_argument("--adaptiveBox", action="store_true", default=False, help="squeeze the x side of boxsize by cos(theta(tlt))")
	parser.add_argument("--useSigma", action="store_true", default=False, help="use the sigma of pixels in each box to represent intensity, rather than use mean of pixels, test only")
       
        parser.add_argument("--addOffset", action="store_true", default=False, help="add 32768 to pixel values")
        parser.add_argument("--inversePixel", action="store_true", default=False, help="inverse pixel values")
        parser.add_argument("--writeClippedRegions", action="store_true", default=False, help="write out the clipped region of interest")
	
	parser.add_argument("--largestNRes", type=int, default=3, help="mark N points with the largest N residuals in the linear mode")
	parser.add_argument("--smallestNRes", type=int, default=3, help="mark N points with the smallest N residuals in the linear mode")
	
        parser.add_argument("--plotData", action="store_true", default=False, help="plot the original data")
	parser.add_argument("--writeResults", action="store_true", default=False, help="write the results into a txt file")
	parser.add_argument("--saveResErrors", action="store_true", default=False, help="save the data in the errors plot into a txt file")
	parser.add_argument("--saveCurvesData", action="store_true", default=False, help="save the data in the curve plot into a txt file")
	parser.add_argument("--saveAverageLinearData", action="store_true", default=False, help="save the data in the averaged & scaled linear plot into a txt file")
	#parser.add_argument("--outputBoxes", type=str, default=0, help="choose which box(es) you select will be shown in the output image, \
	#		    e.g. if you want to show region x0=100, y0=200, you can use --outputRegions 100,200; if you want to show region x0=100,110, y0=200,220, you can use --outputRegions 100,200,110,220")
        
        parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9], higner number means higher level of verboseness")
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)
        
        global options
	(options, args) = parser.parse_args()
        logger = E2init(sys.argv, options.ppid)
        
        tiltseriesLst = options.alltiltseries.split(',')
	tiltanglesLst = options.alltiltangles.split(',')
	
	allResultsDict = collections.OrderedDict()
	allDatasDict = collections.OrderedDict()
	for t in range(len(tiltseriesLst)):
		options.tiltseries = tiltseriesLst[t]
		options.tiltangles = tiltanglesLst[t]
		print "\n*******************************************"
		print "Current tilt series is", options.tiltseries
		print "Current tilt angles is", options.tiltangles
		
		serieshdr = EMData(options.tiltseries,0,True) #"0" means "load the first image in the file/stack", while "True" means "load the header only".
		nslices = serieshdr['nz']
		nx = serieshdr['nx']
		ny = serieshdr['ny']
		print "tiltseries %s: %d*%d*%d"%(options.tiltseries, nx, ny, nslices)
		
		anglesfile = open(options.tiltangles,'r')	#Open tilt angles file
		alines = anglesfile.readlines()			#Read its lines
		anglesfile.close()				#Close the file
		
		tiltangles = [ alines[i].replace('\n','') for i in range(len(alines)) ]	#Eliminate trailing return character, '\n', for each line in the tiltangles file
		ntiltangles = len(tiltangles)
		if (options.verbose>=10): print tiltangles
		tiltanglesArray = np.array(tiltangles)
			
		blocks = []
		boxsize = options.boxsize
		#regionsize = options.regionsize

		if (options.x0 and options.y0):
			x0 = [int(x) for x in options.x0.split(',')]
			y0 = [int(y) for y in options.y0.split(',')]
		else:
			print "Please provide the X/Y coordinates of selected regions using --x0 --y0\n"
			sys.exit(0)
		
		if (options.verbose>=10):
			print "x0 =", options.x0
			print "y0 =", options.y0
		
		assert(len(x0)==len(y0))
		
		origDictionary = collections.OrderedDict()
		#testname = options.tiltseries.split('.')[0]+'_clip.hdf'
		
		for k in range(nslices):
			angle = float(tiltangles[k])
			r0 = Region(0, 0, k, nx, ny, 1)
			tiltedImg = EMData(options.tiltseries, 0, 0, r0)
			
			blockMeanList = []
			for i in range(len(x0)):
				testname = options.tiltseries.split('.')[0]+'_x0%g_y0%g_clip.hdf'%(x0[i], y0[i])
				xp = (x0[i] - nx/2.0) * math.cos(math.radians(angle)) + nx/2.0
				yp = y0[i]
				
				#shrink boxsize
				if (options.adaptiveBox):
					boxsizeX = int(boxsize * math.cos(math.radians(angle)))
				else:
					boxsizeX = boxsize
					
				xp = xp-boxsizeX/2
				yp = yp-boxsize/2
					
				#extract the whole image at each tilt
				r = Region(xp, yp, boxsizeX, boxsize)  
				img = tiltedImg.get_clip(r)
				
				if (options.writeClippedRegions):
					#print "Writing clipped patches to %s"%testname
					img.write_image(testname, k)
				
				blockMeanValues = blockMean(img, boxsizeX, boxsize)
				blockMeanList.append(blockMeanValues)
			
			origDictionary[tiltangles[k]] = flattenList(blockMeanList)
			
		assert(len(origDictionary)==len(tiltangles))
		startZ = 0
		endZ = nslices
		stepZ = 1
		skipZ = []

		global dictionary
		dictionary = collections.OrderedDict()
		n=0
		for key, value in origDictionary.items()[startZ:endZ]:
			if (math.fmod(n, stepZ) == 0 and n not in skipZ):
				dictionary[key] = value	
			n+=1
				
		#check if the tilt angles are from negative to positive, if not, reverse the order of dictionary
		if (float(tiltangles[0]) > 0):
			print "Reversing the order of tilt angles since we usually start from negative tilts to positive tilts"
			items = dictionary.items()
			items.reverse()
			dictionary = collections.OrderedDict(items)
			
		if (options.verbose>=10): print dictionary
		if (options.plotData): plotOriginalData(dictionary, options)
		
		allDatasDict[options.tiltseries] = dictionary	
		thetaCurve, IntensityCurve, thetaLinear, IntensityLinear = generateData2(dictionary, options)
		oneResultDict = fitLinearRegression3(thetaLinear, IntensityLinear, tiltanglesArray, thetaCurve, IntensityCurve, options)
		allResultsDict[options.tiltseries] = oneResultDict
		
	generateStatResults7(allResultsDict, allDatasDict, options)
		
def generateStatResults7(allResultsDict, allDatasDict, options): #calculate mean and std first, and then convert to percentage
	for tiltseries, oneTiltSeriesDict in allResultsDict.iteritems():
		for boxPosition, valueDict in oneTiltSeriesDict.iteritems():
			xLeft0 = valueDict['xLeft']
			xRight0 = valueDict['xRight']
			xdata = valueDict['tiltAngles']
			break
	
	plt.figure()
	plt.figure(figsize=(24, 18))
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
	
	for tiltseries, oneTiltSeriesDict in allResultsDict.iteritems():
		c = 0
		nBoxes = len(oneTiltSeriesDict)
		nTilt = len(xdata)
		curveArray = np.resize(np.array([]), (nTilt, nBoxes + 1))
		i = 1
		for key, valDict in oneTiltSeriesDict.iteritems():
			xdata = valDict['tiltAngles']
			curveArray[:, 0] = xdata
			ydata = valDict['intensityCurve']
			curveArray[:, i] = ydata
			i += 1
		
		#print curveArray
		if (options.saveCurvesData):
			txtFile = tiltseries.replace('.', '_') + '_curve.txt'
			print "Save curves data to %s"%txtFile
			saveCurvesData(txtFile, curveArray)
			
	#print "\n********NEW********"
	subplotLoc = [231, 232, 233, 234, 235, 236]
	subplotNum = -1
	#errorsDict1 = collections.OrderedDict()
	errorsDict2 = collections.OrderedDict()
	labels = ['raw stack', 'prealigned stack', 'aligned stack']
	for tiltseries, oneTiltSeriesDict in allResultsDict.iteritems():
		#print tiltseries
		subplotNum += 1
		suffix = tiltseries.split('.')[-1]
		#errorsDict1[tiltseries] = collections.OrderedDict()
		errorsDict2[tiltseries] = collections.OrderedDict()
		i = 0
		#average yLeftNew and yRightNew after scaling the slopes
		nBoxes = len(oneTiltSeriesDict)
		nTiltLeft = len(xLeft0)
		nTiltRight = len(xRight0)
		#print nBoxes, nTiltLeft, nTiltRight
		yLeftNewArray = np.resize(np.array([]), (nBoxes, nTiltLeft))
		yRightNewArray = np.resize(np.array([]), (nBoxes, nTiltRight))
		#yNewArray = np.resize(np.array([]), (nBoxes * 2, nTiltLeft))
		
		for boxPosition, valueDict in oneTiltSeriesDict.iteritems():
			xLeft = valueDict['xLeft']
			yLeftNew = valueDict['yLeftNew']
			yLeftNewArray[i, :] = yLeftNew
			#yNewArray[i, :] = yLeftNew
			
			xRight = valueDict['xRight']
			yRightNew = valueDict['yRightNew']
			yRightNewArray[i, :] = yRightNew
			#yNewArray[i+1, :] = yRightNew
			i += 1
		
		yLeftNewAverage = np.mean(yLeftNewArray, axis = 0)
		yLeftNewStd = np.std(yLeftNewArray, axis = 0)
		#A, B = linearRegression2(xLeft, yLeftNewAverage)
		#fitLeftNew = A * xLeft + B
		
		yRightNewAverage = np.mean(yRightNewArray, axis = 0)
		yRightNewStd = np.std(yRightNewArray, axis = 0)
		#C, D = linearRegression2(xRight, yRightNewAverage)
		#fitRightNew = C * xRight + D
		
		#calculate the fitting line to both yLeftNewAverage and yRightNewAverage
		x = np.append(xLeft, xRight)
		y = np.append(yLeftNewAverage, yRightNewAverage)
		A, B = linearRegression2(x, y)
		#print "A=", A
		#print "B=", B
		fitLeftNew = A * xLeft + B
		fitRightNew = A * xRight + B
		
		if (options.saveAverageLinearData):
			txtFile = tiltseries.replace('.', '_') + '_averagedLinear.txt'
			print "Save averaged linear data to %s"%txtFile
			saveAverageLinearData(txtFile, xLeft, yLeftNewAverage, yLeftNewStd, fitLeftNew, xRight, yRightNewAverage, yRightNewStd, fitRightNew)
		
		plt.subplot(subplotLoc[subplotNum])
		plt.errorbar(xLeft, yLeftNewAverage, yerr = yLeftNewStd, fmt='s', color = 'c', ecolor='c', label = 'Left')
		plt.plot(xLeft, fitLeftNew, 'c--')
		plt.errorbar(xRight, yRightNewAverage, yerr = yRightNewStd, fmt='o', color = 'm', ecolor='m', label = 'Right')
		plt.plot(xRight, fitRightNew, 'm--')
		#plt.title(tiltseries)
		plt.title(labels[subplotNum], fontsize = 24, y=1.05)
		#plt.xlabel('1/cos(theta(tlt))', fontsize = 18)
		plt.xlabel(r'1/cos($\theta$)', fontsize = 24, labelpad = 10)
		plt.ylabel('ln(Intensity)', fontsize = 24, labelpad = 10)
		plt.xticks(fontsize = 20)
		plt.yticks(fontsize = 20)
		plt.legend(fontsize = 22)
		#plt.xlim(0.99,2.01)
		#plt.ylim(8.5,9.1)
		ax = plt.gca()
		ax.tick_params(pad = 10)
		
		resLeftNew = 1000 * ((yLeftNewAverage - fitLeftNew)/fitLeftNew)
		resRightNew = 1000 * ((yRightNewAverage - fitRightNew)/fitRightNew)
		resNew = np.append(resLeftNew, resRightNew)
		
		res2LeftNew = 1000000 * np.square((yLeftNewAverage - fitLeftNew)/fitLeftNew)
		res2RightNew = 1000000 * np.square((yRightNewAverage - fitRightNew)/fitRightNew)
		res2New = np.append(res2LeftNew, res2RightNew)
		#print len(res2New), np.mean(res2New), np.std(res2New)
		
		if (suffix == 'st'):
			stMean, stStd = np.mean(res2New), np.std(res2New)
			errorsDict2[tiltseries]['allRes'] = np.sort(res2New)
			#errorsDict1[tiltseries]['allRes'] = np.sort(resNew)

		elif (suffix == 'preali'):
			prealiMean, prealiStd = np.mean(res2New), np.std(res2New)
			errorsDict2[tiltseries]['allRes'] = np.sort(res2New)
			#errorsDict1[tiltseries]['allRes'] = np.sort(resNew)
			
		else:
			aliMean, aliStd = np.mean(res2New), np.std(res2New)
			errorsDict2[tiltseries]['allRes'] = np.sort(res2New)
			#errorsDict1[tiltseries]['allRes'] = np.sort(resNew)
	
	ref = stMean
	stMean, stStd= stMean/ref, stStd/ref
	prealiMean, prealiStd = prealiMean/ref, prealiStd/ref
	aliMean, aliStd = aliMean/ref, aliStd/ref		
	
	mu = [stMean, prealiMean, aliMean]
	std = [stStd, prealiStd, aliStd]
	
	for tiltseries, oneErrorsDict in errorsDict2.iteritems():
		errorsDict2[tiltseries]['allRes'] = errorsDict2[tiltseries]['allRes']/ref
	
	if (options.saveResErrors):
		saveErrorsData(errorsDict2)
		
	print "\n********Statistic percentage results:********"
	print "st:", stMean, stStd
	print 'preali:', prealiMean, prealiStd
	print 'ali:', aliMean, aliStd
	
	#plot error values
	titleStr = tiltseries.split('.')[0]
	labels = ['raw stack', 'preali', 'ali']
	#linestyles = ['-.', '--', '-']
	linestyles = ['-', '-', '-']
	colors = ['b', 'g', 'r']
	
	i = -1
	for tiltseries, oneErrorsDict in errorsDict2.iteritems():
		#print tiltseries
		suffix = tiltseries.split('.')[-1]
		i+=1
		for branch, errors in oneErrorsDict.iteritems():
			#print np.mean(errors), np.std(errors)
			plt.subplot(subplotLoc[subplotNum + 1])
			plt.plot(errors, linestyles[i], label=labels[i], linewidth=2.0, color = colors[i])
			#plt.title(titleStr)
			plt.legend()
			plt.xlabel('Rank', fontsize = 24)
			#plt.ylabel('Percentage(*100%)', fontsize = 18)
			plt.ylabel('Ratio', fontsize = 24, labelpad = 10)
			plt.xticks(fontsize = 20)
			plt.yticks(fontsize = 20)
			plt.legend(loc=2, fontsize = 22)
			ax = plt.gca()
			ax.tick_params(pad = 10)
			
	#std = [(0, 0, 0), std]
	N = len(mu)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars
	plt.subplot(subplotLoc[subplotNum + 2])
	rects0 = plt.bar(ind[0], mu[0], width, color='b', yerr=np.array([[ 0. ],[std[0]]]), ecolor='k')
	rects1 = plt.bar(ind[1], mu[1], width, color='g', yerr=np.array([[ 0. ],[std[1]]]), ecolor='k')
	rects2 = plt.bar(ind[2], mu[2], width, color='r', yerr=np.array([[ 0. ],[std[2]]]), ecolor='k')
	
	#plt.ylabel('Percentage(*100%)', fontsize = 18)
	plt.ylabel('Ratio', fontsize = 24, labelpad = 10)
	title = options.tiltseries.split('.')[0:-1]
	#plt.title(title[0])
	labels = ['raw stack', 'preali', 'ali']
	plt.xticks(ind+width/2, labels, fontsize = 24)
	#plt.xticks(fontsize = 14)
	plt.yticks(fontsize = 20)
	plt.legend([rects0, rects1, rects2], ['raw stack', 'preali', 'ali'], fontsize = 22)
	ax = plt.gca()
	ax.tick_params(pad = 10)

	pdfName = tiltseries.split('.')[0]+'_%dboxes_panel.pdf'%nBoxes
	print pdfName
	with PdfPages(pdfName) as pdf:
		pdf.savefig()
		plt.close()
	

def saveErrorsData(errorsDict):
	
	stLeft = []
	stRight = []
	prealiLeft = []
	prealiRight = []
	aliLeft = []
	aliRight = []
	
	for tiltseries, oneErrorsDict in errorsDict.iteritems():
		suffix = tiltseries.split('.')[-1]
		if (suffix == 'st'):
			stLeft = errorsDict[tiltseries]['allResLeft']
			stRight = errorsDict[tiltseries]['allResRight']
		elif (suffix == 'preali'):
			prealiLeft = errorsDict[tiltseries]['allResLeft']
			prealiRight = errorsDict[tiltseries]['allResRight']
		else:
			aliLeft = errorsDict[tiltseries]['allResLeft']
			aliRight = errorsDict[tiltseries]['allResRight']

	#print aliLeft
	#print aliRight
	titleStr = tiltseries.split('.')[0] + '_errorsData.txt'
	print "\nData of errors plot is saved to %s"%titleStr
	fp = open(titleStr, 'w')
	fp.write('N\tstLeft\tprealiLeft\taliLeft\tstRight\tprealiRight\taliRight\n')
	for i in range(len(stLeft)):
		line = "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n"%(i, stLeft[i], prealiLeft[i], aliLeft[i], stRight[i], prealiRight[i], aliRight[i])
		fp.write(line)
	fp.close()


def saveCurvesData(txtFile, curveArray):
	fp = open(txtFile, 'w')
	nRow, nCol = curveArray.shape
	for i in range(nRow):
		newLine = ''
		for j in range(nCol):
			tmp = '%.12g\t'%(curveArray[i, j])
			#tmp = str(curveArray[i, j]) + ' '
			newLine = newLine + tmp
		#print newLine
		newLine = newLine + '\n'
		fp.write(newLine)
	fp.close()
	
def saveAverageLinearData(txtFile, xLeft, yLeftNewAverage, yLeftNewStd, fitLeftNew, xRight, yRightNewAverage, yRightNewStd, fitRightNew):
	fp = open(txtFile, 'w')
	fp.write('xLeft\tyLeftNewAverage\tyLeftNewStd\tfitLeftNew\txRight\tyRightNewAverage\tyRightNewStd\tfitRightNew\n')
	for i in range(len(xLeft)):
		line = "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n"%(xLeft[i], yLeftNewAverage[i], yLeftNewStd[i], fitLeftNew[i], xRight[i], yRightNewAverage[i], yRightNewStd[i], fitRightNew[i])
		fp.write(line)
	fp.close()
	


def fitLinearRegression3(thetaLinear, IntensityLinear, tiltanglesArray, thetaCurve, IntensityCurve, options):
	x0 = [int(x) for x in options.x0.split(',')]
	y0 = [int(y) for y in options.y0.split(',')]
	
	resultDict = collections.OrderedDict()
	#returnDict = collections.OrderedDict()
	
	allResLeft = []
	allResRight = []
	for i in range(len(x0)):
		iIntensityLinear = IntensityLinear[:, i]
		iIntensityCurve = IntensityCurve[:, i]
		key = '%g %g'%(x0[i], y0[i])
		
		ret = fitOneLinearRegression(thetaLinear, iIntensityLinear, tiltanglesArray, options)
		fres, stdRes, xLeft, yLeft, fitLeft, xRight, yRight, fitRight, indexLargeLeft, indexLargeRight, indexSmallLeft, indexSmallRight, resLeft, resRight, slopeLeft, interceptLeft, slopeRight, interceptRight = ret
		resultDict[key] = {}
		resultDict[key]['SSE'] = fres
		resultDict[key]['intensityCurve'] = iIntensityCurve
		resultDict[key]['tiltAngles'] = thetaCurve
		resultDict[key]['stdRes'] = stdRes
		resultDict[key]['xLeft'] = xLeft
		resultDict[key]['yLeft'] = yLeft
		resultDict[key]['fitLeft'] = fitLeft
		resultDict[key]['xRight'] = xRight
		resultDict[key]['yRight'] = yRight
		resultDict[key]['fitRight'] = fitRight
		resultDict[key]['indexLargeLeft'] = indexLargeLeft
		resultDict[key]['indexLargeRight'] = indexLargeRight
		resultDict[key]['indexSmallLeft'] = indexSmallLeft
		resultDict[key]['indexSmallRight'] = indexSmallRight
		resultDict[key]['resLeft'] = resLeft
		resultDict[key]['resRight'] = resRight
		resultDict[key]['slopeLeft'] = slopeLeft
		resultDict[key]['interceptLeft'] = interceptLeft
		resultDict[key]['slopeRight'] = slopeRight
		resultDict[key]['interceptRight'] = interceptRight
		
		allResLeft.append(resLeft.tolist())
		allResRight.append(resRight.tolist())

			
	allResLeft = flattenList(allResLeft)
	allResRight = flattenList(allResRight)
	#rescale all linear trends to one common trend
	resultDict = modifyResultDict2(resultDict)

	return resultDict


def modifyResultDict2(oneResultDict):
	slopesLeftArray = np.array([])
	slopesRightArray = np.array([])
	interceptLeftArray = np.array([])
	interceptRightArray = np.array([])
	boxPosLeft = []
	boxPosRight = []
	for boxPosition, value in oneResultDict.iteritems():
		slopesLeftArray = np.append(slopesLeftArray, value['slopeLeft'])
		slopesRightArray = np.append(slopesRightArray, value['slopeRight'])
		interceptLeftArray = np.append(interceptLeftArray, value['interceptLeft'])
		interceptRightArray = np.append(interceptRightArray, value['interceptRight'])
		boxPosLeft.append(boxPosition)
		boxPosRight.append(boxPosition)
	
	slopeArray = np.append(slopesLeftArray, slopesRightArray)
	slopeMedian = np.median(slopeArray)
	interceptArray = np.append(interceptLeftArray, interceptRightArray)
	interceptMedian = np.median(interceptArray)
	
	#scale left and right linear treands to one linear cluster
	for boxPosition, value in oneResultDict.iteritems():
		xLeft = value['xLeft']
		ysLeft = slopeMedian * xLeft + interceptMedian
		y0Left = value['yLeft']
		a, b = linearRegression2(y0Left, ysLeft)
		yScaleLeft = a * y0Left + b
		value['yLeftNew'] = yScaleLeft
		
		xRight = value['xRight']
		ysRight = slopeMedian * xRight + interceptMedian
		y0Right = value['yRight']
		c, d = linearRegression2(y0Right, ysRight)
		yScaleRight = c * y0Right + d
		value['yRightNew'] = yScaleRight

	return oneResultDict


def fitOneLinearRegression(thetaLinear, IntensityLinear, tiltanglesArray, options):
	if (len(tiltanglesArray)%2 == 1):
		halfN = int(len(tiltanglesArray)/2) + 1
		xLeft, yLeft = thetaLinear[0:halfN], IntensityLinear[0:halfN]
		xRight, yRight = thetaLinear[halfN-1:], IntensityLinear[halfN-1:]
		
	else:
		halfN = int(len(tiltanglesArray)/2)
		xLeft, yLeft = thetaLinear[0:halfN], IntensityLinear[0:halfN]
		xRight, yRight = thetaLinear[halfN:], IntensityLinear[halfN:]
	
	slopeLeft, interceptLeft, r2Left = linearRegression(xLeft, yLeft)
        slopeRight, interceptRight, r2Right = linearRegression(xRight, yRight)
	
	assert(len(xLeft)==len(xRight))
	
	fitLeft = slopeLeft*xLeft + interceptLeft
        fitRight = slopeRight*xRight + interceptRight
        
        #the sum of squared residuals
        resLeft = yLeft - fitLeft
	resLeft = resLeft / fitLeft
	#print "resLeft", resLeft
        resRight = yRight - fitRight
	resRight = resRight / fitRight
	#print "resRight", resRight
	
	fresLeft = sum(resLeft**2)
        fresRight = sum(resRight**2)
	fres = [fresLeft*1000000, fresRight*1000000]

	#find the points with the largest 3 residuals in left and right branches, use numpy.argpartition
	N = options.largestNRes
        negN = (-1)*N
        indexLargeLeft = np.argpartition(resLeft**2, negN)[negN:]
        indexLargeRight = np.argpartition(resRight**2, negN)[negN:]
	
	M = options.smallestNRes
	posM = M
	indexSmallLeft = np.argpartition(resLeft**2, posM)[:posM]
	indexSmallRight = np.argpartition(resRight**2, posM)[:posM]
	
        #MSE, under the assumption that the population error term has a constant variance, the estimate of that variance is given by MSE, mean square error
        #The denominator is the sample size reduced by the number of model parameters estimated from the same data, (n-p) for p regressors or (n-p-1) if an intercept is used.
        #In this case, p=1 so the denominator is n-2.
        stdResLeft = np.std(resLeft, ddof=2)
        stdResRight = np.std(resRight, ddof=2)
	stdRes = [stdResLeft*1000, stdResRight*1000]
	ret = fres, stdRes, xLeft, yLeft, fitLeft, xRight, yRight, fitRight, indexLargeLeft, indexLargeRight, indexSmallLeft, indexSmallRight, resLeft, resRight, slopeLeft, interceptLeft, slopeRight, interceptRight
	return ret

def linearRegression(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        # To get slope, intercept and coefficient of determination (r_squared)
        return slope, intercept, r_value**2

def linearRegression2(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        # To get slope, intercept and coefficient of determination (r_squared)
        return slope, intercept

		
        
def generateData2(dictionary, options):
	x0 = [int(x) for x in options.x0.split(',')]
	thetaLst = []
        intensityLst = []
        thetaLinearLst = []
	
	for theta, intensity in dictionary.iteritems():
		thetaLst.append(float(theta))
		intensityLst.append(intensity)
		cosAngle = math.cos((float(theta)/360.)*math.pi*2)
		tmp = (1./(cosAngle))
		thetaLinearLst.append(tmp)
		
	thetaArray = np.asarray(thetaLst)
	thetaLinearArray = np.asarray(thetaLinearLst)	
	intensityArray = np.asarray(intensityLst)
	intensityLinearArray = np.log(intensityArray)	

	return thetaArray, intensityArray, thetaLinearArray, intensityLinearArray        
        
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
        

def flattenList(nestedLst):
	flattenLst = list(chain.from_iterable(nestedLst))
	return flattenLst
	

	
def blockMean(img, boxsizeX, boxsize):
        nx, ny = img.get_xsize(), img.get_ysize()
        nxBlock = int(nx/boxsizeX)
        nyBlock = int(ny/boxsize)
    
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

	if (options.addOffset):
		blkMean = blkMean + 32768  #use for T7MAY29043.ali
	elif (options.inversePixel):
		blkMean = blkMean * (-1) + 32768 #use for virusJAN15003.ali
	else:
		blkMean = math.fabs(blkMean)
	
	if (options.useSigma):
		return blkSigma
	else:
		return blkMean
    

def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]  

	
	
if __name__ == "__main__":
    main()