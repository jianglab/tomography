# tomography
software for alignment and 3-D reconstruction of electron tomography tilt series

This project includes the Python scripts reported in these papers:

Yan, R., Edwards, T. J., Pankratz, L. M., Kuhn, R. J., Lanman, J. K., Liu, J., & Jiang, W. (2015a). **A fast cross-validation method for alignment of electron tomography images based on Beer-Lambert law.** Journal of Structural Biology, 192(2), 297–306. https://doi.org/10.1016/j.jsb.2015.10.004

Yan, R., Edwards, T. J., Pankratz, L. M., Kuhn, R. J., Lanman, J. K., Liu, J., & Jiang, W. (2015b). **Simultaneous determination of sample thickness, tilt, and electron mean free path using tomographic tilt images based on Beer-Lambert law.** Journal of Structural Biology, 192(2), 287–296. https://doi.org/10.1016/j.jsb.2015.09.019

### tomoThickness.py
This file is an instruction of *tomoThickness.py* which is used to simultaneously determine the sample thickness, sample tilt and mean free path.

Example: python tomoThickness.py --tiltseries virus009.ali --tiltangles virus009.tlt --x0 1600,1500,1600,300 --y0 1600,1700,1800,300 --boxsize 200 --B 240 --d0 100 --alpha0 0 --theta0 0 --niter 400 --plotData --plotResults --modifyTiltFile --modifiedTiltName modifiedTilt.tlt --logName myLog.log

* --tiltseries: the file name of the aligned tilt series with tilt axis along Y-axis. Usually, this is the aligned tilt series (*.ali) generated by IMOD software after fine alignment.

* --tiltangles: the file name of the tilt angle file. Usually, this is the tilt angles file (*.tlt) generated by IMOD software after fine alignment without tilt angle offset correction. If the tilt angle offset is corrected during fine alignment, the determined sample tilt in our script will not reflect the initial orientation of the specimen in 3D space.

* --x0 and --y0: these two arguments need to be used and explained together. Our script requires users to provide the coordinates of patch center in order to locate and track it throughout the entire tilt series. --x0 and --y0 are the X- and Y- coordination of the patch center, respectively. For example, you want to select 2 patches, the first patch center is (xa, ya), the second patch center is (xb, yb), you should specify like --x0 xa, xb --y0 ya, yb

* --boxsize or --adaptiveBox: the size of the patch which is clipped and tracked along the tilt series. For example, one patch center is selected as (xa, ya), the script will clip a region with X ranging from (xa-boxsize/2) to (xa+boxsize/2), Y ranging from (ya-boxsize/2) to (ya+boxsize/2) on the untilted image, and track it along the entire tilt series. Our script works for both constant and adaptive (tilt angle dependent) boxsizes along the tilt series.

* --d0 --alpha0 --theta0 --B --MFP: initials of sample thickness, sample tilt along X axis and Y axis, B, and mean free path. For vitreous ice, mean free path is usually initialized by 350nm@300kV and 300nm@200kV, when energy filter is **ON**. The initial of B is an issue sometimes. I printed out the **min average pixel value** in the output, the value of B must be smaller than the min average pixel value. We are still looking for a way to determine B internally. 
	```
	max: max average pixel value = 8701.26 @ tilt angles =  -2.05
	min: min average pixel value = 5815.55 @ tilt angles =  59.78
	```

* --niter: the number of iterations in the basinhopping method (optimization method) we use to fit the results.

* --plotData: plot of the original data before the step of optimization, including curvilinear mode and linear mode. It can be a preliminary examination about the dataset. If the distribution of data points follows the Beer-Lambert law, the script should be run successfully. Close the popup figures and continue.

* --plotResults: plot of the original data before the step of optimization, including curvilinear mode; and plot of the same tilt series after correction of the sample tilt.

* --modifyTiltFile: you can use this argument to correct the tilt angles and write out to a new tilt angle file. If the user specifies the file name by --modifiedTiltName, the new tilt angles will be written to this file; if not, the script will assign a file name (such as tiltSeriesName_modified.tlt).

* --logName: The output information is written into a log file with the file name either provided by the user (such as myLog.log in the example) or assigned by the script (such as tiltSeriesName.log).

In the output, theta0 and alpha0 are the initial angles of sample tilt around Y-axis and X-axis, respectively. And gamma0 is the 3D angle between the normal vector of the sample plane and the Z-axis.
