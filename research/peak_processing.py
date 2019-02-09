import os, string, glob, sys
from scipy import *
import scipy.io.array_import

EXT='*.x_y'
TEMPEXT='*_temp.dat'
OUTFILE ='tixrpd_out.dat'
PLOTFILE ='tixrpd_plot.eps'
TEMPFILE=glob.glob(os.getcwd()+'/'+TEMPEXT)[0]
TIMESTEP=150

LOWER=27.
UPPER=28.5
A=3000.
B=27.5
C=0.1
D=300
initpar= []
initpar.append(A), initpar.append(B), initpar.append(C), initpar.append(D)

tempdata = scipy.io.array_import.read_array(TEMPFILE)
FILELIST=glob.glob(EXT)
dataarray = zeros((len(FILELIST),len(initpar)*2+3), Float)
reduced = zeros((len(FILELIST),len(initpar)*2+3), Float)

for filenumber in range(len(FILELIST)):
	stdin, stdout, stderr = os.popen3('gnuplot')
	print >>stdin, "set xrange [%f:%f]" % (LOWER,UPPER)
	print >>stdin, "f(x) = d + a*exp(-0.5*((x-b)/c)**2)" 
	print >>stdin, "a = %f; b=%f; c=%f; d=%f" % (A,B,C,D)
	print >>stdin, "set fit logfile 'fit1.log'"
	print >>stdin, "set fit errorvariables"
	print >>stdin, "fit f(x) '%s' using 1:2:(sqrt($2)) via a, b, c, d" % FILELIST[filenumber]
	if len(sys.argv) > 1 and sys.argv[1] == 'plot':
		print >>stdin, "plot f(x) with lines, \
		'%s' using 1:2:(sqrt($2)) with errorbars; \
		pause 1" % FILELIST[filenumber] 
	print >>stdin, "print a,a_err,b,b_err,c,c_err,d,d_err"
	stdin.close()
	stdout.close()
	fitpar = string.split(stderr.readlines()[-1:][0])
	FAIL=1
	if float(fitpar[4]) > 0.02 and float(fitpar[0]) > float(fitpar[1]) \
	and 2*float(fitpar[5]) < 0.05 and float(fitpar[2]) > 0 \
	and 2*float(fitpar[5]) < 0.5*float(fitpar[4]) \
	and float(fitpar[2]) < UPPER and float(fitpar[2]) > LOWER: 
		FAIL=0
		A = float(fitpar[0])
		for data in range(len(initpar)*2):
			dataarray[filenumber,0]=float(filenumber)+1
			dataarray[filenumber,1]=dataarray[filenumber,0]*TIMESTEP
			dataarray[filenumber,2]=tempdata[filenumber,1]
			dataarray[filenumber,data+3]=float(fitpar[data])
	if len(sys.argv) > 2 and sys.argv[2] == 'data':
		print dataarray[filenumber,:]
	if FAIL==0:
		SUCCES='succes'
	else:
		SUCCES='failure'
	print "Processing file nr.: %s of %s with %s" % (filenumber, len(FILELIST), SUCCES)
	stderr.close() 

count=0
for rowindex in range(len(dataarray[:,0])):
	if not sum(dataarray[rowindex,:])==0:
		reduced[count,:]=dataarray[rowindex,:]	
		count = count +1
dataarray=resize(reduced, (count,len(reduced[0,:])))

file = open(OUTFILE, 'w')
for linenr in range(len(dataarray[:,0])):
	file.write("%d %d %3.1f %5.2f %3.0f %3.5f %1.5f %1.6f %1.6f %d %d\n" % \
	(dataarray[linenr,0],\
	dataarray[linenr,1], dataarray[linenr,2], \
	dataarray[linenr,3], dataarray[linenr,4], \
	dataarray[linenr,5], dataarray[linenr,6], \
	dataarray[linenr,7], dataarray[linenr,8], \
	dataarray[linenr,9], dataarray[linenr,10], \
	))
file.close()

f=os.popen('gnuplot' ,'w')
print >>f, "set terminal postscript enhanced 14 portrait; set out '%s'" % PLOTFILE
print >>f, "set size 1,1; set xlabel 'Time [s]'; set ylabel 'Temperature [{^o}C]' ; set origin 0,0; set lmargin 10; set rmargin 2; set multiplot; set size 1,0.26; set bmargin 3; set tmargin 0"
print >>f, "plot '%s' using 2:3 with  linespoints notitle " % OUTFILE
print >>f, " set bmargin 0; set tmargin 0; set ylabel 'Amplitude [a.u.]';set format x ' '; set size 1,0.23; set origin 0,0.26; set xlabel ' '"
print >>f, "plot '%s' using 2:4:5 with errorbars notitle " % OUTFILE
print >>f, " set bmargin 0; set tmargin 0; set ylabel 'Peak pos. [{^o}]'; set size 1,0.23; set origin 0,0.49;  set xlabel ' '"
print >>f, "plot '%s' using 2:6:7 with errorbars notitle" % OUTFILE
print >>f, "set bmargin 0; set tmargin 0; set size 1,0.23; set ylabel 'Peak half width [{^o}]'; set origin 0,0.72; set xlabel ' '"
print >>f, "plot '%s' using 2:8:9 with errorbars notitle" % OUTFILE
print >>f, "replot"
f.flush()

status = os.system('ggv %s' % PLOTFILE)
print "Status: ", status
status = os.system('pstopnm %s' % PLOTFILE)
print "Status: ", status
