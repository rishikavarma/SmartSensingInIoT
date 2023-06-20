import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
from haversine import haversine, Unit
from math import radians
import statistics
import seaborn as sns
from tabulate import tabulate
f1 = open("inside.txt")
f2 = open("window.txt")
f3 = open("outside.txt")
a=f1.readlines()
lat=[]
lon=[]
no_of_sat=[]
temp=0
snr=[]
avg_snr=[]
azimuth={}
elevate={}
avsnr={}
for x in a:
	words=x.split(',')
	if len(words)>4:
		if (words[0]=='Fix') & (words[1]=='GPS'):
			lat.append(float(words[2]))
			lon.append(float(words[3]))
			no_of_sat.append(temp)
			if len(snr)!=0:
				avg_snr.append(statistics.mean(snr))
			else:
				avg_snr.append(0)
			snr=[]
			temp=0
		elif (words[0]=='Status') & (words[4]=='1'):
			th=int(words[10])
			temp=temp+th
			if th==1:
				snr.append(float(words[7]))
				if words[5] in azimuth:
					avsnr[words[5]].append(float(words[7]))
					azimuth[words[5]].append(float(words[8]))
					elevate[words[5]].append(float(words[9]))
				else:
					avsnr[words[5]]=[float(words[7])]
					azimuth[words[5]]=[float(words[8])]
					elevate[words[5]]=[float(words[9])]
mlat=statistics.mean(lat)
mlon=statistics.mean(lon)
sat_num=statistics.median(no_of_sat)
print("Mean latitude:",end=' ')
print(mlat)
print("Mean longitude:",end=' ')
print(mlon)
print("Median of number of satellites:",end=' ')
print(sat_num)
t2=(mlat,mlon)
error=[]
for (x,y) in zip(lat,lon):
	t1=(x,y)
	res=haversine(t1,t2)
	error.append(res)
err_var=statistics.variance(error)
print("Variance of Haversine Error:",end=' ')
print(err_var)
plt.figure(figsize=(8,6))
sns.ecdfplot(error)
plt.xlabel('Error(Km)')
plt.ylabel('CDF')
plt.title('Plotting CDF of Error')
plt.show()
plt.scatter(no_of_sat,error)
plt.xlabel('Number of Satellites')
plt.ylabel('Error(Km)')
plt.title('Plot of Error vs number of satellites')
plt.show()
plt.scatter(error,avg_snr)
plt.xlabel('Error(Km)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Error')
plt.show()
keys=avsnr.keys()
table=[]
azpl=[]
elpl=[]
snpl=[]
for key in keys:
	table.append([key,statistics.mean(azimuth[key]),statistics.mean(elevate[key]),statistics.mean(avsnr[key])])
	azpl.append(statistics.mean(azimuth[key]))
	elpl.append(statistics.mean(elevate[key]))
	snpl.append(statistics.mean(avsnr[key]))
print (tabulate(table, headers=["Svid", "Azimuth", "Elevation", "Snr"], tablefmt="fancy_grid"))
plt.scatter(azpl,snpl)
plt.xlabel('Azimuth(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Azimuth')
plt.show()
plt.scatter(elpl,snpl)
plt.xlabel('Elevation(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Elevation')
plt.show()
ax = plt.subplot(111, projection='polar')
ax.scatter(azpl, snpl)
# plt.xlabel('Error(Km)')
# plt.ylabel('Snr')
plt.title('Angular Plot of Snr vs Azimuth')
plt.show()
f1.close()
a=f2.readlines()
lat=[]
lon=[]
no_of_sat=[]
temp=0
snr=[]
avg_snr=[]
azimuth={}
elevate={}
avsnr={}
for x in a:
	words=x.split(',')
	if len(words)>4:
		if (words[0]=='Fix') & (words[1]=='GPS'):
			lat.append(float(words[2]))
			lon.append(float(words[3]))
			no_of_sat.append(temp)
			if len(snr)!=0:
				avg_snr.append(statistics.mean(snr))
			else:
				avg_snr.append(0)
			snr=[]
			temp=0
		elif (words[0]=='Status') & (words[4]=='1'):
			th=int(words[10])
			temp=temp+th
			if th==1:
				snr.append(float(words[7]))
				if words[5] in azimuth:
					avsnr[words[5]].append(float(words[7]))
					azimuth[words[5]].append(float(words[8]))
					elevate[words[5]].append(float(words[9]))
				else:
					avsnr[words[5]]=[float(words[7])]
					azimuth[words[5]]=[float(words[8])]
					elevate[words[5]]=[float(words[9])]
mlat=statistics.mean(lat)
mlon=statistics.mean(lon)
sat_num=statistics.median(no_of_sat)
print("Mean latitude:",end=' ')
print(mlat)
print("Mean longitude:",end=' ')
print(mlon)
print("Median of number of satellites:",end=' ')
print(sat_num)
t2=(mlat,mlon)
error=[]
for (x,y) in zip(lat,lon):
	t1=(x,y)
	res=haversine(t1,t2)
	error.append(res)
err_var=statistics.variance(error)
print("Variance of Haversine Error:",end=' ')
print(err_var)
plt.figure(figsize=(8,6))
sns.ecdfplot(error)
plt.xlabel('Error(Km)')
plt.ylabel('CDF')
plt.title('Plotting CDF of Error')
plt.show()
plt.scatter(no_of_sat,error)
plt.xlabel('Number of Satellites')
plt.ylabel('Error(Km)')
plt.title('Plot of Error vs number of satellites')
plt.show()
plt.scatter(error,avg_snr)
plt.xlabel('Error(Km)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Error')
plt.show()
keys=avsnr.keys()
table=[]
azpl=[]
elpl=[]
snpl=[]
for key in keys:
	table.append([key,statistics.mean(azimuth[key]),statistics.mean(elevate[key]),statistics.mean(avsnr[key])])
	azpl.append(statistics.mean(azimuth[key]))
	elpl.append(statistics.mean(elevate[key]))
	snpl.append(statistics.mean(avsnr[key]))
print (tabulate(table, headers=["Svid", "Azimuth", "Elevation", "Snr"], tablefmt="fancy_grid"))
plt.scatter(azpl,snpl)
plt.xlabel('Azimuth(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Azimuth')
plt.show()
plt.scatter(elpl,snpl)
plt.xlabel('Elevation(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Elevation')
plt.show()
ax = plt.subplot(111, projection='polar')
ax.scatter(azpl, snpl)
# plt.xlabel('Error(Km)')
# plt.ylabel('Snr')
plt.title('Angular Plot of Snr vs Azimuth')
plt.show()
f2.close()
a=f3.readlines()
lat=[]
lon=[]
no_of_sat=[]
temp=0
snr=[]
avg_snr=[]
azimuth={}
elevate={}
avsnr={}
for x in a:
	words=x.split(',')
	if len(words)>4:
		if (words[0]=='Fix') & (words[1]=='GPS'):
			lat.append(float(words[2]))
			lon.append(float(words[3]))
			no_of_sat.append(temp)
			if len(snr)!=0:
				avg_snr.append(statistics.mean(snr))
			else:
				avg_snr.append(0)
			snr=[]
			temp=0
		elif (words[0]=='Status') & (words[4]=='1'):
			th=int(words[10])
			temp=temp+th
			if th==1:
				snr.append(float(words[7]))
				if words[5] in azimuth:
					avsnr[words[5]].append(float(words[7]))
					azimuth[words[5]].append(float(words[8]))
					elevate[words[5]].append(float(words[9]))
				else:
					avsnr[words[5]]=[float(words[7])]
					azimuth[words[5]]=[float(words[8])]
					elevate[words[5]]=[float(words[9])]
mlat=statistics.mean(lat)
mlon=statistics.mean(lon)
sat_num=statistics.median(no_of_sat)
print("Mean latitude:",end=' ')
print(mlat)
print("Mean longitude:",end=' ')
print(mlon)
print("Median of number of satellites:",end=' ')
print(sat_num)
t2=(mlat,mlon)
error=[]
for (x,y) in zip(lat,lon):
	t1=(x,y)
	res=haversine(t1,t2)
	error.append(res)
err_var=statistics.variance(error)
print("Variance of Haversine Error:",end=' ')
print(err_var)
plt.figure(figsize=(8,6))
sns.ecdfplot(error)
plt.xlabel('Error(Km)')
plt.ylabel('CDF')
plt.title('Plotting CDF of Error')
plt.show()
plt.scatter(no_of_sat,error)
plt.xlabel('Number of Satellites')
plt.ylabel('Error(Km)')
plt.title('Plot of Error vs number of satellites')
plt.show()
plt.scatter(error,avg_snr)
plt.xlabel('Error(Km)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Error')
plt.show()
keys=avsnr.keys()
table=[]
azpl=[]
elpl=[]
snpl=[]
for key in keys:
	table.append([key,statistics.mean(azimuth[key]),statistics.mean(elevate[key]),statistics.mean(avsnr[key])])
	azpl.append(statistics.mean(azimuth[key]))
	elpl.append(statistics.mean(elevate[key]))
	snpl.append(statistics.mean(avsnr[key]))
print (tabulate(table, headers=["Svid", "Azimuth", "Elevation", "Snr"], tablefmt="fancy_grid"))
plt.scatter(azpl,snpl)
plt.xlabel('Azimuth(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Azimuth')
plt.show()
plt.scatter(elpl,snpl)
plt.xlabel('Elevation(Degrees)')
plt.ylabel('Snr')
plt.title('Plot of Snr vs Elevation')
plt.show()
ax = plt.subplot(111, projection='polar')
ax.scatter(azpl, snpl)
# plt.xlabel('Error(Km)')
# plt.ylabel('Snr')
plt.title('Angular Plot of Snr vs Azimuth')
plt.show()
f3.close()