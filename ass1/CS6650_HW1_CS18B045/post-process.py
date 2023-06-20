import cv2
import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image 
import scipy.fftpack 

name="Data9"
print("                                                                               Analysis of",end=" ")
print(name)
pa="./Dataset/"+name+".mp4"
cam = cv2.VideoCapture(pa) 
if(cam.isOpened() == False):
	print("Error: Couldn't open Video")
fps_ac= int(cam.get(cv2.CAP_PROP_FPS))
total = 0
f=1
p=1
i=0
j=0
fps=fps_ac/f
total = int(cam.get(7))
array=np.zeros(math.ceil(total/f))
time=math.floor(fps*5)
fftval=[]
peakval=[]
while(True): 
    ret,frame = cam.read() 
    if ret: 
    	x,y,z=frame.mean(axis=0).mean(axis=0)
    	if i%f==0 :
    		array[j]=z
    		j=j+1
    	i=i+1

    else: 
        break
    if i%time==0:
        signalFFT = scipy.fftpack.fft(array[0 : j])
        signalPSD = np.abs(signalFFT) ** 2
        fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
        fi=fftFreq>1
        temp=fftFreq[fi]
        idx=np.argmax(10*np.log10(signalPSD[fi]))
        val1=round(60*temp[idx])
        fftval.append(val1)
        times = np.arange(len(array))/fps
        b, a = scipy.signal.butter(1, 0.06)
        filtered = scipy.signal.filtfilt(b, a, array)
        peak_indices = signal.find_peaks(filtered)[0]
        peak_count = len(peak_indices)
        val2=round((peak_count)*fps*60/j)
        peakval.append(val2)
cam.release() 
cv2.destroyAllWindows() 
print("Values given by FFT:")
print("\n")
for x in fftval:
    print(x)
print("\n")
print("Values given by peak detection algorithm:")
print("\n")
for x in peakval:
    print(x)
print("\n")
print("Final values:", end="      ")
print("\n")
signalFFT = scipy.fftpack.fft(array)
signalPSD = np.abs(signalFFT) ** 2
fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
fi=fftFreq>1
temp=fftFreq[fi]
idx=np.argmax(10*np.log10(signalPSD[fi]))
val1=round(60*temp[idx])

times = np.arange(len(array))/fps
b, a = scipy.signal.butter(1, 0.06)
filtered = scipy.signal.filtfilt(b, a, array)
peak_indices = signal.find_peaks(filtered)[0]
peak_count = len(peak_indices)
val2=round((peak_count)*fps*60/len(array))
print(val1,end=",         ")
print(val2)
print("\n")
arx=[1,2,3,4,5,6,10,12,15,20]
ary1=[0,0,0,0,0,0,0,0,0,0]
ary2=[0,0,0,0,0,0,0,0,0,0]
id=0
for f in arx:
  cam = cv2.VideoCapture(pa)
  if(cam.isOpened() == False):
    print("Error: Couldn't open Video")
  fps_ac= int(cam.get(cv2.CAP_PROP_FPS))
  total = 0
  i=0
  j=0
  fps=fps_ac/f
  total = int(cam.get(7))
  array=np.zeros(math.ceil(total/f))
  time=math.floor(fps*5)
  while(True): 
      ret,frame = cam.read() 
      if ret: 
        x,y,z=frame.mean(axis=0).mean(axis=0)
        if i%f==0 :
          array[j]=z
          j=j+1
        i=i+1

      else: 
          break
  cam.release() 
  cv2.destroyAllWindows() 
  signalFFT = scipy.fftpack.fft(array)
  signalPSD = np.abs(signalFFT) ** 2
  fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
  fi=fftFreq>1
  temp=fftFreq[fi]
  idx=np.argmax(10*np.log10(signalPSD[fi]))
  ary1[id]=60*temp[idx]
    
  times = np.arange(len(array))/fps
  b, a = scipy.signal.butter(1, 0.06)
  filtered = scipy.signal.filtfilt(b, a, array)
  peak_indices = signal.find_peaks(filtered)[0]
  peak_count = len(peak_indices)
  ary2[id]=(peak_count)*fps*60/len(array)
  
  id=id+1
plt.figurefigsize = (12, 16);
px=[]
for i in arx:
  px.append(fps_ac/i)
fig,(ax1,ax2)=plt.subplots(1,2)
fig.suptitle('Variation of heartrate value with fps')
ax1.plot(px,ary1)
ax1.set_title('FFT')
ax2.plot(px,ary2)
ax2.set_title('Peak detection')
plt.show()


path1="./Dataset/ResolutionProcessedVideos/"+name
arr=["_54.mp4","_108.mp4","_216.mp4","_360.mp4","_540.mp4","_720.mp4","_1080.mp4"]
px=[54,108,216,360,540,720,1080]
py=[96,192,384,640,960,1280,1920]
ary1=[0,0,0,0,0,0,0]
ary2=[0,0,0,0,0,0,0]
id=0
f=1
for path in arr:
  cam = cv2.VideoCapture(path1+path)
  if(cam.isOpened() == False):
    print("Error: Couldn't open Video")
  fps_ac= int(cam.get(cv2.CAP_PROP_FPS))
  total = 0
  i=0
  j=0
  fps=fps_ac/f
  total = int(cam.get(7))
  array=np.zeros(math.ceil(total/f))
  time=math.floor(fps*5)
  while(True): 
      ret,frame = cam.read() 
      if ret: 
        x,y,z=frame.mean(axis=0).mean(axis=0)
        if i%f==0 :
          array[j]=z
          j=j+1
        i=i+1

      else: 
          break
  cam.release() 
  cv2.destroyAllWindows() 
  signalFFT = scipy.fftpack.fft(array)
  signalPSD = np.abs(signalFFT) ** 2
  fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
  fi=fftFreq>1
  temp=fftFreq[fi]
  idx=np.argmax(10*np.log10(signalPSD[fi]))
  ary1[id]=60*temp[idx]
    
  times = np.arange(len(array))/fps
  b, a = scipy.signal.butter(1, 0.06)
  filtered = scipy.signal.filtfilt(b, a, array)
  peak_indices = signal.find_peaks(filtered)[0]
  peak_count = len(peak_indices)
  ary2[id]=(peak_count)*fps*60/len(array)
  
  id=id+1
plt.figurefigsize = (12, 6);
fig,(ax1,ax2)=plt.subplots(1,2)
fig.suptitle('Variation of heartrate value with resolution')
ax1.set_ylim([40, 120])
ax1.plot(px,ary1)
ax1.set_title('FFT')
ax2.set_ylim([40, 120])
ax2.plot(px,ary2)
ax2.set_title('Peak detection')
plt.show()