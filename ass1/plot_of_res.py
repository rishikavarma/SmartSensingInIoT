import cv2
import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image 
import scipy.fftpack 

arr=["./sample54.mp4","./sample108.mp4","./sample216.mp4","./sample360.mp4","./sample540.mp4","./sample720.mp4","./sample.mp4"]
px=[54,108,216,360,540,720,1080]
py=[96,192,384,640,960,1280,1920]
ary1=[0,0,0,0,0]
ary2=[0,0,0,0,0]
id=0
f=1
for path in arr:
  cam = cv2.VideoCapture(path)
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
  b, a = scipy.signal.butter(1, 0.075)
  filtered = scipy.signal.filtfilt(b, a, array)
  peak_indices = signal.find_peaks(filtered)[0]
  peak_count = len(peak_indices)
  ary2[id]=(peak_count)*fps*60/len(array)
  
  id=id+1
plt.figurefigsize = (12, 6);
fig,(ax1,ax2)=plt.subplots(1,2)
fig.suptitle('Variation of heartrate value with resolution')
ax1.set_ylim([0, 100])
ax1.plot(px,ary1)
ax1.set_title('FFT')
ax2.set_ylim([40, 120])
ax2.plot(px,ary2)
ax2.set_title('Peak detection')
plt.show()