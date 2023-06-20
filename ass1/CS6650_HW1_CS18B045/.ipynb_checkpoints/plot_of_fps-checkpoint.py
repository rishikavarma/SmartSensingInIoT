import cv2
import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image 
import scipy.fftpack 

fft_val=1

arx=[1,2,3,4,5,6,10,12,15,20]
ary=[0,0,0,0,0,0,0,0,0,0]
id=0
for f in arx:
  cam = cv2.VideoCapture("./sample5.mp4")
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
  if fft_val:
    signalFFT = scipy.fftpack.fft(array)
    signalPSD = np.abs(signalFFT) ** 2
    fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
    fi=fftFreq>1
    temp=fftFreq[fi]
    idx=np.argmax(10*np.log10(signalPSD[fi]))
    ary[id]=60*temp[idx]
    id=id+1
  else:
    times = np.arange(len(array))/fps
    b, a = scipy.signal.butter(1, 0.075)
    filtered = scipy.signal.filtfilt(b, a, array)
    peak_indices = signal.find_peaks(filtered)[0]
    peak_count = len(peak_indices)
    ary[id]=(peak_count)*fps*60/len(array)
    id=id+1
plt.figurefigsize = (8, 4);
px=[]
for i in arx:
  px.append(fps_ac/i)

plt.plot(px,ary)
plt.xlabel('Fps')
plt.ylabel('Pulse [Bpm]')
plt.show()

