import cv2
import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
from PIL import Image 
import scipy.fftpack 


cam = cv2.VideoCapture("./sample18.mp4") 
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
    	val1=60*temp[idx]
    	times = np.arange(len(array))/fps
    	b, a = scipy.signal.butter(1, 0.06)
    	filtered = scipy.signal.filtfilt(b, a, array)
    	peak_indices = signal.find_peaks(filtered)[0]
    	peak_count = len(peak_indices)
    	val2=((peak_count)*fps*60/j)
    	print(val1,end="	")
    	print(val2)
cam.release() 
cv2.destroyAllWindows() 
print("Final value:")
signalFFT = scipy.fftpack.fft(array)
signalPSD = np.abs(signalFFT) ** 2
fftFreq = scipy.fftpack.fftfreq(len(signalPSD), 1/fps)
fi=fftFreq>1
temp=fftFreq[fi]
idx=np.argmax(10*np.log10(signalPSD[fi]))
val1=(60*temp[idx])

times = np.arange(len(array))/fps
b, a = scipy.signal.butter(1, 0.06)
filtered = scipy.signal.filtfilt(b, a, array)
peak_indices = signal.find_peaks(filtered)[0]
peak_count = len(peak_indices)
val2=((peak_count)*fps*60/len(array))

print(val1,end="	")
print(val2)


# plt.figurefigsize = (8, 4);
# plt.plot(fftFreq[fi], 10*np.log10(signalPSD[fi]));
# plt.xlabel('Frequency [Hz]');
# plt.ylabel('PSD [dB]')
# plt.show()


# # print(spectrum)
# print(freqs)
# idx = np.argmax(np.abs(spectrum))
# print(idx)
# plt.figure(figsize=(10, 4))

# plt.subplot(121)
# plt.plot(times, array)
# plt.title("ECG Signal with Noise")
# plt.margins(0, .05)

# plt.subplot(121)
# plt.plot(times, filtered)
# plt.title("Filtered ECG Signal")
# plt.margins(0, .05)
# times1 = np.arange(len(freqs))/fps
# plt.subplot(122)
# plt.plot(times, freqs)
# plt.title("Weights of frequencies")
# plt.margins(0, .05)


# plt.tight_layout()
# plt.show()
# freq = freqs[idx]
# freq_in_hertz = abs(freq * 60*60/(2*3.14))
# print(freq_in_hertz)
# print(freq)
# threshold = 0.75 * max(abs(spectrum))
# mask = abs(spectrum) > threshold
# peaks = freqs[mask]
# print(peaks)