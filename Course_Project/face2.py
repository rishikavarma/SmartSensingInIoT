import cv2
from PIL import Image
import dlib
import os
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
i=0

# In the following boolean values, if 1 of them is made true, then script outputs a file having pixel values corresponding to magnified area of that part of the face. If all of them are false then pixel values are that of the whole face without the surroundings.
forehead=False
nose=False
cheek1=False
cheek2=False
chin=False
lips=False
pa="video1.mp4"
cam = cv2.VideoCapture(pa) 
if(cam.isOpened() == False):
	print("Error: Couldn't open Video")
while(True): 
	ret,img = cam.read() 
	if ret: 
		gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
		faces = detector(gray) 
		for face in faces:
		    x1 = face.left() 
		    y1 = face.top()-10
		    x2 = face.right() 
		    y2 = face.bottom() 
		    landmarks = predictor(image=gray, box=face)
		    
		    roi_color = img[y1:y2, x1:x2]
		    img1=img
		    if forehead:
		    	xa = landmarks.part(19).x
		    	ya = landmarks.part(19).y
		    	xb = landmarks.part(24).x
		    	yb = landmarks.part(24).y
		    	cv2.rectangle(img=img1, pt1=(xa, min(ya,yb)-5), pt2=(xb, y1), color=(0, 255, 0), thickness=4)# show the image
		    	roi_color=img[y1:min(ya,yb)-5,xa:xb] 
		    elif nose:
		    	ya=landmarks.part(27).y
		    	yb=landmarks.part(33).y
		    	xa=landmarks.part(31).x
		    	xb=landmarks.part(35).x
		    	cv2.rectangle(img=img1, pt1=(xa, ya), pt2=(xb, yb), color=(0, 255, 0), thickness=4)
		    	roi_color=img[ya:yb,xa:xb] 
		    elif cheek1:
		    	ya=landmarks.part(46).y+5
		    	yb=landmarks.part(33).y
		    	xa=landmarks.part(35).x
		    	xb=landmarks.part(14).x		    	
		    	cv2.rectangle(img=img1, pt1=(xa, ya), pt2=(xb, yb), color=(0, 255, 0), thickness=4)
		    	roi_color=img[ya:yb,xa:xb]
		    elif cheek2:
		    	ya=landmarks.part(40).y+5
		    	yb=landmarks.part(33).y
		    	xa=landmarks.part(2).x
		    	xb=landmarks.part(31).x
		    	cv2.rectangle(img=img1, pt1=(xa, ya), pt2=(xb, yb), color=(0, 255, 0), thickness=4)
		    	roi_color=img[ya:yb,xa:xb]
		    elif lips:
		    	ya=int((landmarks.part(50).y+landmarks.part(52).y)/2)
		    	yb=landmarks.part(57).y
		    	xa=landmarks.part(48).x
		    	xb=landmarks.part(54).x
		    	cv2.rectangle(img=img1, pt1=(xa, ya), pt2=(xb, yb), color=(0, 255, 0), thickness=4)
		    	roi_color=img[ya:yb,xa:xb]
		    elif chin:
		    	ya=landmarks.part(57).y+3
		    	yb=int((landmarks.part(6).y+landmarks.part(10).y)/2)
		    	xa=landmarks.part(6).x
		    	xb=landmarks.part(10).x
		    	cv2.rectangle(img=img1, pt1=(xa, ya), pt2=(xb, yb), color=(0, 255, 0), thickness=4)
		    	roi_color=img[ya:yb,xa:xb]
		    else:
		    	cv2.rectangle(img=img1, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
		    cv2.imwrite(str(i)+'_facef.jpg', img1)
		    if len(roi_color):
			    cv2.imwrite(str(i)+'_facep.jpg', roi_color)
			    im = Image.open(str(i)+'_facep.jpg','r')
			    pix_val = list(im.getdata())
			    f=open('f'+str(i) +'_faces.txt','w')
			    f.write(str(pix_val))
			    f.close()
			    os.remove(str(i)+'_facep.jpg')
			    i=i+1
	else:
		break
print('Number of frames=',end=' ')
print(i)

# References:
# https://www.geeksforgeeks.org/python-os-remove-method/
# https://www.researchgate.net/figure/The-68-landmarks-detected-by-dlib-library-This-image-was-created-by-Brandon-Amos-of-CMU_fig2_329392737
# https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e
# https://github.com/cmusatyalab/openface/issues/404