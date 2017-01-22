import numpy as np
from sklearn import svm
import cPickle
import pyaudio
import wave
import audioop
import cv2
from time import sleep

#################################################################
##                          AUDIO                              ##
#################################################################

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 32000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
VECTOR_SIZE = 8

#################################################################
##                          VIDEO                              ##
#################################################################

cap = cv2.VideoCapture(0)
cascPath = 'haarcascade_frontalface_default.xml'
windowName = 'A.S.H.'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
#cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)          
#cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

squareColor = (0, 255, 0)
squareColor2 = (255, 0, 0)

#################################################################
##                          AUDIO                              ##
#################################################################

#################################################################
# Create train set
#original = []
#with open('noise.txt', 'r') as f:
#    for line in f:
#        original.append(int(line))
#
#N = len(original)
#
#diff = []
#for x in range(2, N):
#	diff.append(abs(original[x] - original[x-1]))
#
#X_train = []
#for x in range(VECTOR_SIZE, N-1):
#	temp_vector = []
#	for y in range(x - VECTOR_SIZE, x):
#		temp_vector.append(diff[y])
#	temp_vector.append(original[x])
#	X_train.append(temp_vector)

#################################################################
# Create test set
#original = []
#with open('noise_test.txt', 'r') as f:
#    for line in f:
#        original.append(int(line))
#
#N = len(original)
#
#diff = []
#for x in range(2, N):
#  diff.append(abs(original[x] - original[x-1]))
#
#X_test = []
#for x in range(VECTOR_SIZE, N-1):
#  temp_vector = []
#  for y in range(x - VECTOR_SIZE, x):
#    temp_vector.append(diff[y])
#  temp_vector.append(original[x])
#  X_test.append(temp_vector)

#################################################################
# Fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

# Train and save SVM
#clf.fit(X_train)
#with open('noise.pkl', 'wb') as fid:
#    cPickle.dump(clf, fid)  
#quit()

# Load and save SVM
with open('silence.pkl', 'rb') as fid:
	clf = cPickle.load(fid)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
				input_device_index = 3,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)

print("* recording")

frames = []

#################################################################
##                          VIDEO                              ##
#################################################################
while(cap.isOpened()):
#################################################################
##                          AUDIO                              ##
#################################################################

#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	currentData = []
	data = stream.read(CHUNK)
	lastRms = audioop.rms(data, 2)
	#print lastRms
	frames.append(data)
	for j in range(0, 8):
		data = stream.read(CHUNK)
		rms = audioop.rms(data, 2)
		#print rms
		frames.append(rms)
		currentData.append(abs(rms - lastRms))
	data = stream.read(CHUNK)
	rms = audioop.rms(data, 2)
	frames.append(rms)
	currentData.append(rms)
	realData = []
	realData.append(currentData)

	print "\n"
	y_pred_current = clf.predict(realData)
	n_error_test = y_pred_current[y_pred_current == -1].size
	print 'Predicted \"' + str(bool((y_pred_current[0] + 1)/2)) + '\"\t\tCurrent RMS: ' + str(rms)
	sleep(0.001)


#################################################################
##                          VIDEO                              ##
#################################################################
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	##############################################

	faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.9,
	minNeighbors=3,
	minSize=(10, 10),
	flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
	#print("Found {0} faces!".format(len(faces)))

#    # Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        cv2.rectangle(frame, (x, y), (x+w, y+h), squareColor2, 2)
#
#    cv2.imshow(windowName, frame)
#    #cv2.imshow('frame',gray)
#    c = cv2.waitKey(1) 
#    if c & 0xFF == ord('q'):
#        break
#    if c == 27:
#        break

#################################################################
#text_file = open("Output.txt", "w")
#for i in range(len(y_pred_test)):
#  text_file.write(str(int(y_pred_test[i])) + "\n")
#text_file.close()

print("* done recording")

#################################################################
# Stop audio stream
stream.stop_stream()
stream.close()
p.terminate()

#################################################################
# Save audio stream
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

#################################################################
##                          VIDEO                              ##
#################################################################
# Release OpenCV attributes
cap.release()
cv2.destroyAllWindows()
