import numpy as np
from sklearn import svm
import cPickle
import pyaudio
import wave
import audioop
import cv2
from time import sleep
import threading

#################################################################
##                          AUDIO                              ##
#################################################################
class AudioRecord():
	def __init__(self):
		self.open = True
		self.CHUNK = 512
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = 2
		self.RATE = 32000
		self.RECORD_SECONDS = 5
		self.WAVE_OUTPUT_FILENAME = "output.wav"
		self.VECTOR_SIZE = 8
		#self.createTrainSet()
		#self.createTestSet()
		# Fit the model
		self.clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

		# Train and save SVM
		#self.clf.fit(self.X_train)
		#with open('noise.pkl', 'wb') as fid:
		#    cPickle.dump(self.clf, fid)  
		#quit()

		# Load and save SVM
		with open('silence.pkl', 'rb') as fid:
			self.clf = cPickle.load(fid)

		self.p = pyaudio.PyAudio()

		self.stream = self.p.open(format=self.FORMAT,
						#input_device_index = 3,
						channels=self.CHANNELS,
						rate=self.RATE,
						input=True,
						frames_per_buffer=self.CHUNK)

		print("* recording")

		self.frames = []


	def record(self):
		while(self.open==True):
			self.currentData = []
			self.data = self.stream.read(self.CHUNK)
			self.lastRms = audioop.rms(self.data, 2)
			#print self.lastRms
			self.frames.append(self.data)
			for j in range(0, 8):
				self.data = self.stream.read(self.CHUNK)
				self.rms = audioop.rms(self.data, 2)
				#print self.rms
				self.frames.append(self.rms)
				self.currentData.append(abs(self.rms - self.lastRms))
			self.data = self.stream.read(self.CHUNK)
			self.rms = audioop.rms(self.data, 2)
			self.frames.append(self.rms)
			self.currentData.append(self.rms)
			self.realData = []
			self.realData.append(self.currentData)

			print "\n"
			self.y_pred_current = self.clf.predict(self.realData)
			self.n_error_test = self.y_pred_current[self.y_pred_current == -1].size
			print 'Predicted \"' + str(bool((self.y_pred_current[0] + 1)/2)) + '\"\t\tCurrent RMS: ' + str(self.rms)

	def stop(self):
		self.open=False
		#################################################################
		#text_file = open("Output.txt", "w")
		#for i in range(len(y_pred_test)):
		#  text_file.write(str(int(y_pred_test[i])) + "\n")
		#text_file.close()

		print("* done recording")

		#################################################################
		# Stop audio stream
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()

		#################################################################
		## Save audio stream
		#wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
		#wf.setnchannels(self.CHANNELS)
		#wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
		#wf.setframerate(self.RATE)
		#wf.writeframes(b''.join(self.frames))
		#wf.close()

	# Launches the video recording function using a thread			
	def start(self):
		video_thread = threading.Thread(target=self.record)
		video_thread.start()

	def createTrainSet(self):
		self.original = []
		with open('noise.txt', 'r') as f:
		    for line in f:
		        self.original.append(int(line))
		
		self.N = len(self.original)
		
		self.diff = []
		for x in range(2, self.N):
			self.diff.append(abs(self.original[x] - self.original[x-1]))
		
		self.X_train = []
		for x in range(VECTOR_SIZE, N-1):
			temp_vector = []
			for y in range(x - VECTOR_SIZE, x):
				temp_vector.append(diff[y])
			temp_vector.append(self.original[x])
			self.X_train.append(temp_vector)

	def createTestSet(self):
		self.original = []
		with open('noise_test.txt', 'r') as f:
		    for line in f:
		        self.original.append(int(line))
		
		self.N = len(self.original)
		
		self.diff = []
		for x in range(2, N):
			self.diff.append(abs(self.original[x] - self.original[x-1]))
		
		self.X_test = []
		for x in range(VECTOR_SIZE, N-1):
			temp_vector = []
			for y in range(x - VECTOR_SIZE, x):
				temp_vector.append(self.diff[y])
				temp_vector.append(self.original[x])
			self.X_test.append(temp_vector)


#################################################################
##                          VIDEO                              ##
#################################################################
class VideoRecord():
	def __init__(self):
		self.open = True
		self.cap = cv2.VideoCapture(0)
		self.cascPath = 'haarcascade_frontalface_default.xml'
		self.windowName = 'A.S.H.'

		# Create the haar cascade
		self.faceCascade = cv2.CascadeClassifier(self.cascPath)
		#cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)          
		#cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

		self.squareColor = (0, 255, 0)
		self.squareColor2 = (255, 0, 0)

	def record(self):
		while(self.cap.isOpened() and self.open == True):
			ret, self.frame = self.cap.read()
			if (ret==True):
				gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

				##############################################

				faces = self.faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.9,
				minNeighbors=3,
				minSize=(10, 10),
				flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
				#print("Found {0} faces!".format(len(faces)))

				# Draw a rectangle around the faces
				for (x, y, w, h) in faces:
					cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.squareColor2, 2)

				cv2.imshow(self.windowName, self.frame)
				#cv2.imshow('frame',gray)
				c = cv2.waitKey(1) 
				if c & 0xFF == ord('q'):
					break
				if c == 27:
					break
				sleep(0.16)
		self.open=False
			
	# Release OpenCV attributes
	def stop(self):
		self.open=False
		sleep(0.5)
		self.cap.release()
		cv2.destroyAllWindows()
	
	# Launches the audio recording function using a thread
	def start(self):
		audio_thread = threading.Thread(target=self.record)
		audio_thread.start()


#################################################################
##                          THREAD                             ##
#################################################################
def start_ASH():
	global video_thread
	global audio_thread
	
	video_thread = VideoRecord()
	audio_thread = AudioRecord()

	video_thread.start()
	audio_thread.start()

def stop_ASH():
	video_thread.stop()
	audio_thread.stop()

	# Makes sure the threads have finished
	while threading.active_count() > 1:
		sleep(1)
	
if __name__== "__main__":	
	start_ASH()
	
	while video_thread.open:
		sleep(0.5)
	
	stop_ASH()
	print "Done"

