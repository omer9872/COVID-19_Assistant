import os
import serial
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
import numpy as np
import time
from smbus2 import SMBus
from mlx90614 import MLX90614
import serial

# initializing serialport
# Arduino used to use lights
# and we are going to make serial read-write operations to handle this.
serialPort = serial.Serial('/dev/ttyUSB0',9600)

# here is our model name, cascade name and image names in array.
model_path = "maskdetection011721.h5"
face_cascade_path = "face_cascade.xml"
res_images_p = "res_images"
res_mask = ['mask_ok.jpg', 'mask_no.jpg']
res_temp = ['temp_ok.jpg', 'temp_no.jpg']

# load keras model and face cascade classifier.
model = tf.keras.models.load_model(model_path)
face_clsfr = cv.CascadeClassifier(face_cascade_path)

# this section is kind of a preprocessing to show res_images on screen 
masked_ok = cv.imread(os.path.join(res_images_p, res_mask[0]), -1) 
masked_no = cv.imread(os.path.join(res_images_p, res_mask[1]), -1) 
masked_ok_rsz = cv.resize(masked_ok, (122,110))
masked_no_rsz = cv.resize(masked_no, (122,110))

temp_ok = cv.imread(os.path.join(res_images_p, res_temp[0]), -1) 
temp_no = cv.imread(os.path.join(res_images_p, res_temp[1]), -1) 
temp_ok_rsz = cv.resize(temp_ok, (122,110))
temp_no_rsz = cv.resize(temp_no, (122,110))

# make program fullscreen.
cv.namedWindow("MASK-DETECTOR", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("MASK-DETECTOR", cv.WND_PROP_FULLSCREEN,
                          cv.WINDOW_FULLSCREEN)


#      ENVIRONMENT VARIABLES

# capture video from webcam.
cap = cv.VideoCapture(0)

# defining labels and box colors to use after prediction.
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
label_dict = {0: 'Masked', 1: 'Non Masked'}

# simple function to get text size.
# used to make text horizontally centered.
def getTextSize(text):
    return cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]

# these are text sizes to use in program.
textsize1 = getTextSize('Please Come Closer')
posX1 = int((640-textsize1)/2)

textsize2 = getTextSize('Mask Detected')
posX2 = int((640-textsize2)/2)

textsize3 = getTextSize('Please Wear Mask')
posX3 = int((640-textsize3)/2)

textsize4 = getTextSize('Please Hold Your Wrist To Sensor')
posX4 = int((640-textsize4)/2)

textsize5 = getTextSize('High Temperature (XX.X)')
posX5 = int((640-textsize5)/2)

textsize6 = getTextSize('Normal Temperature (XX.X)')
posX6 = int((640-textsize6)/2)

textsize7 = getTextSize('Please Show QR Code')
posX7 = int((640-textsize7)/2)

textsize8 = getTextSize('QR: 1234567890')
posX8 = int((640-textsize8)/2)

# function that checkes if person is masked.
# takes 14 frames to decide if person is masked or not.
def getIfMasked(cap):
    maskCounter = 0
    frameCounter = 0
    isRunnin = True
    while isRunnin:
        ret, frame = cap.read()
        gry_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gry_img)
        for x, y, w, h in faces:
            area = w* h
            # do not make predictions on small faces because they can be far from panel.
            if area >= 30000 and x>120 and x+w<520:
                cv.rectangle(frame, (x, y), (x+w, y+h), (20, 240, 20), 3)
                face_img = frame[y:y + w, x:x + w]
                # preprocess input.
                resized = cv.resize(face_img, (224, 224))
                img_ary = img_to_array(resized)
                img_ary = np.expand_dims(img_ary, axis=0)
                # predict input data.
                predictions = model.predict_on_batch(img_ary)
                predictions = tf.nn.sigmoid(predictions)
                predictions = tf.where(predictions < 0.5, 0, 1)

                if predictions.numpy()[0][0] == 0:
                    maskCounter += 1

                if frameCounter == 14:
                    isRunnin = False
                else:
                    frameCounter += 1
            else:
                cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
                cv.putText(frame, 'Please Come Closer', (posX1, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


        cv.imshow('MASK-DETECTOR', frame) 
        if cv.waitKey(1) & 0xFF == ord('q'):
            runApp = False
            break

    return maskCounter

# function that indicates person mask status on screen for 5 secs.
# if frames 12/14 masked then person is masked.
def showMaskResult(maskCounter, cap):
    isRunnin = True
    maskTimer = time.time()
    while isRunnin:
        ret, frame = cap.read()
        if time.time() - maskTimer > 5:
            isRunnin = False
        else:
            if maskCounter >= 12:
                frame[:110, :122] = masked_ok_rsz
                cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
                cv.putText(frame, 'Mask Detected', (posX2, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                serialPort.write('d'.encode())
            else:
                frame[:110, :122] = masked_no_rsz
                cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
                cv.putText(frame, 'Please Wear Mask', (posX3, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                serialPort.write('e'.encode())

        cv.imshow('MASK-DETECTOR', frame) 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# function that checkes person's temperature.
# waits till detect any temperature which is greater then 32.0
def getTemp(cap):
    isRunnin = True
    maxTemp = 32.0
    while isRunnin:
        ret, frame = cap.read()
        bus = SMBus(1)
        sensor = MLX90614(bus, address=0x5A)
        temp = float(sensor.get_object_1())
        bus.close()
        if temp >= maxTemp:
            maxTemp = temp
            maxTemp += 4.2

        if maxTemp >= 35.0:
            isRunnin = False

        cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
        cv.putText(frame, 'Please Hold Your Wrist To Sensor', (posX4, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.imshow('MASK-DETECTOR', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    return maxTemp


# function that indicates person's temperature status on screen for 5 secs.
def showTempResult(maxTemp, cap):
    isRunnin = True
    tmpTimer = time.time()
    while isRunnin:
        if time.time() - tmpTimer > 5:
            isRunnin = False
        else:
            ret, frame = cap.read()
            if maxTemp >= 38:
                frame[:110, :122] = temp_no_rsz
                cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
                cv.putText(frame, 'High Temperature ({:.1f})'.format(maxTemp), (posX5, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                serialPort.write('h'.encode())

            else:
                frame[:110, :122] = temp_ok_rsz
                cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
                cv.putText(frame, 'Normal Temperature ({:.1f})'.format(maxTemp), (posX6, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                serialPort.write('g'.encode())

        cv.imshow('MASK-DETECTOR', frame) 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# function that checkes person's QR code.
# in some of countries (i do not know if a lot of them using that but in my case it is necessary)...
# ...QR codes reveals person's health about if they are infected or not so i decided to add this...
# ...feture to the program
def getQR(cap):
    decoded_data = ""
    isRunnin = True
    while isRunnin:
        ret, frame = cap.read()
        qrDecoder = cv.QRCodeDetector()
        data,bbox,rectifiedImage = qrDecoder.detectAndDecode(frame)
        if len(data)>0:
            decoded_data = data
            isRunnin = False
        else:
            cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
            cv.putText(frame, 'Please Show QR Code', (posX7, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv.imshow('MASK-DETECTOR', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    return decoded_data

# function that indicates person's qr code status on screen for 5 secs.
def showQRResult(decoded_data, cap):
    isRunnin = True
    tmpTimer = time.time()
    while isRunnin:
        if time.time() - tmpTimer > 5:
            isRunnin = False
        else:
            ret, frame = cap.read()
            cv.rectangle(frame, (0, 430), (640, 480), (0,0,0), -1)
            cv.putText(frame, 'QR: {}'.format(decoded_data[:10]), (posX8, 460), cv.FONT_HERSHEY_SIMPLEX, 1, (130,130,0), 2)

        cv.imshow('MASK-DETECTOR', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

while True:
    # make all lights blue(which means system is waiting for calculations).
    serialPort.write('c'.encode())
    serialPort.write('f'.encode())
    serialPort.write('j'.encode())

    # program first checks if user masked.    
    maskCounter = getIfMasked(cap)
    showMaskResult(maskCounter, cap)

    # then checks user temperature.    
    maxTemp = getTemp(cap)
    showTempResult(maxTemp, cap)

    # finally checks qr code.   
    decoded_data = getQR(cap)
    showQRResult(decoded_data, cap)

# release video capture and destroy all of program windows after exitting from program.
cap.release()
cv.destroyAllWindows()
