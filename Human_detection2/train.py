from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'C:\Files\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:

    image = cv2.imread(img)
    
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] # C:\Files\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append([label]) # [[1], [0], [0], ...]

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# define model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first", 
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')





























import numpy as np
import argparse
from imutils import face_utils
import imutils
import time
import cv2
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
# init age and gender parameters 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

# load our serialized model from disk
print("[INFO] loading models...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_age.prototxt", 
                        "age_gender_models/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_gender.prototxt", 
                        "age_gender_models/gender_net.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

cap =cv2.VideoCapture(0)


# vs = VideoStream(src=0).start()
time.sleep(1.0)

font = cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
person = 0
cnt = 0
while True:
	# read the next frame from the video stream and resize it
    ret,frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = imutils.resize(frame, width=400)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)

    new_rects = []

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if y<0:
            print("a")
            continue
        new_rects.append((x, y, x + w, y + h))

        face_img = frame[y:y+h, x:x+w].copy()

        blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        if cnt%10 == 0:
            # Predict gender
            gender_net.setInput(blob2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
        
        overlay_text = "%s, %s" % (gender, age)
        cv2.putText(frame, overlay_text ,(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)

    save_frame = frame.copy()       
	# update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(new_rects)

	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        if(objectID>person):
            cv2.imwrite('img/'+str(person)+'.jpg',save_frame)
            person = person+1
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
    cv2.imshow("Frame", frame)
    cnt += 1
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
