import dlib
import face_recognition
import os
import cv2

KNOWN_FACES_DIR =  "known_faces" # training data
TOLERANCE = 0.7
FRAME_THICKNESS = 3 # rectangle thickness
FONT_THICKNESS = 2 # font thickness
MODEL = "cnn" # convolutional
video = cv2.VideoCapture(0) # use webcan as input (source of test images)
print("loading known faces")
known_faces = [] # store known faces here
known_names = [] # store name of them here

print(dlib.DLIB_USE_CUDA) # ture, if cuda is enabled.

### TRAIN THE CNN MODEL ON KNOWN FACES #############################
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
####################################################################


print("processing unknown faces")
while True:
    ret, image = video.read() # obtain new image from webcan
    locations = face_recognition.face_locations(image, model=MODEL) # find face
    encodings = face_recognition.face_encodings(image, locations) # extrac face features
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results: # if there is a known face
            match = known_names[results.index(True)] # who that person is
            print(f"Match found: {match}")

            ### DRAW RECTANGLE AND #####################################
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0] # green 
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            #############################################################

    cv2.imshow("webcam images", image) # make window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()