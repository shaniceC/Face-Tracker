import cv2
import numpy as np


def show_detection(frame, face):
    """ Draw a rectangle over the detected face """
    x, y, w, h = face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 3)

    return frame


def detect_face(cascade, frame):
    """ Detect a face in a frame """

    # perform the detection
    faces = cascade.detectMultiScale(frame)

    if len(faces) > 0:
        return faces[0]
    else:
        return None


if __name__ == '__main__':
    # load cascade classifiers:
    haar_face_alt_2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
    haar_face_default = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5

    # check if the webcam is opened correctly
    if cap.isOpened() == False:
        raise IOError("Cannot open webcam")

    while True:
        # read the current frame from the webcam and convert to grayscale
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

        # perform the detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_face(haar_face_default, gray)
        if face is not None:
            # draw face detection
            img_face = show_detection(frame, face)
            cv2.imshow('Face Detector', img_face)

        else:
            cv2.imshow('Face Detector', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()