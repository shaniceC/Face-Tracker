import cv2
import dlib
import numpy as np

# define the landmarks
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))


# initialize the shape predictor
p = "landmarks/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)


def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """ Draw the shape of a facial landmark using the points """
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)


def draw_shape_lines(np_shape, image):
    """ Draw the outline of the face using the landmarks """
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)

    return image


def detect_landmarks(frame, face):
    """ Detect landmarks of a face """

    # find the landmarks for the face
    # get the shape using the predictor
    face_rect = to_dlib_rect(face)
    shape = predictor(frame, face_rect)

    # convert the shape to numpy array
    shape = shape_to_np(shape)

    return shape


def to_dlib_rect(face):
    x, y, w, h = face
    rect = dlib.rectangle(x, y, x+w, y+h)
    return rect


def shape_to_np(dlib_shape, dtype='int'):
    """ Converts dlib shape object to numpy array """
    # initialize the list of (x, y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them to a tuple with (x,y) coordinates
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # return the list of (x,y) coordinates
    return coordinates


if __name__ == '__main__':
    from detection import detect_face
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

        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect a face
        face = detect_face(haar_face_default, gray)

        if face is not None:
            # find landmarks
            face_shape = detect_landmarks(gray, face)
            # draw landmark detection
            img_landmarks = draw_shape_lines(face_shape, frame)
            cv2.imshow('Landmark Detector', img_landmarks)

        else:
            cv2.imshow('Landmark Detector', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()









