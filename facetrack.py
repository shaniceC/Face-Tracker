import cv2
import dlib
from detection import detect_face, load_cascade
from facemark import to_dlib_rect

haar_cascade = None
tracking_face = False

def draw_text_info(frame):
    """ Draw text information """
    menu_pos_1 = (10, 10)
    menu_pos_2 = (10, 40)

    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "dtecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


def track(frame, face):
    """ Track the face in the frame """
    face_pos = to_dlib_rect(face)
    tracker.start_track(frame, face_pos)


if __name__ == '__main__':
    haar_cascade = load_cascade()
    tracker = dlib.correlation_tracker()

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

        # draw the instructions on the screen
        draw_text_info(frame)

        if tracking_face is False:
            # convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect a face
            face = detect_face(haar_cascade, gray)

            if face is not None:
                # start tracking
                print(track(frame, face))
                tracking_face = True

        if tracking_face is True:
            # update tracking 
            tracker.update(frame)
            # get the position of the tracked object
            pos = tracker.get_position()
            # draw the position
            cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0,255,0), 3)

        c = cv2.waitKey(1)

        if c == ord('1'):
            # stop tracking
            tracking_face = False
        if c == 27:
            # exit
            break

        cv2.imshow('Face Tracker', frame)

    cap.release()
    cv2.destroyAllWindows()