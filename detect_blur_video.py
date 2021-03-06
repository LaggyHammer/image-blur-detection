# coding: utf-8
# =====================================================================
#  Filename:    detect_blur_video.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Determines if a video/video-stream is blurred or not.
#
#  Usage: python detect_blur_video.py
#         or
#         python detect_blur_video.py --threshold 40
#         or
#         python detect_blur_video.py --video test.mp4
#         or
#         python detect_blur_video.py --video test.mp4 --threshold 40
#
#  Note: Requires opencv 3.4.2 or later
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import argparse
import cv2
import imutils
import time
from imutils.video import FPS
from blur_util import blur_detector


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', type=str,
                    help='video stream to detect the blur in')
    ap.add_argument('-t', '--threshold', type=int, default=20,
                    help='threshold for a blurry image')
    arguments = vars(ap.parse_args())

    return arguments


def main(video, threshold):
    print('[INFO] Starting video stream...')
    if not video:
        # start web-cam feed
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)

    else:
        # start video stream
        vs = cv2.VideoCapture(video)

    fps = FPS().start()

    # main loop
    while True:

        grabbed, frame = vs.read()

        if frame is None:
            break

        # resize the frame & convert to gray color space
        frame = imutils.resize(frame, width=500)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect blur
        mean, blur = blur_detector(gray_frame, threshold=threshold)

        # label & display result
        color = (0, 0, 255) if blur else (0, 255, 0)
        info = f'Blurry ({round(mean, 2)})' if blur else f'Not Blurry ({round(mean, 2)})'
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Blur Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # exit if 'q' is pressed
        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print(f'[INFO] Elapsed time: {round(fps.elapsed(), 2)}')
    print(f'[INFO] approximate FPS: {round(fps.fps(), 2)}')

    # release endpoint(s) & cleanup
    cv2.destroyAllWindows()
    vs.release()


if __name__ == '__main__':

    args = get_arguments()

    main(video=args.get('video', False), threshold=args['threshold'])
