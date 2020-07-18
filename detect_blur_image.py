import argparse
import numpy as np
import imutils
import cv2
from blur_util import blur_detector


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', type=str, required=True,
                    help='image to detect the blur in')
    ap.add_argument('-t', '--threshold', type=int, default=20,
                    help='threshold for a blurry image')
    ap.add_argument('-v', '--vis', type=int, default=-1,
                    help='minimum confidence for a detection')
    ap.add_argument('-d', '--test', type=int, default=-1,
                    help='subsequently blur the image')
    arguments = vars(ap.parse_args())

    return arguments


def main(image, threshold, vis, test):
    orig_image = cv2.imread(image)
    orig_image = imutils.resize(orig_image, width=500)
    gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

    mean, blur = blur_detector(gray_image, threshold=threshold, draw=vis > 0)

    image = np.dstack([gray_image] * 3)
    color = (0, 0, 255) if blur else (0, 255, 0)
    info = f'Blurry ({mean})' if blur else f'Not Blurry ({mean})'
    cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Blur Detection", image)
    cv2.waitKey(0)

    if test > 0:
        test_detection(image_in=gray_image, threshold=threshold, vis=vis)


def test_detection(image_in, threshold, vis):

    for radius in range(1, 30, 2):
        image = image_in.copy()

        if radius > 0:
            image = cv2.GaussianBlur(image, (radius, radius), 0)
            print(radius)
            mean, blur = blur_detector(image, threshold=threshold, draw=vis > 0)

            image = np.dstack([image] * 3)
            color = (0, 0, 255) if blur else (0, 255, 0)
            info = f'Blurry ({mean})' if blur else f'Not Blurry ({mean})'
            cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            print(f'[INFO] Kernel: {radius}, Result: {info}')

            cv2.imshow("Blur Testing", image)
            cv2.waitKey(0)


if __name__ == '__main__':

    args = get_arguments()

    main(image=args['image'], threshold=args['threshold'], vis=args['vis'], test=args['test'])
