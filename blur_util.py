import matplotlib.pyplot as plt
import numpy as np
import cv2


def blur_detector(image, size=60, threshold=10, draw=False):
    height, width = image.shape
    center_x, center_y = int(height / 2), int(width / 2)

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    if draw:
        mag = 20 * np.log(np.abs(fft_shift))

        fig, ax = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(mag, cmap='gray')
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.show()

    fft_shift[center_y - size: center_y + size, center_x - size: center_x + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    inv_fft = np.fft.ifft2(fft_shift)

    mag = 20 * np.log(np.abs(inv_fft))
    mean = np.mean(mag)

    return mean, mean <= threshold


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
