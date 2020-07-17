import matplotlib.pyplot as plt
import numpy as np


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
