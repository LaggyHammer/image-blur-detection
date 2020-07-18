# image-blur-detection
Detecting blurred images with the help of Fast Fourier Transform.

The module performs numpy FFT on an image or a video-frame (converted to grayscale). The FFT result is then passed 
through the inverse FFT function & the mean magnitude of the spectrum is calculated.

This mean is then compared to a preset threshold to determine if the input is blurry or not.

# Blurred Image Detection
Takes in an image as an input and determines if its blurred or not.
## Usage
```commandline
python detect_blur_image.py --image test.png
python detect_blur_image.py --image test.png --threshold 40
```
Test mode at a custom threshold:
```commandline
python detect_blur_image.py --image test.png --threshold 40 --test 1
```

# Blurred Video Detection
Takes a video or the web-cam video feed as an input and detects if its blurred or not.
## Usage
```commandline
python detect_blur_video.py 
python detect_blur_video.py --threshold 40
python detect_blur_video.py --video test.mp4
python detect_blur_video.py --video test.mp4 --threshold 40
```
