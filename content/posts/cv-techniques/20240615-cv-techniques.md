+++
title = 'Thresholding, filtering and morphological operations'
date = 2024-10-25T17:51:55+02:00
tags = ['computer-vision', 'traditional-computer-vision']
[cover]
    image = "https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/thresholding_caption.png?raw=true"
    # caption = "Generated using [OG Image Playground by Vercel](https://og-playground.vercel.app/)"

+++

**Traditional computer vision techniques** involve methods and algorithms that do not rely on deep learning or neural networks. Instead, these approaches are not data-driven and they use classical approaches to process and analyze images. So, in this post, we'll explore **three thresholding techniques!**

# Thresholding

When the task is to distinguish the background from the foreground, thresholding provides a straightforward solution. We will use this image as an example.

![Image content](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/text_image.png?raw=true#center)

This technique segments an image by assigning one value (typically white) to all pixels above a specified threshold and another value (usually black) to the remaining pixels. Thresholding is a simple yet effective method for separating objects from the background, especially when the background is not complicated.

![Thresholding ](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/histogram.png?raw=true#center)

Code of the histogram:
```python
img = cv2.imread("images/document.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.hist(img.reshape(-1), bins=255, range=(0, 255), zorder=3, color="teal")
plt.xlim([0,255])
plt.ylim(plt.ylim())
plt.vlines(130, *plt.ylim(), label='Threshold', linestyles='dotted', color='red', zorder=3)
plt.grid(alpha=0.3, zorder=1)
plt.title("Histogram of the image")
```

## Binary Thresholding

Binary thresholding is the most simple method of thresholding. Each pixel in the image is compared to a threshold value: if the pixel value is higher than the threshold, it is set to the _maximum value_ (white); otherwise, it is set to the _minimum value_ (black).

In [opencv](https://pypi.org/project/opencv-python/), we almost always use the method `cv2.threshold` to apply umbralization over an image. In this case, we are setting the threshold to **130** and the maximum value of the image to **255**. `cv2.THRESH_BINARY` is the _flag_ to indicate which kind of thresholding method we want.

```python
threshold = 130
threshold, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY) # The first element of the response is worthless here.
```

![Histogram binary thresholding](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/binary_histogram.png#center)

In our case, the image should turn into:

![Binary thresholding](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/binary-img.png#center)

## Otsu Thresholding

But... how to find the best threshold? Well, a japanese person named Otsu did a greatjob. **Otsu's thresholding** is an automatic method that calculates the _optimal threshold_ value by minimizing the intra-class variance. This technique is particularly useful for images with bimodal histograms.

![alt text](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/otsu_hist.png#center)

```python
otsu_threshold, otsued_img = cv2.threshold(img ,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

In our case, the image should turn into:

![alt text](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/otsu_image.png#center)

## Adaptive Thresholding

What if we have different lightning conditions? **Adaptive thresholding** is particularly useful in cases where **background intensity vary** across the image. Instead of a single global threshold, this technique calculates different threshold values for different regions of the image.

Adaptive thresholding typically uses either the *mean* or *Gaussian weighted sum of the neighborhood of each pixel* (typical, right?). In OpenCV, we can specify the type of adaptive method with `cv2.ADAPTIVE_THRESH_MEAN_C` or `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`. Additionally, a constant `C` is subtracted from the calculated threshold value to further adjust the results.


```python
adaptive_threshold_img = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
```

In this case, **adaptative thresholding** success thresholding by avoiding background variations.

![Comparison of different images](https://raw.githubusercontent.com/Oriolac/oriolac.github.io/refs/heads/main/content/posts/cv-techniques/imgs/comparison.png#center)

```python
fig, axs = plt.subplots(3, 1, figsize=(6, 10))
for ax in axs:
    ax.axis('off')

axs[0].imshow(img, cmap='gray'); axs[0].set_title("Original image")

ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
axs[1].imshow(thresh, cmap='gray'); axs[1].set_title("Binary thresholding")

thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 20) 
axs[2].imshow(thresh2, cmap='gray'); axs[2].set_title("Adaptive thresholding");
```
