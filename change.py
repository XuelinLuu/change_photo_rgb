import cv2
import numpy as np
import os


def cvtBackground(path, color):
    """
        功能：给证件照更换背景色（常用背景色红、白、蓝）
        输入参数：path:照片路径
                color:背景色 <格式[B,G,R]>
    """

    im = cv2.imread(path)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(im_hsv, np.array([im_hsv[0, 0, 0] - 5, 100, 100]),
                       np.array([im_hsv[0, 0, 0] + 5, 255, 255]))
    mask1 = mask
    img_median = cv2.medianBlur(mask, 5)
    mask = img_median
    mask_inv = cv2.bitwise_not(mask)
    img1 = cv2.bitwise_and(im, im, mask=mask_inv)
    bg = im.copy()
    rows, cols, channels = im.shape
    bg[:rows, :cols, :] = color
    img2 = cv2.bitwise_and(bg, bg, mask=mask)
    img = cv2.add(img1, img2)
    image = {'im': im, 'im_hsv': im_hsv, 'mask': mask1, 'img': img, 'img_median': img_median}
    cv2.startWindowThread()
    for key in image:
        cv2.namedWindow(key)
        cv2.imshow(key, image[key])
        cv2.imwrite("your path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img
# test
if __name__ == '__main__':
    img = cvtBackground('your path', [255, 255, 255])
