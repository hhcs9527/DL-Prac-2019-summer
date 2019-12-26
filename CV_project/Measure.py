from PIL import Image
import numpy 
import math
import cv2
from skimage.measure import compare_ssim


def Measurement(img1, img2):
    original = numpy.squeeze(img1.cpu().detach().numpy(), axis=0)
    contrast = numpy.squeeze(img2.cpu().detach().numpy(), axis=0) 

    mse = numpy.mean( (original - contrast) ** 2 )
    if mse == 0:
        PSNR =  100
    else:
        PIXEL_MAX = 255.0
        PSNR =  10 * math.log10(PIXEL_MAX / math.sqrt(mse))

    original = original.reshape(256,256,3)
    contrast = contrast.reshape(256,256,3)


    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    
    SSIM = compare_ssim(original, contrast , gaussian_weights=True)
    return PSNR, SSIM

if __name__ == '__main__':
    img1 = '/home/hh/Desktop/CV_project/results/cityp2p_e25/test_latest/images/137_real_B.png'
    img2 = '/home/hh/Desktop/CV_project/results/cityp2p_e25/test_latest/images/137_fake_B.png'
    PSNR, SSIM = Measurement(img1, img2)
    print('SSIM ', SSIM)
    print('PSNR ', PSNR)