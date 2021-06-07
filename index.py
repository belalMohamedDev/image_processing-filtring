from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon
import cv2
import numpy as np
from scipy.ndimage import generic_filter
from skimage import exposure
from scipy import fftpack 
from skimage.restoration import estimate_sigma
from pylab import *
from matplotlib.colors import LogNorm
  
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)
        self.btn1.clicked.connect(lambda: self.button1_click())
        self.btn3.clicked.connect(lambda: self.show_gray_image())
        self.btn4.clicked.connect(lambda: self.addConstToEachPixel())
        self.btn5.clicked.connect(lambda: self.addConstToBlue())
        self.btn6.clicked.connect(lambda: self.Swapping_image())
        self.btn7.clicked.connect(lambda: self.Eliminating_color())
        self.btn8.clicked.connect(lambda: self.look_up_table())
        self.btn9.clicked.connect(lambda: self.Multiplication_Function())
        self.btn10.clicked.connect(lambda: self.Gamma_Correction_Function())
        self.btn11.clicked.connect(lambda: self.Stretching_function())
        self.btn12.clicked.connect(lambda: self.equ())
        self.btn13.clicked.connect(lambda: self.matching())
        self.btn14.clicked.connect(lambda: self.Neighbourhood())
        self.btn15.clicked.connect(lambda: self.Smoothing_Average())
        self.btn16.clicked.connect(lambda: self.Smoothing_Weighted())
        self.btn17.clicked.connect(lambda: self.Smoothing_median())
        self.btn18.clicked.connect(lambda: self.Laplacian_Filter())
        self.btn19.clicked.connect(lambda: self.outlier())
        self.btn20.clicked.connect(lambda: self.image_Averaging())
        self.btn21.clicked.connect(lambda: self.Adaptive_filtering())
        self.btn22.clicked.connect(lambda: self.notch_filtering())
        self.show()
        

    def url_input(self):
        self.url_input1=self.lineEdit.text()
        self.url_input0=self.lineEdit1.text()
       

    def button1_click(self):
        input=self.url_input()
        self.btn2.setStyleSheet("border-image : url({0});".format(self.url_input1))
        self.btn2_1.setStyleSheet("border-image : url({0});".format(self.url_input0))
    
    def gray_image(self):
        image=self.url_input()
        self.image1 = cv2.imread(self.url_input1,0)
        self.image2 = cv2.imread(self.url_input0,1)
        

    def show_gray_image(self):
        gray=self.gray_image()
        cv2.imshow('gray',self.image1)
        cv2.waitKey(0)
        

    def addConstToEachPixel(self):
        gray_=self.gray_image()
        rows = self.image1.shape[0]
        cols = self.image1.shape[1]
        for i in range(rows):
            for j in range(cols):
                pixel = self.image1[i,j]
                newPixel = pixel + 128
                if newPixel > 255 :
                    self.image1[i,j] = 255
                elif newPixel < 0:
                    self.image1[i ,j] = 0
                else:
                    self.image1[i,j] = newPixel

        cv2.imshow('gray_', self.image1)
        cv2.waitKey(0)


    def color_image(self):
        imagee=self.url_input()
        image = cv2.imread(self.url_input1,1)
        self.blue = image[:,:,0]
        self.green = image[:,:,1]
        self.red = image[:,:,2]
        return image
    

    def addConstToBlue(self):
        color_=self.color_image()
        blue_row , blue_col = self.blue.shape

        for i in range(blue_row):
            for j in range(blue_col):
                pixel = self.blue[i,j]
                newPixel = pixel + 128
                if newPixel > 255:
                    self.blue[i, j] = 255
                elif newPixel < 0:
                    self.blue[i, j] = 0
                else:
                    self.blue[i, j] = newPixel
        
        cv2.imshow('color_', color_)
        cv2.waitKey(0)
        
    

      

    def Swapping_image(self):
        color_=self.color_image()
        color_[:,:,0] = self.green
        cv2.imshow('Swapping_image', color_)
        cv2.waitKey(0)
            
            
    def Eliminating_color(self):
        color_=self.color_image()
        color_[:,:,1] = np.zeros([color_.shape[0], color_.shape[1]])
        cv2.imshow('Girl', color_)
        cv2.waitKey(0)

    def look_up_table(self):
        gray=self.gray_image()
        newImage = cv2.applyColorMap(self.image1, cv2.COLORMAP_INFERNO)
        cv2.imshow('Low contrast image after applying lut ',newImage)
        cv2.waitKey(0)
       


    def Multiplication_Function(self):
        low_cont_image=self.gray_image()
        rows,cols =self.image1.shape
        for i in range(rows):
            for j in range(cols):
                pixel = self.image1[i,j]
                newPixel = pixel * 0.3
                if newPixel > 255:
                    self.image1[i, j] = 255
                elif newPixel < 0:
                    self.image1[i, j] = 0
                else:
                    self.image1[i, j] = newPixel

        cv2.imshow('Low contrast image after applying mult function ',self.image1)
        cv2.waitKey(0)


    def Gamma_Correction_Function(self):
        low_cont_image=self.gray_image()
        rows,cols = self.image1.shape
        for i in range(rows):
            for j in range(cols):
                pixel = self.image1[i,j]
                newPixel = pixel ** 1.1
                if newPixel > 255:
                    self.image1[i, j] = 255
                elif newPixel < 0:
                    self.image1[i, j] = 0
                else:
                    self.image1[i, j] = newPixel

        cv2.imshow('Low contrast image after applying Gamma function ',self.image1)
        cv2.waitKey(0)


    def Stretching_function(self):
        low_pixel = 10
        high_pixel = 250
        low_cont_image=self.gray_image()
        rows,cols = self.image1.shape
        for i in range(rows):
            for j in range(cols):
                pixel = self.image1[i, j]
                if pixel <= low_pixel:
                    self.image1[i,j]=0
                elif high_pixel <= pixel:
                    self.image1[i,j]=255
                elif pixel >= low_pixel and pixel <= high_pixel:
                    self.image1[i,j] = 255*((pixel - low_pixel)/(high_pixel-low_pixel))


        cv2.imshow('Low contrast image after applying Stretching function ',self.image1)
        cv2.waitKey(0)


    def equ(self):
        low_cont_image=self.gray_image()
        eq = cv2.equalizeHist(self.image1)
        cv2.imshow('Low contrast image after applying Histogram Equalization ',eq)

    def matching(self):
        gray=self.gray_image()
        src = self.image2
        ref = self.color_image()
        matched = exposure.match_histograms(src,ref,multichannel=True)

        cv2.imshow('Low contrast image after applying Histogram Matching ',matched)
        cv2.waitKey(0)

    def Neighbourhood(self):
        gray=self.gray_image()
        newImage = generic_filter(self.image1,max,size=(3,3))
        cv2.imshow('noisy image after applying Neighbourhood operation',newImage)
        cv2.waitKey(0)

    def Smoothing_Average(self):
        gray=self.gray_image()
        newImage = cv2.blur(self.image1,(9,9))
        cv2.imshow('noisy image after applying Smoothing Average Operation',newImage)
        cv2.waitKey(0)


    def Smoothing_Weighted(self): 
        gray=self.gray_image()
        newImage = cv2.GaussianBlur(self.image1,(3,3),sigmaX=0,sigmaY=0)
        cv2.imshow('noisy image after applying Smoothing Weighted Average Operation',newImage)
        cv2.waitKey(0)


    def Smoothing_median(self):
        gray=self.gray_image()
        newImage = cv2.medianBlur(self.image1,9)
        cv2.imshow('noisy image after applying Smoothing median Operation',newImage)
        cv2.waitKey(0)

    def Laplacian_Filter(self):
        gray=self.gray_image()
        newImage = cv2.Laplacian(self.image1,cv2.CV_16S,(3,3))
        cv2.imshow('noisy image after applying Laplacian Filter',newImage)
        cv2.waitKey(0)
    
    def outlier(self):
        gray=self.gray_image()
        D = 0.2
        size = int(input('please enter the size of mask: '))
        step = int((size-1)/2)

        rows,cols = self.image1.shape
        for i in range(0,rows):
            for j in range(0,cols):
                p = self.image1[i,j]
                neighbor = []
                for x in range(i-step , i+step+1):
                    for y in range(j-step , j+step+1):

                        if x < 0 or y < 0 or x > rows-1 or y > cols-1:
                            pass
                        else:
                            neighbor.append(self.image1[x, y])

                print(neighbor)
                m = mean(neighbor)
                print(m)
                if abs(p-m) > D:
                    self.image1[i,j]= m


        cv2.imshow('Image after applying outlier method',self.image1)
        cv2.waitKey(0)



    def image_Averaging(self):
        gray=self.gray_image()
        sum=0
        for i in range(0,10):
            sum += self.image1
        out_img = sum/10
        cv2.imshow('Image after applying image Averaging',out_img)
        cv2.waitKey(0)

    
    def Adaptive_filtering(self):
        gray=self.gray_image()
        size = int(input('please enter the size of mask: '))
        step = int((size-1)/2)

        sd = estimate_sigma(self.image1, multichannel=True, average_sigmas=True)
        nv = sd**2
        rows,cols = self.image1.shape
        for i in range(0,rows):
            for j in range(0,cols):
                p = self.image1[i,j]
                neighbor = []
                for x in range(i-step , i+step+1):
                    for y in range(j-step , j+step+1):
                        if x < 0 or y < 0 or x > rows-1 or y > cols-1:
                            neighbor.append(0)
                        else:
                            neighbor.append(self.image1[x, y])


                m = mean(neighbor)
                v = np.var(neighbor)
                self.image1[i, j] = m + (v/(v+nv))*(p-m)

        cv2.imshow('Image after applying Adaptive filtering',self.image1)
        cv2.waitKey(0)


    def notch_filtering(self):
        gray=self.gray_image()
        img = self.image1
        cv2.imshow('Image after applying Adaptive filtering',img)
        cv2.waitKey(0)

        im_fft = fftpack.fft2(img) #Fourier Form Transforms
        plt.imshow(np.abs(im_fft),norm=LogNorm(100)) #lofnorm ÎÇÕÉ ÈÇáÇáæÇä

        plt.show()



        


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

 

