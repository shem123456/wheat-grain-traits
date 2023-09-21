# 设置一个kernel的类
# 读取一张籽粒的照片，获取籽粒的颜色和形态特征、
# 颜色特征：R,G,B均值和方差，H,S,V均值和方差
# 形态特征：面积、周长、等效圆直径、离心率、圆形度、椭圆度、矩形度
# 纹理特征： 平均亮度、平滑度、三阶矩、一致性、熵

import cv2
import numpy as np
import math
from skimage import morphology

import skimage.feature as feature
import skimage.feature as feature
from skimage import measure

class Kernel():
    """
    一颗籽粒的读取与分析
    """

    def __init__(self, path):
        """初始化属性"""
        self.path = path

    # opencv的图像图像
    def image(self):
        image = cv2.imread(self.path)
        return image

    # 图像分割，二值化，大津算法（thresh_otsu）
    def binary_image(self):
        image = self.image()
        # 转灰度
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh_img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        return image_gray,thresh_img

    def no_background_image(self):
        image = self.image()
        gray,thresh_img = self.binary_image()
        # 去噪
        kernel = np.ones((5,5),np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img,cv2.MORPH_OPEN,kernel)
        # 反向
        mask = cv2.bitwise_not(thresh_img)
        # 变成3通道
        mask1 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        # 相加
        result = cv2.add(image,mask1)
        return result

    # 颜色特征
    def RGB_mean(self):
        img = self.no_background_image()
        R_mean = np.mean(img[:,:,2])
        G_mean = np.mean(img[:,:,1])
        B_mean = np.mean(img[:,:,0])
        return R_mean, G_mean, B_mean
    def HSV_mean(self):
        img = self.no_background_image()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        H_mean = np.mean(img[:,:,0])
        S_mean = np.mean(img[:,:,1])
        V_mean = np.mean(img[:,:,2])
        return H_mean,S_mean,V_mean

    # 形状特征
    def morphology_trait(self):
        image_gray,thresh_img = self.binary_image()
        contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #高级函数的用法，key直接调用函数，取面积最大的轮廓
        cnt = sorted(contours,key=cv2.contourArea)[-1]
        area = cv2.contourArea(cnt)#面积
        length = cv2.arcLength(cnt,True)#周长
        minrectangle = cv2.minAreaRect(cnt)#最小外接矩形
        (x,y),radius = cv2.minEnclosingCircle(cnt)#最小外接圆
        radius = int(radius)*2 #最小外接圆直径
        equi_diameter = int(np.sqrt(4*area/np.pi))#等效圆直径
        (x, y) , (a, b), angle = cv2.fitEllipse(cnt)#椭圆拟合
        if (a > b):
            eccentric = np.sqrt(1.0 - (b / a) ** 2)  # a 为长轴
        else:
            eccentric = np.sqrt(1.0 - (a / b) ** 2)#偏心率,范围为 [0,1]，圆的偏心率为 0 最小，直线的偏心率为 1最大
        compact = length ** 2 / area  # 轮廓的紧致度 (compactness),紧致度是一个无量纲的测度，圆的紧致度最小，为 4 π 4\pi4π，正方形的紧致度 是 16
        rectangle_degree = area / (minrectangle[1][0]*minrectangle[1][1])#矩形度
        roundness = (4 * math.pi * area) / (length * length)#圆形度,圆的圆度为 1 最大，正方形的圆度为 π / 4 \pi / 4π/4
        
        # print("面积：",area)
        # print("周长：",length)
        # print("最小外接圆直径：",radius)
        # print("等效圆直径：",equi_diameter)
        # print("偏心率：",eccentric)
        # print("紧致度：",compact)
        # print("矩形度：",rectangle_degree)
        # print("圆形度：",roundness)
        return area,length,radius,equi_diameter,eccentric,compact,rectangle_degree,roundness

    # 纹理特征
    def texture_trait(self):
        image_gray,thresh_img = self.binary_image()
        graycom = feature.greycomatrix(image_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        # Find the GLCM properties
        contrast = feature.greycoprops(graycom, 'contrast')
        dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
        homogeneity = feature.greycoprops(graycom, 'homogeneity')
        energy = feature.greycoprops(graycom, 'energy')
        correlation = feature.greycoprops(graycom, 'correlation')
        ASM = feature.greycoprops(graycom, 'ASM')

        contrast_mean = np.mean(contrast)#对比度
        dissimilarity_mean = np.mean(dissimilarity)#相异性
        homogeneity_mean = np.mean(homogeneity)#同质性
        energy_mean = np.mean(energy)#能量
        correlation_mean = np.mean(correlation)#相关性
        ASM_mean = np.mean(ASM)#ASM
        entropy = measure.shannon_entropy(image_gray)#熵

        # print("Contrast: {}".format(contrast_mean))
        # print("Dissimilarity: {}".format(dissimilarity_mean))
        # print("Homogeneity: {}".format(homogeneity_mean))
        # print("Energy: {}".format(energy_mean))
        # print("Correlation: {}".format(correlation_mean))
        # print("ASM: {}".format(ASM_mean))
        # print(measure.shannon_entropy(image_gray))

        return correlation_mean,dissimilarity_mean,homogeneity_mean,energy_mean,correlation_mean,ASM_mean,entropy



if __name__=="__main__":
    my_kernel = Kernel('F:/syl_kernel_traits/resize_data/2018-2019kernal/39/4-251-39-18.jpg')
    image = my_kernel.image()
    # cv2.imshow('image',image)

    image_gray,thresh_img = my_kernel.binary_image()
    cv2.imshow("image_gray",image_gray)
    cv2.imshow("thresh",thresh_img)

    no_background_image = my_kernel.no_background_image()
    cv2.imshow("result",no_background_image)


    R,G,B = my_kernel.RGB_mean()
    print("R:",R)
    print("G:",G)
    print("B:",B)

    H,S,V = my_kernel.HSV_mean()
    print("H:",H)
    print("S:",S)
    print("V:",V)

    # my_kernel.morphology_trait()

    my_kernel.texture_trait()

    cv2.waitKey(0)
    cv2.destroyAllWindows()