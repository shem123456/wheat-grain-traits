# 主函数
# 批量提取籽粒不同时期的颜色，形状和纹理特征

import os
import cv2
import numpy as np
import math
from skimage import morphology
from kernel import *
import pandas as pd

def main(path):
    files = os.listdir(path)
    file_csv = []
    for file in files:
        images = os.listdir(os.path.join(path,file))
        for image in images:
            try:
                # 每张图片的绝对路径
                image_path = os.path.join(path,file,image)
                print(image_path)

                my_kernel = Kernel(image_path)
                # 图像名字
                name = image
                name_list = name.split('-')
                new_name_list = []
                for s in name_list[1]:
                    new_name_list.append(s)
                variety,nitrogen,replicate = new_name_list
                # 颜色特征
                R,G,B = my_kernel.RGB_mean()
                H,S,V = my_kernel.HSV_mean()
                # 形状特征
                area,length,radius,equi_diameter,eccentric,compact,rectangle_degree,roundness = my_kernel.morphology_trait()
                # 纹理特征
                correlation_mean,dissimilarity_mean,homogeneity_mean,energy_mean,correlation_mean,ASM_mean,entropy = my_kernel.texture_trait()

                # 字典
                result_dict = {
                        "days":int(file),
                        "variety":variety,"nitrogen":nitrogen,"replicate":replicate,
                        "image_name":name,
                        "R":R,"G":G,"B":B,"H":H,"S":S,"V":V,
                        "area":area,"length":length,"radius":radius,"equi_diameter":equi_diameter,"eccentric":eccentric,
                        "compact":compact,"rectangle_degree":rectangle_degree,"roundness":roundness,
                        "correlation":correlation_mean,"dissimilarity":dissimilarity_mean,
                        "homogeneity":homogeneity_mean,"energy":energy_mean,"correlation":correlation_mean,"ASM":ASM_mean,"entropy":entropy
                }
                # 加
                file_csv.append(result_dict)
            except:
                pass
        
        print("保存{}.csv文件！".format(file))
    df = pd.DataFrame(file_csv)
    df.to_csv('result.csv')


if __name__=="__main__": 
    path = 'F:/syl_kernel_traits/resize_data/2018-2019kernal/'
    main(path)