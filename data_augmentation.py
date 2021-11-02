import os
# 解析xml文件
import xml.etree.ElementTree as ET
import sys
import cv2
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from skimage import io, util

class DataAugmentation():
    """
    data augmentation class, include *** methods
    """
    def __init__(self, img_path, label_path):
         self.img_path = img_path
         self.label_path = label_path
         
    def load_data(self):
        img_names = os.listdir(self.img_path)
        # img_names.sort(key=lambda x:int(x[:-4]))
        label_names = os.listdir(self.label_path)
        # label_names.sort(key=lambda x:int(x[:-4]))
        return img_names, label_names

    def image_flip(self, mode, result_path):
        # augment image data by flipping
        img_names, label_names = self.load_data()
        file_num = len(img_names)
        for i in range(file_num):
            #            print('----------------processing the {} image----------------'.format(i))
            # flip the image
            img_name = img_names[i]
            img = cv2.imread(img_path + '/{}'.format(img_name), cv2.IMREAD_COLOR)
            flip_img = cv2.flip(img, mode)
            img_new_name = img_name
            cv2.imwrite(result_path+'/{}'.format(img_new_name), flip_img)
            # process the corresponding label files
            label_name = os.path.splitext(img_name)[0] + '.xml'
            xmlFilePath = os.path.abspath(label_path+'/{}'.format(label_name))
            try:
                tree = ET.parse(xmlFilePath)
                root = tree.getroot()
            except Exception as e:
                print('parse test.xml fail!')
                sys.exit()

            # get the image size
            for element in root.findall('size'):
                width = element.find('width').text
                height = element.find('height').text

            if mode == -1:
                # 遍历root的下一层, both direction flip
                for element in root.findall('object'):
                    for item in element.findall('bndbox'):
                        xmin = item.find('xmin').text
                        xmax = item.find('xmax').text
                        ymin = item.find('ymin').text
                        ymax = item.find('ymax').text
                        new_xmin = int(width) - int(xmax)
                        new_xmax = int(width) - int(xmin)
                        new_ymin = int(height) - int(ymax)
                        new_ymax = int(height) - int(ymin)
                        item.find('xmin').text = str(new_xmin)
                        item.find('xmax').text = str(new_xmax)
                        item.find('ymin').text = str(new_ymin)
                        item.find('ymax').text = str(new_ymax)

                xml_name = label_name
                tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')

            elif mode == 1:
                # 遍历root的下一层, horizontal flip
                for element in root.findall('object'):
                    for item in element.findall('bndbox'):
                        xmin = item.find('xmin').text
                        xmax = item.find('xmax').text
                        new_xmin = int(width) - int(xmax)
                        new_xmax = int(width) - int(xmin)
                        item.find('xmin').text = str(new_xmin)
                        item.find('xmax').text = str(new_xmax)

                xml_name = label_name
                tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')
            else:
                # 遍历root的下一层, vertical flip
                for element in root.findall('object'):
                    for item in element.findall('bndbox'):
                        ymin = item.find('ymin').text
                        ymax = item.find('ymax').text
                        new_ymin = int(height) - int(ymax)
                        new_ymax = int(height) - int(ymin)
                        item.find('ymin').text = str(new_ymin)
                        item.find('ymax').text = str(new_ymax)

                xml_name = label_name
                tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')

    def image_gamma(self, result_path):
        # augment image data by gamma transform(random)
        img_names, label_names = self.load_data()
        file_num = len(img_names)
        factor = [0.4, 1.5]
        for i in range(file_num):
            img_name = img_names[i]
            ori_img = io.imread(img_path + '/{}'.format(img_name))
            pre_gamma_img = ori_img / 255.0
            # gamma transform
            random.shuffle(factor)
            gamma = factor[0]
            ga_img = np.power(pre_gamma_img, gamma)
            img_new_name = img_name
            io.imsave(result_path + '/{}'.format(img_new_name), ga_img*255)
            # process the corresponding label files
            label_name = os.path.splitext(img_name)[0] + '.xml'
            xmlFilePath = os.path.abspath(label_path + '/{}'.format(label_name))
            try:
                tree = ET.parse(xmlFilePath)
            #                root = tree.getroot()
            except Exception as e:
                print('parse test.xml fail!')
                sys.exit()
            xml_name = label_name
            tree.write(result_path + '/{}'.format(xml_name), encoding='utf-8')

    def image_color(self, result_path):
        #随机对图像亮度、饱和度、对比度、锐度进行变换
        img_names, label_names = self.load_data()
        file_num = len(img_names)
        for i in range(file_num):
            #
            img_name = img_names[i]
            ori_img = Image.open(img_path + '/{}'.format(img_name))
            random_factor = (np.random.random() + 6.) / 10.  # 随机因子
            brightness_image = ImageEnhance.Brightness(ori_img).enhance(random_factor)  # 调整图像的亮度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            color_image = ImageEnhance.Color(brightness_image).enhance(random_factor)  # 调整图像的饱和度
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            contrast_image = ImageEnhance.Contrast(color_image).enhance(random_factor)  # 调整图像对比度
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            img_new = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
            img_new_name = img_name
            img_new.save(result_path+'/{}'.format(img_new_name))
            # process the corresponding label files
            label_name = os.path.splitext(img_name)[0] + '.xml'
            xmlFilePath = os.path.abspath(label_path+'/{}'.format(label_name))
            try:
                tree = ET.parse(xmlFilePath)
            #                root = tree.getroot()
            except Exception as e:
                print('parse test.xml fail!')
                sys.exit()
            xml_name = label_name
            tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')

    def image_gaussian_noise(self, result_path):
        img_names, label_names = self.load_data()
        file_num = len(img_names)
        # index_list = list(range(file_num))
        # add gaussian noise
        # selected_list.sort()
        # random_num = len(selected_list)
        for i in range(file_num):
            img_name = img_names[i]
            ori_img = io.imread(img_path + '/{}'.format(img_name))
            gaussian_img = util.random_noise(ori_img, mode='gaussian', seed=None, clip=True, var=0.05)  #添加高斯噪声
            img_new_name = img_name
            io.imsave(result_path+'/{}'.format(img_new_name), gaussian_img)
            # process the corresponding label files
            label_name = os.path.splitext(img_name)[0] + '.xml'
            xmlFilePath = os.path.abspath(label_path+'/{}'.format(label_name))
            try:
                tree = ET.parse(xmlFilePath)
#                root = tree.getroot()
            except Exception as e:
                print('parse test.xml fail!')
                sys.exit()
            xml_name = label_name
            tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')

    def image_sp_noise(self, result_path):
        # add salt & pepper noise
        img_names, label_names = self.load_data()
        file_num = len(img_names)
        for i in range(file_num):
            img_name = img_names[i]
            ori_img = io.imread(img_path + '/{}'.format(img_name))
            sp_img = util.random_noise(ori_img, mode='s&p', seed=None, clip=True)  #添加椒盐噪声
            img_new_name = img_name
            io.imsave(result_path+'/{}'.format(img_new_name), sp_img)
            # process the corresponding label files
            label_name = os.path.splitext(img_name)[0] + '.xml'
            xmlFilePath = os.path.abspath(label_path+'/{}'.format(label_name))
            try:
                tree = ET.parse(xmlFilePath)
#                root = tree.getroot()
            except Exception as e:
                print('parse test.xml fail!')
                sys.exit()
            xml_name = label_name
            tree.write(result_path+'/{}'.format(xml_name), encoding='utf-8')

if __name__ == '__main__':
    img_path = r'D:\Project\diaokuai\data2\fault_data2\type34\original_data\JPEGImages'
    label_path = r'D:\Project\diaokuai\data2\fault_data2\type34\original_data\Annotations'
    # result_path = r'D:\Project\diaokuai\data2\fault_data/gaussian_noise/{}'
    result_path = r'D:\Project\diaokuai\data2\fault_data2\type34'
    # print(result_path[:-3])
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    da = DataAugmentation(img_path, label_path)

    #mode: 1水平翻转，0垂直翻转，-1垂直水平翻转
    # filp_path_1 = result_path + '/filp/-1'
    # # os.makedirs(result_path + './filp'+'./-1')
    # if not os.path.exists(filp_path_1):
    #     os.makedirs(filp_path_1)
    # da.image_flip(mode=-1, result_path=filp_path_1)

    # filp_path_2 = result_path + '/filp/0'
    # if not os.path.exists(filp_path_2):
    #     os.makedirs(filp_path_2)
    # da.image_flip(mode=0, result_path=filp_path_2)
    #
    # filp_path_3 = result_path + '/filp/1'
    # if not os.path.exists(filp_path_3):
    #     os.makedirs(filp_path_3)
    # da.image_flip(mode=1, result_path=filp_path_3)
    #
    # light_path = result_path + '/light'
    # if not os.path.exists(light_path):
    #     os.makedirs(light_path)
    # da.image_gamma(result_path=light_path)

    color_path = result_path + '/color'
    if not os.path.exists(color_path):
        os.makedirs(color_path)
    da.image_gamma(result_path=color_path)

    # gaussian_path = result_path + '/gaussian_noise'
    # if not os.path.exists(gaussian_path):
    #     os.makedirs(gaussian_path)
    # da.image_gaussian_noise(result_path=gaussian_path)
    #
    # sp_path = result_path + '/sp_noise'
    # if not os.path.exists(sp_path):
    #     os.makedirs(sp_path)
    # da.image_sp_noise(result_path=sp_path)