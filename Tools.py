import os
import cv2
import math

from skimage import morphology

import numpy as np


def pathFile(path):
    """
    验证路径是否存在，如果不存在则创建路径
    @param path: 输入路径/待创建路径
    @return: 经验证/创建后的路径
    """
    if os.path.exists(path):
        return path
    else:
        os.makedirs(path)
    return path


def showImage(winname, img, k=1):
    """
    调用cv模块显示单张图片
    @param winname: 待显示图像窗口的名称
    @param img: 待显示图像
    @param k: 待显示窗口的缩放比，默认为1，即安照图像大小显示
    """
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, int(k * img.shape[1]), int(k * img.shape[0]))
    cv2.imshow(winname, img)
    cv2.waitKey(0)


def imgRotate(img, k):
    """
    采用仿射变换旋转图像，旋转后产生的黑边以镜像方式进行填充
    @param img: 待校正图像，三通道或单通道
    @param k: 图像中目标对象倾斜斜率，以水平向右为0斜率基准
    @return: 校正后图像，类型（通道数、深度）与输入图像一致
    """
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), math.atan(k) * 180 / math.pi, 1)  # 旋转变化矩阵
    dst = cv2.warpAffine(img, matRotate, (width, height), borderMode=cv2.BORDER_REFLECT, borderValue=(255,255,255))  # 旋转
    return dst


def inverse_imgRotate(coordinate, matRotate):
    """
    对于仿射变换旋转后的图像中某一点，进行逆变换得到原图中坐标
    @param coordinate: 旋转后的图像中某一点坐标
    @param matRotate: 仿射变换矩阵
    @return: 求解得到的原图中坐标
    """
    coefficient_matrix = matRotate[:,:2]
    constant_matrix = np.array([[coordinate[0],coordinate[1]]]).T - matRotate[:,-1:]
    inverse_coord = linalg.solve(coefficient_matrix, constant_matrix)
    return (int(inverse_coord[0]), int(inverse_coord[1]))



def percent_range(dataset, min=0.20, max=0.80):
    """
    1型离群点剔除，采用比例法剔除离群值
    @param dataset: 待剔除数据，类型要求为1维ndarray数组
    @param min: 最小阈值系数，即剔除升序排列后所处位置小于该 系数*数据长度 的值
    @param max: 最大阈值系数，即剔除升序排列后所处位置大于该 系数*数据长度 的值
    @return: 剔除离群点后数据，类型为1维ndarray数组
    """
    if len(dataset) < 3:
        min=0
        max=1
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)
    # 搜索离群位置，前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value <= range_max and value >= range_min:
            new_data.append(value)
    return np.array(new_data)


def three_sigma(dataset, n=1.0):
    """
    2型离群点剔除，采用均值和标准差方法剔除离群值
    @param dataset: 待剔除数据，格式要求为1维ndarray数组
    @param n: 标准差倍数阈值，某一数据与均值的差值超过 n*标准差 将被剔除
    @return: 剔除离群点后数据，类型为1维ndarray数组
    """
    mean = np.mean(dataset)
    sigma = np.std(dataset)
    reserve_idx = np.where(abs(dataset - mean) <= n * sigma)
    new_data = dataset[reserve_idx]
    return new_data


def MAD(dataset, n):
    """
    3型离群点剔除，采用基于中位数方法剔除离群值
    @param dataset: 待剔除数据，格式要求为1维ndarray数组
    @param n: 倍数阈值，某一数据与中位数的差值超过 n*mad 将被剔除
    @return: 剔除离群点后数据，类型为1维ndarray数组
    """
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)
    reserve_idx = np.where(abs(dataset - median) <= n * mad)
    new_data = dataset[reserve_idx]
    return new_data


def gammaAdjust(img, In, Out, gamma=1.0):
    """
    对图像进行gamma校正
    @param img: 待校正图像，三通道或单通道
    @param In: 待校正的灰度区域，以比例区间的形式, eg: (0, 1)
    @param Out: 输出校正的灰度区域，以比例区间的形式, rg: (0.1, 0.7)
    @param gamma: gamma校正的参数，默认为1，即保持原灰度值
    @return: 校正后图像，类型（通道数、深度）与输入图像一致
    """
    lowIn, highIn = In
    lowOut, highOut = Out
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    table = np.where(table < lowIn * 255, lowOut * 255, table)
    table = np.where(table > highIn * 255, highOut * 255, table)
    return cv2.LUT(np.array(img, dtype=np.uint8), table)


def histeq(img, flag=0, clipLimit=2.0, titleGridSize=7):
    """
    直方图均衡化，默认采用全局均衡化
    @param img: 待直方图均衡化图像，要求是灰度图像
    @param flag: 均衡化方式选择。 0: 全局均衡化； 1: 局部均衡化，默认为0
    @param clipLimit: 颜色对比度的阈值
    @param titleGridSize: 进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    @return: 直方图均衡化后的图像，类型（通道数、深度）与输入图像一致
    """
    # flag为0表示采全局均衡化，flag非0表示采用局部均衡化
    if flag:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(titleGridSize, titleGridSize))
        return clahe.apply(img)
    else:
        return cv2.equalizeHist(img)


def brightness_alpha(gray_img, gray_value = 125):
    """
    计算单张灰度图像的暗度系数，即较暗像素点所占比例
    @param gray_img: 待计算图像，要求为灰度图
    @param gray_value: 暗度阈值，小于该值的将被认为是较暗像素点，计入暗度统计中
    @return: 输入图像的暗度系数
    """
    r, c = gray_img.shape
    piexs_sum = r * c  # 整个灰度图的像素个数为r*c

    # 遍历灰度图的所有像素
    dark_points = (gray_img < gray_value)
    target_array = gray_img[dark_points]
    dark_sum = target_array.size
    return dark_sum/(piexs_sum)


def edge_enhance(img, flag = 'mysobelx01'):
    """
    对输入图像进行边缘增强，暂时包括sobel、scharr、prewitt三种算法
    @param img: 待处理图像，三通道或单通道图
    @param flag: 增强算子标识符
    @return: 经过边缘增强后的图像，类型（通道数、深度）与输入图像一致
    """
    kernel_sobelx_01 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)  # sobel，增强黑变白边界
    kernel_sobelx_10 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)  # sobel,增强白变黑边界

    kernel_scharrx_01 = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32)  # scharr,sobel的进阶版
    kernel_scharrx_10 = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], np.float32)  # scharr

    kernel_prewittx_01 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)  # prewitt
    kernel_prewittx_10 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)  # prewitt

    #LOG
    #先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    # # 再通过拉普拉斯算子做边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=5)
    LOG = cv2.convertScaleAbs(dst)

    #scharr算子,水平锐化
    scharrx = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)

    # scharr算子,垂直锐化
    scharry = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.convertScaleAbs(scharry)

    if flag == 'mysobelx01':
        return cv2.filter2D(img, -1, kernel=kernel_sobelx_01)
    if flag == 'mysobelx10':
        return cv2.filter2D(img, -1, kernel=kernel_sobelx_10)
    if flag == 'myscharrx01':
        return cv2.filter2D(img, -1, kernel=kernel_scharrx_01)
    if flag == 'myscharrx10':
        return cv2.filter2D(img, -1, kernel=kernel_scharrx_10)
    if flag == 'myprewittx01':
        return cv2.filter2D(img, -1, kernel=kernel_prewittx_01)
    if flag == 'myprewittx10':
        return cv2.filter2D(img, -1, kernel=kernel_prewittx_10)
    if flag == 'cvLOG':
        return LOG
    if flag == 'cvscharrx':
        return scharrx
    if flag == 'cvscharry':
        return scharry


def th(img, thresh):
    """
    对输入图像二值化处理，例如将灰度图像变为二值图像
    @param img: 单通道或多通道图像
    @param thresh: 二值化阈值，大于该值的像素点将被设为255，否则设为0
    @return: 经过固定阈值二值化后的图像
    """
    ret, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return img.astype('uint8')


def thin_skeleton(binary):
    """
    提取二值图像白色区域骨架；对图像做细化处理
    @param binary: 待提取骨架图像，要求是二值图
    @return: 经过骨架提取/细化后的图像
    """
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton

def connectedDomainAnalysis(img, th, thx, thy):
    """
    1型连通域分析，采用黑区覆盖的方法筛选不满足条件的连通域，特点是快速，但覆盖时不精准
    @param img: 待分析图像，一般为二值图
    @param th: 连通域面积阈值，面积小于该值的连通域将被抹除
    @param thx: 连通域宽度（水平方向）阈值，宽度小于该值的连通域将被抹除
    @param thy: 连通域高度（垂直方向）阈值，高度小于该值的连通域将被抹除
    @return: 经连通域分析过后的图像，通道数及深度与输入图像一致
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    stats = np.delete(stats, 0, axis=0)
    for istat in stats:
        if istat[4] < th or (istat[2] < thx or istat[3] < thy):
            cv2.rectangle(img, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26

    return img

def connectedDomainAnalysis_accly(img, th, thx, thy, pt = 0): #效果好速度慢
    """
    2型连通域分析，采用像素点级方法抹除不满足条件的连通域，特点是抹除区域精准，但时间花销大
    @param img: 待分析图像，一般为二值图
    @param th: 连通域面积阈值，面积小于该值的连通域将被抹除
    @param thx: 连通域宽度（水平方向）阈值，宽度小于该值的连通域将被抹除
    @param thy: 连通域高度（垂直方向）阈值，高度小于该值的连通域将被抹除
    @param pt: 输出输入图像全部连通域信息，默认为0，即不输出
    @return: 经连通域分析过后的图像，通道数及深度与输入图像一致
    """
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    stats = np.delete(stats, 0, axis=0)
    nb_components = nb_components - 1
    if pt:
        print(stats)
    for i in range(0, nb_components):
        istat = stats[i]
        if istat[4] < th or (istat[2] < thx or istat[3] < thy):
            img[labels == i + 1] = 0
    return img


def save_data(file,save_path,txt_name='temp.txt'):
    """
    将变量数据存入txt文本中，适用于列表类型数据，当前保存格式为列表内每元素单独一行，可按需修改
    @param file: 待保存变量，类型当前适用于list，其它数据类型需适当调整
    @param save_path: txt文本存储路径
    @param txt_name: 保存的文本名，需要带后缀
    """
    txt_path = save_path + '/' + txt_name
    st = '\n'
    if len(file):
        with open(txt_path, 'a') as f:
            new_file = [str(x) for x in file]
            f.write(st.join(new_file))
            f.write('\n\n')
    return


def Hex_to_BGR(hex):
    """
    将形如'#ff0000'的16进制颜色格式颜色转换为如(0,0,255)BGR或者RGB格式
    @param hex: 16进制颜色格式
    @return: BGR或者RGB格式
    """
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    bgr = (b,g,r)
    # print(bgr)
    return bgr