import cv2
import numpy as np

a = 5  #放大倍数

for i in range(10765):

	#图像差分，ShortImage为原始图像文件夹
	img1 = cv2.imread('OriginImage/img'+ '%05d'%0 + '.png',0)     #背景图
	img2 = cv2.imread('OriginImage/img'+ '%05d'%i + '.png',0) #帧图像
	img = cv2.absdiff(img2,img1)  #图像差分
	for t in range(img.shape[0]):
		for r in range(img.shape[1]):
			img[t,r] = min(254,a*img[t,r])
			if img[t,r] <= 55:
				img[t,r] = 0
	#cv2.imwrite('NewDiffImage/Grimg_'+'%04d'%i + '.png',img)   #img输出为差分图片

	#差分图像与原图像按权重融合
	dst=cv2.addWeighted(img,0.6,img2,0.4,0)
	#res = cv2.add(img1,img)      #差分图像和原图像直接融合
	#cv2.imwrite('AddImage/img_'+'%04d'%(i+1) + '.png',dst) #dst输出为融合后图像
	#中值滤波
	blur  = cv2.medianBlur(dst, 3)  #中值滤波,消除椒盐噪声，ksize=3

	#图像限制对比度直方图均衡化  增大图像对比度
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
	img_CLAHE = clahe.apply(blur)
	cv2.imwrite('OutImage/img'+'%05d'%i + '.png',img_CLAHE)  #img_CLAHE输出为结果


#  图像存在黑边和水印，对其进行处理。截去黑边以及水印裁剪。
#  考虑到考虑拍摄时的相机抖动等因素，不同帧之间图像的拍摄位置可能存在不一致的情况，需要图像配准，即将第一帧作为基准图像，剩余帧分别与其进行对准。即可得到全部帧的对准图像。
#  将各帧图像与第一帧进行差分，并将差分图像进行放大5倍后得到的差分图像（为了提升对比度，将灰度值小于55的命为0）于原图像按权重融合。
#  将得到的图像进行中值滤波，消除椒盐噪声。
#  滤波在一定程度上模糊了图像，因此对其进行直方图均衡化，提高图像对比度。

