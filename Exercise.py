import cv2
from PIL import Image
#读取彩色图像
color_img = cv2.imread(r'me.jpg')
#在窗口中显示图像，该窗口和图像的原始大小自适应

#cvtColor的第一个参数是处理的图像，第二个是RGB2GRAY
gray_img=cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
cv2.imshow('original image',gray_img)
#gray_img此时还是二维矩阵表示,所以要实现array到image的转换
#gray=Image.fromarray(gray_img)
blur  = cv2.medianBlur(gray_img, 3)
cv2.imshow('blur',blur)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#生成自适应均衡化图像
dst2 = clahe.apply(blur)
cv2.imshow("dst2", dst2)
#将图片保存到当前路径下，参数为保存的文件名
#gray.save('gray.jpg')
#cv2.imshow('Gray Image',gray_img)
#如果想让窗口持久停留，需要使用该函数
cv2.waitKey(0)

