from PIL import Image
from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO('yolov8n.pt')

# 在'bus.jpg'上运行推理
# 定义图像文件的路径
#path = 'D:\\PrProject\\ver4\\OutImage\\img'

# 对来源进行推理

model.train(data='gasvid.yaml',epochs=100)

model.val()

# 在'bus.jpg'上运行推理，并附加参数
#  model.predict(source = 'D:\\PrProject\\ver4\\OutImage', save=True,max_det = 1,show_labels = False, conf=0.6,stream = True)

#results = model(source+'%05d'%3888+'.png')  # Results 对象列表

# # 展示结果
# for r in results:
#     im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
#     im.show()  # 显示图像
#     im.save('results.jpg')  # 保存图像