import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 列出待获取数据内容的文件位置
    # v5、v8都是csv格式的，v7是txt格式的
    result_dict = {
        'YOLOv8m': r'C:\Users\dell\Desktop\trainmod\train4\results.csv',
    }

    # 绘制map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()    # 6是指map50的下标（每行从0开始向右数）
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')   # 线条粗细设为1

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("Recall", dpi=600)   # dpi可设为300/600/900，表示存为更高清的矢量图
    plt.show()


    # 绘制map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        data = pd.read_csv(res_path, usecols=[4]).values.ravel()    # 7是指map50-95的下标（每行从0开始向右数）
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("Precision", dpi=600)
    plt.show()


    # 绘制训练的总loss
    res_path = result_dict[modelname]
    box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
    obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
    cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
    x = range(len(box_loss))
    y = range(len(obj_loss))
    z = range(len(cls_loss))
    plt.plot(x, box_loss, label='box_loss', linewidth='1')
    plt.plot(y, obj_loss, label='obj_loss', linewidth='1')
    plt.plot(z, cls_loss, label='cls_loss', linewidth='1')
    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("loss.png", dpi=600)
    plt.show()
