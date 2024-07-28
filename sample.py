import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 加載中文字型
font_path = 'C:/Windows/Fonts/msjhbd.ttc'  # 微軟正黑體
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# 1. 加載CIFAR-10數據集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10的類別名稱
class_names = ['飛機', '汽車', '鳥', '貓', '鹿', '狗', '青蛙', '馬', '船', '卡車']

# 隨機選擇一些圖片
num_images = 15
random_indices = np.random.choice(x_train.shape[0], num_images, replace=False)
sample_images = x_train[random_indices]
sample_labels = y_train[random_indices]

# 2. 顯示圖片和標籤
plt.figure(figsize=(15, 3))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_images[i])
    plt.xlabel(class_names[sample_labels[i][0]])
plt.show()
