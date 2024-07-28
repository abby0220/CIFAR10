import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 加載中文字型
font_path = 'C:/Windows/Fonts/msjhbd.ttc'  # 微軟正黑體
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# 1. 加載CIFAR-10數據集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 正規化像素值到0-1之間
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 構建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# 查看模型架構
model.summary()

# 3. 編譯和訓練模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# 4. 可視化accuracy和loss圖表
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.xlabel('Epoch')
plt.ylabel('準確率')
plt.legend(loc='lower right')
plt.title('訓練和驗證準確率')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='驗證損失')
plt.xlabel('Epoch')
plt.ylabel('損失')
plt.legend(loc='upper right')
plt.title('訓練和驗證損失')

plt.show()

# 7. 输出ValAcc、TrainAcc、ValLoss、TrainLoss数据阵列
val_acc = history.history['val_accuracy']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
train_loss = history.history['loss']

print("ValAcc: ", val_acc)
print("TrainAcc: ", train_acc)
print("ValLoss: ", val_loss)
print("TrainLoss: ", train_loss)

# 5. 可視化一些測試圖片及其預測結果
class_names = ['飛機', '汽車', '鳥', '貓', '鹿', '狗', '青蛙', '馬', '船', '卡車']

# 預測測試集中的前25張圖片
predictions = model.predict(x_test[:25])

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[tf.argmax(predictions[i])])

plt.show()
