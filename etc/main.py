import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.constraints import maxnorm
# from keras.utils import np_utils
# from keras.datasets import cifar10
import cv2

# # Set random seed for purposes of reproducibility
# seed = 21

# #Подготовка данных, в эти переменные будут загружены данные
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# #В большинстве случаев вам потребуется выполнить некоторую предварительную обработку ваших данных, чтобы подготовить их к использованию, но поскольку мы используем уже готовый и упакованный набор данных, то такая обработка сведена к минимуму. Одним из действий, которые мы хотим сделать, будет нормализация входных данных.

# #Если значения входных данных находятся в слишком широком диапазоне, это может отрицательно повлиять на работу сети. В нашем случае входными значениями являются пиксели в изображении, которые имеют значение от 0 до 255.

# #Таким образом, чтобы нормализовать данные, мы можем просто разделить значения изображения на 255. Для этого нам сначала нужно перевести данные в формат с плавающей запятой, поскольку в настоящее время они являются целыми числами. Мы можем сделать это, используя Numpy команду astype(), а затем объявить желаемый тип данных:
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# #Следующая вещь, которую нам нужно сделать, чтобы подготовить данные для сети, - перевести их в унитарный код. 
# #Мы успешно применяем здесь двоичную классификацию, потому что изображение либо принадлежит одному определённому классу, либо нет: оно не может быть где-то посередине. Для унитарного кодирования используется команда Numpy to_categorical(). Вот почему мы импортировали функцию np_utils из Keras, так как она содержит to_categorical().

# # Нам также нужно задать количество классов в наборе данных, чтобы мы поняли, до скольких нейронов сжимать конечный слой:
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# class_num = y_test.shape[1]

# #Создание модели
# #Первый слой нашей модели - это сверточный слой. Он будет принимать входные данные и пропускать их через сверточные фильтры.
# #При реализации этого в Keras, мы должны указать количество каналов (фильтров), которое нам нужно (а это 32), размер фильтра (3 x 3 в нашем случае), форму входа (при создании первого слоя), функцию активации и отступы.
# model = Sequential()

# #Как уже упоминалось, relu является наиболее распространенной функцией активации, а отступы мы определим через padding = 'same', то есть, мы не меняем размер изображения:

# model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
# model.add(Activation('relu'))

# #Теперь мы создадим исключающий слой для предотвращения переобучения, который случайным образом устраняет соединения между слоями (0,2 означает, что он отбрасывает 20% существующих соединений):
# #Также мы можем выполнить пакетную нормализацию
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# #Теперь следует еще один сверточный слой, но размер фильтра увеличивается, так что сеть уже может изучать более сложные представления:
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))

# #А вот и объединяющий слой, который, как обсуждалось ранее, помогает сделать классификатор изображений более корректным, чтобы он мог изучать релевантные шаблоны. Также опишем исключение (Dropout) и пакетную нормализацию:

# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# #Теперь вы можете повторить эти слои, чтобы дать вашей сети больше представлений для работы:
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# #После того, как мы закончили со сверточными слоями, нам нужно сжать данные, поэтому мы импортировали функцию Flatten выше. Мы также добавим слой исключения снова:
# model.add(Flatten())
# model.add(Dropout(0.2))

# #Теперь мы используем импортированную функцию Dense и создаем первый плотно связанный слой. Нам нужно указать количество нейронов в плотном слое.
# #В этом последнем слое мы уравниваем количество классов с числом нейронов. Каждый нейрон представляет класс, поэтому на выходе этого слоя будет вектор из 10 нейронов, каждый из которых хранит некоторую вероятность того, что рассматриваемое изображение принадлежит его классу.
# model.add(Dense(256, kernel_constraint=maxnorm(3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(128, kernel_constraint=maxnorm(3)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# #Наконец, функция активации softmax выбирает нейрон с наибольшей вероятностью в качестве своего выходного значения, предполагая, что изображение принадлежит именно этому классу
# model.add(Dense(class_num))
# model.add(Activation('softmax'))

# #Теперь, когда мы разработали модель, которую хотим использовать, остаётся лишь скомпилировать ее. Давайте укажем количество эпох для обучения, а также оптимизатор, который мы хотим использовать.

# epochs = 25
# optimizer = 'adam'
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# print(model.summary())

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# # Final evaluation of the model

# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

#нужно распохнать текст, обьект на картинке, человек/животное/растение/не опознано

#Первым шагом разобьем текст на отдельные буквы. Для этого пригодится OpenCV, точнее его функция findContours.
#Откроем изображение (cv2.imread), переведем его в ч/б (cv2.cvtColor + cv2.threshold), слегка увеличим (cv2.erode) и найдем контуры.
#Мы получаем иерархическое дерево контуров (параметр cv2.RETR_TREE). Первым идет общий контур картинки, затем контуры букв, затем внутренние контуры. Нам нужны только контуры букв, поэтому я проверяю что «родительским» является общий контур.
image_file = "/home/zeroff/Рабочий стол/project_pic/examples/3.jpg"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy()

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)


cv2.imshow("Input", img)
cv2.imshow("Enlarged", img_erode)
cv2.imshow("Output", output)
cv2.waitKey(0)


#Следующим шагом сохраним каждую букву, предварительно отмасштабировав её до квадрата 28х28 (именно в таком формате хранится база MNIST). OpenCV построен на базе numpy, так что мы можем использовать функции работы с массивами для кропа и масштабирования.

def letters_extract(image_file: str, out_size=28) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

#В конце мы сортируем буквы по Х-координате, также как можно видеть, мы сохраняем результаты в виде tuple (x, w, letter), чтобы из промежутков между буквами потом выделить пробелы.
cv2.imshow("0", letters[0][2])
cv2.imshow("1", letters[1][2])
cv2.imshow("2", letters[2][2])
cv2.imshow("3", letters[3][2])
cv2.imshow("4", letters[4][2])
cv2.waitKey(0)