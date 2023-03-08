import numpy as np
import cv2

#загрузка изображенния и перевод в чб
def loading_gray_image(part):
    img =cv2.imread(part)
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


#для того, чтобы детектировать каждую букыу, нам необходимо написать три ключевые функции, которые:
#находит контуры всех букв;

def find_contours(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0) #блюрим, чтоб было легче искать контуры
    T, thresh_img = cv2.threshold(blurred_image, 215, 255,  #преобразуем изображение в бинарное. 
                                  cv2.THRESH_BINARY) #любое значение пикселя, превышающее 215, устанавливается 255, а любое значение, меньше 215, устанавливается равным нулю.THRESH_BINARY(), который указывает на то, что значения пикселей, которые больше 215 устанавливаются в максимальное значение, которое мы передали третьим параметром
    (_, cnts, _) = cv2.findContours(thresh_img, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    return cnts


#находит координаты каждой отдельной буквы;
#распознаёт букыу с помощью ключевых точек.
#потом надо порезать на буквы в массиве и что нейронка нашла и вывела в виде текста, если текст повернут, то поаернуть и вывести

part = "/home/zeroff/Рабочий стол/project_pic/examples/0.png"
image_gray = loading_gray_image(part)
find_contours(image_gray)

#нужно распознать текст, обьект на картинке, человек/животное/растение/не опознано

#Первым шагом разобьем текст на отдельные буквы. Для этого пригодится OpenCV, точнее его функция findContours.
#Откроем изображение (cv2.imread), переведем его в ч/б (cv2.cvtColor + cv2.threshold), слегка увеличим (cv2.erode) и найдем контуры.
#Мы получаем иерархическое дерево контуров (параметр cv2.RETR_TREE). Первым идет общий контур картинки, затем контуры букв, затем внутренние контуры. Нам нужны только контуры букв, поэтому я проверяю что «родительским» является общий контур.

part = "/home/zeroff/Рабочий стол/project_pic/examples/0.png"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy() #Метод copy используется для создания копии существующего массива в памяти

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour) #boundingRect нахождение контуроы
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