import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from collections import Counter


image_folder_portraits = '/home/ermak/semester_2/Images/Project/portraits/'
image_folder_landscapes = '/home/ermak/semester_2/Images/Project/landscapes/'

files_portraits = os.listdir(image_folder_portraits) # список изображений
files_landscapes = os.listdir(image_folder_landscapes)


# 1. Среднее значение яркости
def pixels_dataset(file):
    mean_bright_list = []
    image = Image.open(file)
    image_pixels = np.asarray(image)
    bright = np.zeros((image.size[1], image.size[0]))

    for i in range(image.size[1]):
        for j in range(image.size[0]):
            bright[i,j] = 0.222 * image_pixels[i][j][0] + 0.707 * image_pixels[i][j][1] + 0.071 * image_pixels[i][j][2]

    mean_bright = np.mean(bright) 
    mean_bright_list.append(mean_bright)
    mean_bright_list = np.array(mean_bright_list)

    return mean_bright_list

# pixels_dataset('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(pixels_dataset('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 2. Отношение сторон, где 1 - пейзаж, 0 - портрет
def sides_image(file):
    width = np.zeros((1, ))  
    height = np.zeros((1, )) 
    relation_sides = np.zeros((1, ))  

    image = Image.open(file)
    width[0] = image.size[0]
    height[0] = image.size[1]
    relation_sides[0] = width[0] / height[0]    # отношение ширины к высоте изображения

    vect = np.zeros(1).reshape((1, 1))  

    relation_sides = relation_sides.reshape((1, 1))  

    if relation_sides[0] > 1:   # отношение сторон    
        vect[0] = 1             # 1 - ширина больше высоты
    else:
        vect[0] = 0             # 0 - высота больше ширины 

    # print(relation_sides)
    vect_double = np.array(vect.tolist())
    vect = vect_double.flatten()
    vect = np.array(vect)

    return vect

# sides_image('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(sides_image('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 3. Соотношение красного и зеленого цветов на изображении
def correlation_r_g(file):
    img_arrs = []
    img = Image.open(file)
    img_arr = np.array(img.convert('RGB'))
    img_arrs.append(img_arr)

    r_g = []
    for img_arr in img_arrs:
        r_mean = np.mean(img_arr[:,:,0])
        g_mean = np.mean(img_arr[:,:,1])
        r_g_correlation = r_mean / g_mean
        r_g.append(r_g_correlation)
    
    r_g = np.array(r_g)
        
    return r_g

# correlation_r_g('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(correlation_r_g('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 4. Отношение зеленого и синего ко всем цветам на изображении
def correlation_g_b(file):
    img_arrs = []
    img = Image.open(file)
    img_arr = np.array(img.convert('RGB'))
    img_arrs.append(img_arr)

    g_b = []
    for img_arr in img_arrs:
        r_mean = np.mean(img_arr[:,:,0])
        g_mean = np.mean(img_arr[:,:,1])
        b_mean = np.mean(img_arr[:,:,2])
        g_b_correlation = (g_mean + b_mean) / (r_mean + g_mean + b_mean)
        g_b.append(g_b_correlation)
    
    g_b = np.array(g_b)

    return g_b

# correlation_g_b('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(correlation_g_b('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 5. Среднее преобладание голубого канала
def b_mean(file):
    mean_list = []
    img = cv2.imread(file)
    b = img[:,:,0]
    mean_b = np.mean(b)
    mean_list.append(mean_b)

    mean_list = np.array(mean_list)

    return mean_list

# b_mean('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(b_mean('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 6. Красный цвет в области лица
def correlation_face_r(file):
    img_arrs = []
    img = Image.open(file)
    img_arr = np.array(img.convert('RGB'))
    img_arrs.append(img_arr)

    red = []
    for img_arr in img_arrs:
        # где 0.395 - это отношение начала лица (по ширине) к общему размеру изображения (по ширине) в среднем (800x0.395)
        # где 0.598 - это отношение конца лица (по ширине) к общему размеру изображения (по ширине) в среднем 
        # где 0.213 - это отношение начала лица (по высоте) к общему размеру изображения (по высоте) в среднем 
        # где 0.428 - это отношение начала лица (по высоте) к общему размеру изображения (по высоте) в среднем 
        r_mean = np.mean(img_arr[int(img.size[0]*0.395):int(img.size[0]*0.598), int(img.size[1]*0.213):int(img.size[1]*0.428), 0])
        g_mean = np.mean(img_arr[:, :, 1])
        b_mean = np.mean(img_arr[:, :, 2])
        r_correlation = r_mean / (r_mean + g_mean + b_mean)
        red.append(r_correlation)

    red = np.array(red)
        
    return red

# correlation_face_r('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')
# print(type(correlation_face_r('/home/ermak/semester_2/Images/Project/frederik-leiton-blondinka.jpg')))


# 7. Градиент Собеля
# def grad_sob(file):
#     gradient_list = []
#     gray = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape

#     sobelx = np.zeros((height, width))
#     sobely = np.zeros((height, width))

#     for i in range(1, height-1):
#         for j in range(1, width-1):
#             sobelx[i,j] = (gray[i-1, j-1] + 2 * gray[i, j-1] + gray[i+1, j-1]) - (gray[i-1, j+1] + 2 * gray[i, j+1] + gray[i+1, j+1]) # для x
#             sobely[i,j] = (gray[i-1, j-1] + 2 * gray[i-1, j] + gray[i-1, j+1]) - (gray[i+1, j-1] + 2 * gray[i+1, j] + gray[i+1, j+1]) # для y

#     gradient = np.sqrt(sobelx ** 2 + sobely ** 2) # объединяем градиенты 

#     gradient_list.append(gradient)

#     mean_list = []
#     for i in range(len(gradient_list)):
#         mean_g = np.mean(gradient_list[i])
#         mean_list.append(mean_g)

#     mean_grad = np.array(mean_list)

#     return mean_grad

# grad_sob('/home/ermak/semester_2/Images/Project/tests/kulikov-i.s.-semya-lesnika.jpg')
# print(type(grad_sob('/home/ermak/semester_2/Images/Project/tests/kulikov-i.s.-semya-lesnika.jpg')))





# ВЕКТОР ПРИЗНАКОВ (всего 6)
def feature_vector(file):
    first_feature = pixels_dataset(file)
    second_feature = sides_image(file)
    third_feature = correlation_r_g(file)
    fourth_feature = correlation_g_b(file)
    fifth_feature = b_mean(file)
    sixth_feature = correlation_face_r(file)
    # seventh_feature = grad_sob(file)

    result_vector = np.concatenate([first_feature, second_feature, third_feature, fourth_feature, fifth_feature, sixth_feature])
    return result_vector

# feature_vector('/home/ermak/semester_2/Images/Project/tests/kulikov-i.s.-semya-lesnika.jpg')





# АЛГОРИТМЫ
# Для начала создадим общий датасет с портретами и пейзажами
def feature_vector_portraits():
    result_array = []
    for file in files_portraits:
        full_path = os.path.join(image_folder_portraits, file)
        feature_vector_result = feature_vector(full_path)
        result_array.append(['портрет'] + feature_vector_result.tolist())

    result_array = np.array(result_array)
    return np.array(result_array)

def feature_vector_landscapes():
    result_array = []
    for file in files_landscapes:
        full_path = os.path.join(image_folder_landscapes, file)
        feature_vector_result = feature_vector(full_path)
        result_array.append(['пейзаж'] + feature_vector_result.tolist())

    result_array = np.array(result_array)
    return np.array(result_array)

data = np.vstack([feature_vector_portraits(), feature_vector_landscapes()])




# 1. Применим НБА
# преобразуем data
X = data[:, 1:].astype(float)
y = data[:, 0] # все классы

# обучаем модель 
def fit(X, y):
    unique_classes = np.unique(y) # находим уникальные классы (портрет, пейзаж)
    # размерность для двух классов
    mean = np.zeros((len(unique_classes), X.shape[1]))
    var = np.zeros((len(unique_classes), X.shape[1]))
    proba = np.zeros(len(unique_classes))

    # для каждого класса
    for idx, yi in enumerate(unique_classes): # задаем счетчик индексов (0-портрет, 1-пейзаж)
        mean[idx, :] = np.mean(X[y == yi], axis=0)
        var[idx, :] = np.var(X[y == yi], axis=0)
        proba[idx] = sum(y == yi) / len(y)

    return mean, var, proba

# обучаем модель
mean, var, proba = fit(X, y)

# тестовые данные
test_data = feature_vector('/home/ermak/semester_2/Images/Project/tests/dang/kuzma-petrov-vodkin-portret-malchika.jpg')
test_data = test_data.reshape(1, -1) 

# предсказываем 
def predict(test_data, mean, var, proba):
    result = []
    unique_classes = len(proba)

    for xi in test_data: # вычисляем вероятность для каждого класса 
        max_proba, max_class = -1, None # инициализируем максимальную вероятность и соответствующий класс

        for yi in range(unique_classes): # вычисляем вероятность для каждого класса
            p = np.prod(np.exp(-(xi - mean[yi]) ** 2 / (2 * var[yi])) / np.sqrt(2 * np.pi * var[yi])) * proba[yi]
            # если вероятность текущего класса больше максимальной, устанавливаем новую максимальную вероятность и класс
            if p > max_proba:
                max_proba, max_class = p, yi

        # добавляем метку класса с наибольшей вероятностью
        result.append(np.unique(y)[max_class])

    return result

predictions = predict(test_data, mean, var, proba)
naive_bayes = predictions
# print(predictions)




# 2. Применим метод k ближайших соседей
test = feature_vector('/home/ermak/semester_2/Images/Project/tests/dang/kuzma-petrov-vodkin-portret-malchika.jpg')    
test = test.reshape(1, -1) 

arr = data[:, 1:]
arr = arr.astype(np.float64)

clusters = data[:, 0]
clusters = np.where(clusters == 'портрет', 0, clusters)
clusters = np.where(clusters == 'пейзаж', 1, clusters)

k = 3   

neighbours = [] # сохраняем соседей         
              
for n in range(len(test)):
    # записываем расстояния
    distances = []                       
    for i in range(len(arr)):
        dist = np.sqrt(np.sum((test[n] - arr[i])**2))
        distances.append((dist, clusters[i]))
    distances = sorted(distances)
    min_distance = []
    for j in range(k):
        # добавляет классы ближайших соседей в список min_distance
        min_distance.append(distances[j]) # смотрим на первых k ближайших соседей
        second_elements = [x[1] for x in min_distance] # выводим только номера кластеров
# print(distances)
# print(min_distance)
# print(second_elements) 

labels = [label for _, label in min_distance]

major_label = max(set(labels), key=labels.count)

def major(major_label):
    if major_label == '0':
        return ['портрет']
    else:
        return ['пейзаж']

knn = major(major_label)
# print(major(major_label))




# 3. Метод опорных векторов
X = data[:, 1:]
X = X.astype(np.float64)

y = data[:, 0]
y = np.where(y == 'пейзаж', 1, y) # поскольку так лучше гиперплоскость строить
y = np.where(y == 'портрет', -1, y)  
y = y.astype(int)

test = feature_vector('/home/ermak/semester_2/Images/Project/tests/dang/kuzma-petrov-vodkin-portret-malchika.jpg')
test = test.reshape(1, -1)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # нормализуем среднее = 0 и стн отклонение = 1
test = (test - np.mean(X, axis=0)) / np.std(X, axis=0)

w = np.zeros(X.shape[1])
learning_rate = 0.0001
lambda_r = 0.01 

for iter in range(1, 5001):  
    for i in range(X.shape[0]):
        xi, yi = X[i], y[i]
        if yi * np.dot(xi, w) < 1: 
            w += learning_rate * (xi * yi - 2 * lambda_r * w) # обновляем веса 
        else:
            w += learning_rate * (-2 * lambda_r * w) # (-2 * lambda_r * w) регуляризация для предотвращения переобучения

test_label = np.sign(np.dot(test, w)) # sign чтобы вернуть класс

def test(test_label):
    if test_label == [-1.]:
        return ['портрет']
    else:
        return ['пейзаж']

svm = test(test_label)
# print(test(test_label))




# РЕЗУЛЬТАТ
naive_bayes_weight = 0.346
knn_weight = 0.289
svm_weight = 0.365

answers = (naive_bayes) + (knn) + (svm)
answer_key = {"пейзаж": 1, "портрет": -1}
answers_array = np.array([answer_key[answer] for answer in answers])

weights = np.array([svm_weight, naive_bayes_weight, knn_weight])
weighted_results = answers_array * weights

total_error = np.mean(weighted_results) 
total_error = np.mean(weighted_results) * 100

if total_error > 0:
    print('Ответ алгоритмов: пейзаж')
else:
    print('Ответ алгоритмов: портрет')

print(answers)