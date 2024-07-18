import cv2
import numpy as np
import time
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Путь для сохранения видео
video_output_path = '/root/VideoEnd.avi'
result_video_path = '/root/result1a.avi'

# Запуск GStreamer для записи видео
gst_command = (
    'gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=300 ! '
    'videorate ! video/x-bayer,format=grbg,depth=8,width=1920,height=1080,framerate=25/1 ! '
    'bayer2rgbneon ! videoconvert ! vpuenc_h264 ! avimux ! filesink location=' + video_output_path
)

print("Запуск записи видео с устройства...")
subprocess.run(gst_command, shell=True)

# Проверка пути к записанному видеофайлу
if not os.path.isfile(video_output_path):
    print(f"Файл {video_output_path} не существует или путь указан неверно.")
    exit()

# Формирование строки GStreamer pipeline для чтения видео
gst_pipeline = (
    f'filesrc location={video_output_path} ! decodebin ! videoconvert ! '
    f'video/x-raw,format=BGR ! appsink'
)

# Чтение видеофайла через GStreamer
print(f"Открываю файл: {video_output_path} с использованием GStreamer")
vid = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not vid.isOpened():
    print("Ошибка: не удалось открыть видеофайл")
    exit()

# Импорт меток для YOLO
with open('/root/coco.names') as f:
    labels = [line.strip() for line in f]

# Указание класса "person"
person_class_index = labels.index('person')

# Загрузка модели YOLO
yolo = cv2.dnn.readNetFromDarknet('/root/yolov4-tiny.cfg', '/root/yolov4-tiny.weights')
all_layers = yolo.getLayerNames()

# Получение выходных слоев
output_layers = [all_layers[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]
minimum_probability = 0.3  # Уменьшение порога уверенности для повышения чувствительности
threshold = 0.3

# Инициализация переменных для кадра и времени
f = 0
t = 0
h, w = None, None
writer = None

# Уменьшение частоты кадров видео (при необходимости)
fps = vid.get(cv2.CAP_PROP_FPS)
frame_skip = int(fps // 5)  # Снижение FPS до 5

# Замер времени выполнения программы
program_start_time = time.time()

def process_frame(frame):
    global writer, h, w
    # Получение высоты и ширины кадра
    if w is None or h is None:
        h, w = frame.shape[:2]

    # Создание блоба из кадра для ввода в YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Установка блоба в качестве входа для YOLO и выполнение прямого прохода
    yolo.setInput(blob)
    output = yolo.forward(output_layers)

    # Инициализация списков для ограничивающих рамок, уверенностей и номеров классов
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Обработка вывода YOLO для получения ограничивающих рамок, уверенностей и номеров классов
    for result in output:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > minimum_probability and class_current == person_class_index:
                # Преобразование вывода YOLO в координаты ограничивающей рамки
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Применение не-максимального подавления для получения окончательных результатов ограничивающих рамок
    result = cv2.dnn.NMSBoxes(bounding_boxes, confidences, minimum_probability, threshold)

    # Масштабирование координат ограничивающих рамок обратно к исходному разрешению
    if len(result) > 0:
        for i in result.flatten():
            x_min, y_min, box_width, box_height = bounding_boxes[i]

            # Рисование красной рамки и метки на кадре
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 0, 255), 2)  # Красный цвет: (0, 0, 255)
            text_box_current = 'person: {:.4f}'.format(confidences[i])
            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Красный цвет: (0, 0, 255)
            print(f"Найдена рамка: {x_min}, {y_min}, {box_width}, {box_height}, уверенность: {confidences[i]:.4f}")

    else:
        print("Люди не найдены на этом кадре")

    # Запись обработанного кадра в выходной видеофайл
    if writer is None:
        print("Инициализация VideoWriter...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
        writer = cv2.VideoWriter(result_video_path, fourcc, 5, (w, h), True)
        if not writer.isOpened():
            print("Ошибка: не удалось открыть файл для записи видео")
            exit()

    writer.write(frame)
    return frame

# Инициализация пула потоков
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []

    # Цикл по кадрам
    while True:
        # Пропуск кадров для уменьшения FPS
        for _ in range(frame_skip):
            ret, frame = vid.read()
            if not ret:
                break

        # Прерывание, если кадры закончились
        if not ret:
            print("Кадры закончились или произошла ошибка при чтении кадра.")
            break

        f += 1
        start = time.time()
        future = executor.submit(process_frame, frame)
        futures.append(future)
        end = time.time()
        t += end - start

        # Ограничение на количество активных задач, чтобы избежать перегрузки
        if len(futures) > 10:
            futures = [f for f in futures if not f.done()]

# Дождаться завершения всех задач
for future in futures:
    future.result()

# Освобождение ресурсов
vid.release()
if writer is not None:
    writer.release()

# Расчет и вывод времени выполнения программы
program_end_time = time.time()
total_program_time = program_end_time - program_start_time
average_frame_time = t / f

print(f"Общее время выполнения программы: {total_program_time:.4f} секунд")
print(f"Среднее время обработки кадра: {average_frame_time:.4f} секунд")
