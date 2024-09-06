import cv2
from flask import Flask, Response
from ultralytics import YOLO

# Инициализация Flask приложения
app = Flask(__name__)

# Параметры RTSP потока и YOLO модели
rtsp_source = "rtsp://192.168.3.11:9099/stream"
detection_width = 320  # Уменьшенное разрешение для детекции
detection_height = 240
model = YOLO("yolov8n.pt")
person_class_index = 0  # Класс 'person' в COCO датасете имеет ID = 0

# Частота кадров RTSP-потока
frame_skip = 3  # Обрабатываем каждый третий кадр

# Функция для генерации кадров с детекцией людей
def generate_frames():
    cap = cv2.VideoCapture(rtsp_source)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Пропускаем кадры для снижения частоты обработки
        if frame_count % frame_skip != 0:
            continue

        # Получаем исходные размеры кадра
        original_height, original_width = frame.shape[:2]

        # Уменьшение разрешения кадра для детекции
        frame_resized = cv2.resize(frame, (detection_width, detection_height))

        # Детекция объектов с использованием YOLO
        results = model(frame_resized)

        # Обработка результатов детекции
        for result in results:
            for detection in result.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]

                if confidence > 0.5 and class_id == person_class_index:
                    # Пропорционально масштабируем координаты к исходному размеру
                    x_min = int(x_min * (original_width / detection_width))
                    y_min = int(y_min * (original_height / detection_height))
                    x_max = int(x_max * (original_width / detection_width))
                    y_max = int(y_max * (original_height / detection_height))

                    # Рисование красной рамки и метки на исходном кадре
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Красный цвет
                    text = 'person: {:.4f}'.format(float(confidence))
                    cv2.putText(frame, text, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Кодирование кадра в формат JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Маршрут для стрима
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Запуск приложения Flask
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
