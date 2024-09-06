import cv2
from flask import Flask, Response
from ultralytics import YOLO

# Инициализация Flask приложения
app = Flask(__name__)

# Параметры RTSP потока и YOLO модели
rtsp_source = "rtsp://192.168.3.11:9099/stream"
width = 800
height = 480
model = YOLO("yolov8n.pt")
person_class_index = 0  # Класс 'person' в COCO датасете имеет ID = 0

# Функция для генерации кадров с детекцией людей
def generate_frames():
    cap = cv2.VideoCapture(rtsp_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Детекция объектов с использованием YOLO
        results = model(frame)

        # Обработка результатов детекции
        for result in results:
            for detection in result.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]

                if confidence > 0.5 and class_id == person_class_index:
                    # Рисование красной рамки и метки на кадре
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                                  (0, 0, 255), 2)  # Красный цвет
                    text = 'person: {:.4f}'.format(float(confidence))
                    cv2.putText(frame, text, (int(x_min), int(y_min) - 5),
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

