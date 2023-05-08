# Программа распознавания лиц

Программа распознавания лиц использует библиотеку `face_recognition` для обнаружения и идентификации лиц на изображениях или в потоке с веб-камеры.

## Основные функции

- Обнаружение лиц: Программа способна обнаруживать лица на изображениях или в реальном времени с помощью веб-камеры.
- Распознавание лиц: После обнаружения лица программа может сопоставить его с известными лицами в базе данных и вывести соответствующее имя.
- Посещаемость: Программа также предоставляет возможность записывать посещаемость, сохраняя информацию о времени и именах обнаруженных лиц в файле `Attendance.csv`.

## Требования

Программа требует следующие зависимости:

- Python 3.x
- Библиотеки: `face_recognition`, `opencv-python`, `numpy`, `datetime`

## Установка и запуск

1. Установите Python 3.x, если его еще нет на вашем компьютере.
2. Установите необходимые зависимости, выполнив команду: `pip install face_recognition opencv-python numpy datetime`.
3. Скачайте код программы из репозитория.
4. Подготовьте изображения известных лиц, поместив их в папку `ImageAttendance`. Каждое изображение должно быть именовано в формате `имя.расширение` (например, `john.jpg`).
5. Запустите программу, выполнив команду: `python <путь_к_файлу.py>`.
6. Программа откроет видеопоток с веб-камеры и начнет распознавание лиц. Результаты будут отображены на экране, а также записаны в файл `Attendance.csv`.

## Важно

- Убедитесь, что веб-камера подключена к компьютеру и функционирует корректно.
- Убедитесь, что папка `ImageAttendance` содержит изображения известных лиц для распознавания.
- В файле `Attendance.csv` будут сохраняться данные по посещаемости. Убедитесь, что у вас есть соответствующие разрешения для записи в этот файл.

## Вклад

Вы можете внести свой вклад в улучшение программы, создавая новые функции, улучшая алгоритмы распознавания лиц или исправляя