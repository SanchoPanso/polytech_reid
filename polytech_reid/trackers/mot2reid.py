import os
import cv2

def extract_images_from_gt(gt_file, image_folder, output_folder):
    # Проверяем наличие папки для вывода
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Чтение файла gt и извлечение данных
    with open(gt_file, 'r') as file:
        lines = file.readlines()
    
    cached_image = None
    cached_frame_id = -1

    for line in lines:
        parts = line.strip().split(',')
        frame_id, object_id, x, y, width, height, _, _, _ = parts
        frame_id = int(frame_id)
        object_id = int(object_id)
        x, y, width, height = map(float, [x, y, width, height])

        # Чтение кадра изображения
        if cached_image is not None and cached_frame_id == frame_id:
            image = cached_image
        else:
            image_path = os.path.join(image_folder, f"{frame_id}.jpg")
            image = cv2.imread(image_path)
            cached_frame_id = frame_id
            cached_image = image

        # Вырезаем область с объектом
        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        object_image = image[y1:y2, x1:x2]

        # Сохраняем вырезанное изображение
        output_subdir = os.path.join(output_folder, f"{object_id:04d}")
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f"{frame_id:06d}.jpg")
        cv2.imwrite(output_path, object_image)
        print(frame_id, object_id)

# Пример использования функции
gt_file = r"D:\CodeProjects\PythonProjects\polytech_reid\datasets\person_seq_2_1\gt\gt.txt"
image_folder = r"D:\CodeProjects\PythonProjects\polytech_reid\datasets\person_seq_2_1\img2"
output_folder = r"D:\CodeProjects\PythonProjects\polytech_reid\datasets\person_seq_2_1\reid"
extract_images_from_gt(gt_file, image_folder, output_folder)
