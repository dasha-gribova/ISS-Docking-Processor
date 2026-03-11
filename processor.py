"""
Логика обработки изображений
"""
import os
import re
import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISSDataLoader:
    """Загрузчик данных"""

    def __init__(self):
        self.df = None
        self.images_path = None
        self.csv_path = None

    def set_paths(self, images_path, csv_path):
        self.images_path = Path(images_path)
        self.csv_path = Path(csv_path)

    def load_csv(self):
        """Загрузка CSV файла"""
        try:
            self.df = pd.read_csv(self.csv_path)
            return True, f"Загружено {len(self.df)} записей"
        except Exception as e:
            return False, str(e)

    def parse_location(self, location_str):
        """Парсинг координат из строки"""
        location_str = str(location_str).strip()
        numbers = re.findall(r'-?\d+\.?\d*', location_str)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        return None, None

    def find_image(self, image_id):
        """Поиск изображения по ID"""
        for ext in ['.jpg', '.jpeg', '.png']:
            path = self.images_path / f"{image_id}{ext}"
            if path.exists():
                return path
        return None


class KeypointProcessor:
    """Обработчик ключевых точек"""

    def __init__(self, model_name="yolov8m-pose.pt", conf_threshold=0.25):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def load_model(self):
        """Загрузка модели YOLO"""
        try:
            self.model = YOLO(self.model_name)
            return True, "Модель загружена успешно"
        except Exception as e:
            return False, str(e)

    def process_image(self, image_path, dock_x, dock_y, distance, output_dir):
        """Обработка одного изображения"""
        try:
            # Чтение изображения
            frame = cv2.imread(str(image_path))
            if frame is None:
                return False, "Не удалось прочитать изображение"

            height, width = frame.shape[:2]

            # Нормализация координат стыковочного порта
            norm_dock_x = dock_x / width
            norm_dock_y = dock_y / height

            # Информация для трекинга
            image_info = {
                'image_id': image_path.stem,
                'width': width,
                'height': height,
                'dock_x': dock_x,
                'dock_y': dock_y,
                'distance': distance,
                'timestamp': datetime.now().isoformat()
            }

            # Обнаружение ключевых точек
            results = self.model(frame)[0]

            # Сохранение результатов
            label_file = output_dir / f"{image_path.stem}.txt"

            with open(label_file, 'w') as f:
                if (results.keypoints is not None and
                        results.keypoints.xy is not None and
                        len(results.keypoints.xy) > 0):

                    for obj_idx, keypoints_data in enumerate(results.keypoints.xy):
                        if results.keypoints.conf is not None:
                            conf_data = results.keypoints.conf[obj_idx]
                        else:
                            conf_data = [1.0] * len(keypoints_data)

                        line_parts = [str(obj_idx)]

                        for i, ((x, y), conf) in enumerate(zip(keypoints_data, conf_data)):
                            if conf >= self.conf_threshold and x > 0 and y > 0:
                                norm_x = x / width
                                norm_y = y / height
                                line_parts.extend([f"{norm_x:.6f}", f"{norm_y:.6f}", f"{conf:.4f}"])
                            else:
                                line_parts.extend(['0.000000', '0.000000', '0.0000'])

                        f.write(' '.join(line_parts) + '\n')

                    # Сохраняем JSON для трекинга
                    track_file = output_dir / f"{image_path.stem}_track.json"
                    with open(track_file, 'w') as tf:
                        json.dump(image_info, tf)

                    return True, f"Обработано: {image_path.name} (найдены точки)"
                else:
                    # Только стыковочный порт
                    f.write(f"0 {norm_dock_x:.6f} {norm_dock_y:.6f} 1.0000 " + "0.0000 " * 48 + "\n")
                    return True, f"Обработано: {image_path.name} (только порт)"

        except Exception as e:
            return False, str(e)

    def process_all(self, loader, output_dir, classes_file, keypoint_names):
        """Обработка всех изображений"""
        if self.model is None:
            return False, "Модель не загружена"

        # Создаем выходные папки
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Сохраняем классы
        with open(classes_file, 'w') as f:
            for name in keypoint_names:
                f.write(f"{name}\n")

        # Статистика
        total = len(loader.df)
        processed = 0
        with_keypoints = 0
        errors = 0

        for idx, row in loader.df.iterrows():
            try:
                image_id = row['ImageID']

                # Поиск изображения
                img_path = loader.find_image(image_id)
                if img_path is None:
                    errors += 1
                    if self.progress_callback:
                        self.progress_callback(idx, total, f"Изображение {image_id} не найдено")
                    continue

                # Парсинг координат
                dock_x, dock_y = loader.parse_location(row['location'])
                if dock_x is None:
                    errors += 1
                    if self.progress_callback:
                        self.progress_callback(idx, total, f"Ошибка координат: {row['location']}")
                    continue

                # Обработка
                success, message = self.process_image(
                    img_path, dock_x, dock_y,
                    float(row['distance']), output_dir
                )

                if success:
                    processed += 1
                    if "найдены точки" in message:
                        with_keypoints += 1
                else:
                    errors += 1

                if self.progress_callback:
                    self.progress_callback(idx + 1, total, message)

            except Exception as e:
                errors += 1
                if self.progress_callback:
                    self.progress_callback(idx, total, f"Ошибка: {str(e)}")

        # Итоговая статистика
        stats = {
            'total': total,
            'processed': processed,
            'with_keypoints': with_keypoints,
            'errors': errors,
            'success_rate': f"{(processed / total) * 100:.1f}%" if total > 0 else "0%"
        }

        # Сохраняем статистику
        with open(output_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return True, stats