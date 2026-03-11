"""
Визуализация результатов для UI
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io


class ResultVisualizer:
    """Визуализатор для отображения в UI"""

    def __init__(self, keypoint_names, colors):
        self.keypoint_names = keypoint_names
        self.colors = colors

    def load_keypoints_from_file(self, label_file):
        """Загрузка ключевых точек из файла разметки"""
        if not Path(label_file).exists():
            return None

        keypoints = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    # Первое число - class_id, остальные - координаты
                    obj_keypoints = []
                    for i in range(1, len(parts), 3):
                        if i + 2 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i + 1])
                            conf = float(parts[i + 2])
                            if x > 0 and y > 0 and conf > 0:
                                obj_keypoints.append((x, y, conf))
                    if obj_keypoints:
                        keypoints.append(obj_keypoints)
        return keypoints if keypoints else None

    def draw_on_image(self, image_path, keypoints_data=None):
        """Рисует ключевые точки на изображении"""
        # Чтение изображения
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Если переданы данные ключевых точек
        if keypoints_data:
            for obj_keypoints in keypoints_data:
                for i, (x, y, conf) in enumerate(obj_keypoints):
                    # Преобразование из нормализованных координат
                    px = int(x * w)
                    py = int(y * h)

                    # Рисуем круг
                    color = self.colors[i % len(self.colors)]
                    cv2.circle(img_rgb, (px, py), 5, color, -1)

                    # Рисуем номер точки
                    cv2.putText(img_rgb, str(i), (px + 5, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return img_rgb

    def get_photoimage(self, image_path, keypoints_data=None, max_size=(800, 600)):
        """Возвращает ImageTk.PhotoImage для отображения в Tkinter"""
        img = self.draw_on_image(image_path, keypoints_data)
        if img is None:
            return None

        # Изменение размера
        pil_img = Image.fromarray(img)
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(pil_img)

    def create_preview_grid(self, image_paths, output_path, max_images=9):
        """Создание сетки предпросмотра"""
        n = min(len(image_paths), max_images)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes]
        axes = [ax for row in axes for ax in row]

        for i in range(n):
            if i < len(image_paths):
                img = cv2.imread(str(image_paths[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(f"Frame {image_paths[i].stem}")
                axes[i].axis('off')

        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path