#!/usr/bin/env python3
"""
Простое решение: YOLOv8 для обнаружения космических аппаратов
Не требует ONNX модели
"""
from ultralytics import YOLO
import cv2
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt


class SimpleSpacecraftTracker:
    """
    Отслеживание космического аппарата с помощью YOLOv8
    """

    def __init__(self, model_name="yolov8m.pt"):  # используем стандартную модель
        print("=" * 70)
        print("🚀 ЗАПУСК ОТСЛЕЖИВАНИЯ КОСМИЧЕСКОГО АППАРАТА")
        print("=" * 70)

        # Загружаем модель YOLO (она скачается автоматически)
        print(f"\n📥 Загрузка модели {model_name}...")
        self.model = YOLO(model_name)
        print("✅ Модель загружена")

        # Для хранения истории
        self.track_history = []
        self.angular_velocities = []

    def detect_spacecraft(self, image_path):
        """
        Обнаружение объекта на изображении
        """
        # Запускаем детекцию
        results = self.model(image_path)

        # Визуализируем результаты
        for r in results:
            im_array = r.plot()
            cv2.imshow('Detection', im_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    def calculate_angle_from_bbox(self, bbox):
        """
        Вычисление угла по bounding box
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Угол наклона (аппроксимация)
        if width > 0 and height > 0:
            angle = math.atan2(height, width)
        else:
            angle = 0

        return angle

    def process_image_sequence(self, image_folder, num_images=10):
        """
        Обработка последовательности изображений
        """
        image_folder = Path(image_folder)
        image_files = sorted(image_folder.glob("*.jpg"))[:num_images]

        print(f"\n📸 Обработка {len(image_files)} изображений...")

        angles = []
        frames = []

        for i, img_path in enumerate(image_files):
            print(f"   Кадр {i + 1}/{len(image_files)}: {img_path.name}")

            # Детекция
            results = self.model(img_path)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Берём первый обнаруженный объект
                box = results[0].boxes.xyxy[0].cpu().numpy()

                # Вычисляем угол
                angle = self.calculate_angle_from_bbox(box)
                angles.append(angle)
                frames.append(i)

                print(f"      Угол: {angle:.4f} рад")
            else:
                print(f"      ⚠️ Объект не найден")

        return frames, angles

    def calculate_angular_velocity(self, frames, angles, fps=30):
        """
        Вычисление угловой скорости
        """
        if len(angles) < 2:
            print("❌ Недостаточно данных")
            return []

        velocities = []
        times = [f / fps for f in frames]

        for i in range(1, len(angles)):
            dt = times[i] - times[i - 1]
            if dt > 0:
                dtheta = angles[i] - angles[i - 1]

                # Нормализация
                if dtheta > math.pi:
                    dtheta -= 2 * math.pi
                elif dtheta < -math.pi:
                    dtheta += 2 * math.pi

                omega = dtheta / dt
                velocities.append({
                    'frame': frames[i],
                    'time': times[i],
                    'omega': omega,
                    'dtheta': dtheta
                })

        return velocities

    def plot_results(self, frames, angles, velocities):
        """
        Визуализация результатов
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # График углов
        times = [f / 30 for f in frames]
        ax1.plot(times, angles, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Угол (рад)')
        ax1.set_title('Изменение угла ориентации')
        ax1.grid(True, alpha=0.3)

        # График угловой скорости
        if velocities:
            v_times = [v['time'] for v in velocities]
            v_omegas = [v['omega'] for v in velocities]
            ax2.plot(v_times, v_omegas, 'ro-', linewidth=2, markersize=4)
            ax2.set_xlabel('Время (с)')
            ax2.set_ylabel('Угловая скорость (рад/с)')
            ax2.set_title('Угловая скорость')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        # Сохраняем
        output_path = Path.home() / "Desktop" / "tracking_results.png"
        plt.savefig(str(output_path), dpi=150)
        print(f"\n✅ График сохранён: {output_path}")
        plt.show()

        return output_path

    def print_statistics(self, velocities):
        """
        Вывод статистики
        """
        if not velocities:
            return

        omegas = [v['omega'] for v in velocities]

        print("\n" + "=" * 70)
        print("📊 СТАТИСТИКА УГЛОВОЙ СКОРОСТИ")
        print("=" * 70)
        print(f"   Измерений: {len(velocities)}")
        print(f"   Средняя ω: {np.mean(omegas):.6f} рад/с")
        print(f"   Медианная ω: {np.median(omegas):.6f} рад/с")
        print(f"   Станд. отклонение: {np.std(omegas):.6f} рад/с")
        print(f"   Макс |ω|: {np.max(np.abs(omegas)):.6f} рад/с")
        print(f"   Мин |ω|: {np.min(np.abs(omegas)):.6f} рад/с")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    import numpy as np

    # Создаём трекер
    tracker = SimpleSpacecraftTracker(model_name="yolov8m.pt")

    # Путь к вашим изображениям
    image_folder = "/Users/darya/Desktop/archive/train"

    # Обрабатываем первые 10 изображений
    frames, angles = tracker.process_image_sequence(image_folder, num_images=10)

    if len(angles) >= 2:
        # Вычисляем угловую скорость
        velocities = tracker.calculate_angular_velocity(frames, angles)

        # Статистика
        tracker.print_statistics(velocities)

        # Визуализация
        tracker.plot_results(frames, angles, velocities)
    else:
        print("❌ Недостаточно обнаружений для расчёта")