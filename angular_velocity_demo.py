#!/usr/bin/env python3
"""
Демонстрация расчёта угловой скорости МКС
Работает даже если есть только точка 0 (стыковочный порт)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import cv2

# Настройка matplotlib для отображения
import matplotlib

matplotlib.use('TkAgg')


class AngularVelocityDemo:
    def __init__(self, results_path="/Users/darya/Desktop/iss_results"):
        self.results_path = Path(results_path)

        # Загружаем файлы разметки
        self.label_files = sorted(self.results_path.glob("*.txt"))
        self.label_files = [f for f in self.label_files if f.stem.isdigit()]

        print(f"✅ Найдено {len(self.label_files)} файлов разметки")

        # Параметры
        self.fps = 30  # предполагаемая частота кадров

    def load_point_0(self, label_file):
        """Загрузка только точки 0 (стыковочный порт)"""
        try:
            with open(label_file, 'r') as f:
                line = f.readline().strip()

            parts = line.split()
            if len(parts) >= 4:  # минимум: class_id + x + y + conf
                x = float(parts[1])
                y = float(parts[2])
                conf = float(parts[3])
                if x > 0 and y > 0 and conf > 0.1:
                    return (x, y, conf)
            return None
        except Exception as e:
            return None

    def calculate_angle_from_center(self, point):
        """
        Расчёт угла точки относительно центра изображения
        Используется, когда есть только одна точка
        """
        # Предполагаем, что центр изображения в (0.5, 0.5)
        center_x, center_y = 0.5, 0.5

        dx = point[0] - center_x
        dy = point[1] - center_y

        angle = math.atan2(dy, dx)
        return angle

    def calculate_angular_velocity_single_point(self, start_idx=0, num_frames=100):
        """
        Расчёт угловой скорости по одной точке (относительно центра)
        """
        if len(self.label_files) < start_idx + num_frames:
            num_frames = len(self.label_files) - start_idx

        print(f"\n🔄 Расчёт угловой скорости на {num_frames} кадрах (по одной точке)...")

        angles = []
        valid_indices = []

        for i in range(start_idx, start_idx + num_frames):
            if i >= len(self.label_files):
                break

            label_file = self.label_files[i]
            point = self.load_point_0(label_file)

            if point:
                angle = self.calculate_angle_from_center(point)
                angles.append(angle)
                valid_indices.append(i)

        if len(angles) < 2:
            print("❌ Недостаточно данных для расчёта")
            return None, None

        # Вычисляем угловую скорость
        times = [i / self.fps for i in valid_indices]
        velocities = []
        velocity_times = []

        for j in range(1, len(angles)):
            dt = times[j] - times[j - 1]
            if dt > 0:
                dtheta = angles[j] - angles[j - 1]
                # Нормализация
                if dtheta > math.pi:
                    dtheta -= 2 * math.pi
                elif dtheta < -math.pi:
                    dtheta += 2 * math.pi

                omega = dtheta / dt
                velocities.append(omega)
                velocity_times.append(times[j])

        return {
            'times': times,
            'angles': angles,
            'velocity_times': velocity_times,
            'velocities': velocities
        }

    def plot_results(self, data):
        """Визуализация результатов"""
        if not data:
            print("❌ Нет данных для визуализации")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # График 1: Положение точки
        ax1.plot(data['times'], data['angles'],
                 'bo-', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_xlabel('Время (секунды)', fontsize=12)
        ax1.set_ylabel('Угол (радианы)', fontsize=12)
        ax1.set_title('Изменение угла стыковочного порта\n(относительно центра изображения)',
                      fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([min(data['times']), max(data['times'])])

        # График 2: Угловая скорость
        ax2.plot(data['velocity_times'], data['velocities'],
                 'ro-', linewidth=2, markersize=3, alpha=0.7)
        ax2.set_xlabel('Время (секунды)', fontsize=12)
        ax2.set_ylabel('Угловая скорость (рад/с)', fontsize=12)
        ax2.set_title('Угловая скорость МКС (по движению стыковочного порта)',
                      fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_xlim([min(data['times']), max(data['times'])])

        plt.tight_layout()

        # Сохраняем
        png_path = self.results_path / 'angular_velocity_single_point.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ График сохранён: {png_path}")

        plt.show()

        return png_path

    def print_statistics(self, data):
        """Вывод статистики"""
        if not data or len(data['velocities']) == 0:
            return

        velocities = np.abs(data['velocities'])

        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА УГЛОВОЙ СКОРОСТИ")
        print("=" * 60)
        print(f"\n📈 По стыковочному порту (относительно центра):")
        print(f"   Средняя |ω|: {np.mean(velocities):.6f} рад/с")
        print(f"   Медианная |ω|: {np.median(velocities):.6f} рад/с")
        print(f"   Макс |ω|: {np.max(velocities):.6f} рад/с")
        print(f"   Мин |ω|: {np.min(velocities):.6f} рад/с")
        print(f"   Станд. отклонение: {np.std(velocities):.6f} рад/с")

        # Дополнительная информация
        print(f"\n📊 Дополнительно:")
        print(f"   Всего кадров с данными: {len(data['angles'])}")
        print(f"   Диапазон углов: [{min(data['angles']):.4f}, {max(data['angles']):.4f}] рад")
        print(f"   Полное изменение угла: {max(data['angles']) - min(data['angles']):.4f} рад")
        print(f"   Время наблюдения: {max(data['times']) - min(data['times']):.2f} с")

    def run_demo(self, num_frames=200):
        """Запуск демонстрации"""
        print("=" * 70)
        print("🚀 ДЕМОНСТРАЦИЯ РАСЧЁТА УГЛОВОЙ СКОРОСТИ МКС")
        print("=" * 70)
        print(f"\n📁 Данные из: {self.results_path}")
        print(f"📸 Файлов разметки: {len(self.label_files)}")
        print("\n⚠️  Внимание: Используется только точка 0 (стыковочный порт)")
        print("   Для более точного анализа нужны все 12 точек\n")

        # Расчёт
        data = self.calculate_angular_velocity_single_point(
            start_idx=0,
            num_frames=min(num_frames, len(self.label_files))
        )

        if data:
            # Статистика
            self.print_statistics(data)

            # Визуализация
            self.plot_results(data)

            # Создание видео (опционально)
            self.create_simple_video(data, num_frames=min(100, len(self.label_files)))

        print("\n" + "=" * 70)
        print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 70)
        print("\n📌 Интерпретация:")
        print("   • Положительная скорость = вращение по часовой")
        print("   • Отрицательная скорость = вращение против часовой")
        print("   • Близость к нулю = станция стабилизирована")

        return data

    def create_simple_video(self, data, num_frames=100, output_path="angular_velocity_demo.mp4"):
        """Создание простого видео с визуализацией"""
        print("\n🎥 Создание демо-видео...")

        frames_to_process = min(num_frames, len(self.label_files))

        # Настройки видео
        width, height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.results_path / output_path),
                              fourcc, 10.0, (width, height))

        for i in range(frames_to_process):
            # Создаём кадр
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame.fill(255)  # белый фон

            # Загружаем точку
            label_file = self.label_files[i]
            point = self.load_point_0(label_file)

            # Отображаем информацию
            y_pos = 50
            cv2.putText(frame, f"Кадр: {label_file.stem}", (50, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if point:
                # Координаты в пикселях
                px = int(point[0] * width)
                py = int(point[1] * height)

                # Рисуем точку
                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(frame, "Стыковочный порт", (px + 20, py - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Угол относительно центра
                center_x, center_y = width // 2, height // 2
                cv2.line(frame, (center_x, center_y), (px, py), (255, 0, 0), 2)

                angle = self.calculate_angle_from_center(point)
                cv2.putText(frame, f"Угол: {angle:.3f} рад", (50, y_pos + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Рисуем центр
            cv2.circle(frame, (width // 2, height // 2), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Центр", (width // 2 + 20, height // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)

        out.release()
        print(f"✅ Видео сохранено: {self.results_path / output_path}")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    demo = AngularVelocityDemo(
        results_path="/Users/darya/Desktop/iss_results"
    )

    # Запускаем на 200 кадрах
    data = demo.run_demo(num_frames=200)