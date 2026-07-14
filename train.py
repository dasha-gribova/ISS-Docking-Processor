import os
import shutil
import random
from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class TrainOnCurrentData:
    """
    Обучение модели на существующих данных из iss_results
    С КОНВЕРТАЦИЕЙ ФОРМАТА
    """

    def __init__(self,
                 results_path="/Users/darya/Desktop/iss_results",
                 images_path="/Users/darya/Desktop/archive/train",
                 output_path="/Users/darya/Desktop/iss_trained_model"):

        self.results_path = Path(results_path)
        self.images_path = Path(images_path)
        self.output_path = Path(output_path)

        # Создаём выходную папку
        self.output_path.mkdir(exist_ok=True)

        # Временная папка для датасета
        self.dataset_path = self.output_path / 'dataset'
        self.dataset_path.mkdir(exist_ok=True)

        print(f"Результаты обучения: {self.output_path}")
        print(f"Временный датасет: {self.dataset_path}")

    def convert_label_format(self, input_label_file, output_label_file):
        """
        Конвертирует формат разметки в правильный для YOLO
        Из формата: [class_id x1 y1 conf1 x2 y2 conf2 ...]
        В формат:   class_id x1 y1 conf1 x2 y2 conf2 ... (без скобок, все числа)
        """
        try:
            with open(input_label_file, 'r') as f:
                lines = f.readlines()

            with open(output_label_file, 'w') as f_out:
                for line in lines:
                    # Очищаем строку от лишних символов
                    line = line.strip()
                    if not line:
                        continue

                    # Разделяем по пробелам
                    parts = line.split()

                    # Проверяем, что есть как минимум class_id + 3 координаты
                    if len(parts) >= 4:
                        # Записываем в новом формате
                        f_out.write(' '.join(parts) + '\n')
                        return True

            return False
        except Exception as e:
            print(f"   Ошибка конвертации {input_label_file}: {e}")
            return False

    def prepare_dataset(self, train_ratio=0.8):
        """
        Подготовка датасета из существующих файлов с конвертацией формата
        """
        print("\n" + "=" * 60)
        print("ПОДГОТОВКА ДАТАСЕТА")
        print("=" * 60)

        # Получаем все файлы разметки
        label_files = sorted(self.results_path.glob("*.txt"))
        label_files = [f for f in label_files if f.stem.isdigit()]

        print(f"Найдено {len(label_files)} файлов разметки")

        # Проверяем наличие соответствующих изображений
        valid_pairs = []
        for label_file in label_files:
            # Ищем изображение
            img_found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = self.images_path / f"{label_file.stem}{ext}"
                if img_path.exists():
                    valid_pairs.append((label_file, img_path))
                    img_found = True
                    break

        print(f"Найдено {len(valid_pairs)} пар изображение-разметка")

        if len(valid_pairs) < 10:
            print("Слишком мало данных для обучения")
            return None

        # Перемешиваем
        random.shuffle(valid_pairs)

        # Разделяем на train/val
        train_count = int(len(valid_pairs) * train_ratio)
        train_pairs = valid_pairs[:train_count]
        val_pairs = valid_pairs[train_count:]

        print(f"\nРазделение датасета:")
        print(f"   Train: {len(train_pairs)} изображений")
        print(f"   Val: {len(val_pairs)} изображений")

        # Создаём структуру папок
        train_img_dir = self.dataset_path / 'images' / 'train'
        train_label_dir = self.dataset_path / 'labels' / 'train'
        val_img_dir = self.dataset_path / 'images' / 'val'
        val_label_dir = self.dataset_path / 'labels' / 'val'

        for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        # Копируем и конвертируем файлы
        print("\nКопирование и конвертация файлов...")

        converted_count = 0
        error_count = 0

        # Train
        for idx, (label_file, img_file) in enumerate(train_pairs):
            # Копируем изображение
            dst_img = train_img_dir / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)

            # Конвертируем и сохраняем разметку
            dst_label = train_label_dir / label_file.name
            if self.convert_label_format(label_file, dst_label):
                converted_count += 1

            if (idx + 1) % 500 == 0:
                print(f"   ...обработано {idx + 1} файлов")

        # Validation
        for idx, (label_file, img_file) in enumerate(val_pairs):
            # Копируем изображение
            dst_img = val_img_dir / img_file.name
            if not dst_img.exists():
                shutil.copy2(img_file, dst_img)

            # Конвертируем и сохраняем разметку
            dst_label = val_label_dir / label_file.name
            if self.convert_label_format(label_file, dst_label):
                converted_count += 1

        print(f"Конвертировано {converted_count} файлов разметки")
        print("Датасет готов!")

        # Показываем пример конвертированного файла
        if train_pairs:
            example_file = train_label_dir / train_pairs[0][0].name
            if example_file.exists():
                print(f"\nПример конвертированного файла {example_file.name}:")
                with open(example_file, 'r') as f:
                    for i, line in enumerate(f.readlines()[:3]):
                        print(f"   {line.strip()}")

        return {
            'train': train_pairs,
            'val': val_pairs,
            'train_count': len(train_pairs),
            'val_count': len(val_pairs)
        }

    def create_dataset_yaml(self):
        """
        Создание YAML конфига для обучения
        """
        # Сначала проверяем, сколько точек в файлах
        sample_label_dir = self.dataset_path / 'labels' / 'train'
        sample_files = list(sample_label_dir.glob("*.txt"))

        num_keypoints = 1  # по умолчанию 1 точка (стыковочный порт)
        if sample_files:
            try:
                with open(sample_files[0], 'r') as f:
                    first_line = f.readline().strip()
                    parts = first_line.split()
                    if len(parts) > 1:
                        # (len(parts) - 1) / 3 = количество точек
                        num_keypoints = (len(parts) - 1) // 3
                        print(f"\nОбнаружено {num_keypoints} ключевых точек в файлах")
            except:
                pass

        yaml_content = {
            'path': str(self.dataset_path),  # корневая папка
            'train': 'images/train',  # папка с train изображениями
            'val': 'images/val',  # папка с val изображениями

            # Ключевые точки
            'kpt_shape': [num_keypoints, 3],  # [количество точек, 3 координаты]

            # Названия классов
            'names': {
                0: 'ISS'
            }
        }

        yaml_path = self.dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\nСоздан конфиг: {yaml_path}")
        print("\nСодержимое:")
        with open(yaml_path, 'r') as f:
            print(f.read())

        return yaml_path

    def verify_dataset(self):
        """
        Проверка датасета перед обучением
        """
        print("\n" + "=" * 60)
        print("ПРОВЕРКА ДАТАСЕТА")
        print("=" * 60)

        train_labels_dir = self.dataset_path / 'labels' / 'train'
        train_images_dir = self.dataset_path / 'images' / 'train'

        label_files = list(train_labels_dir.glob("*.txt"))
        image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))

        print(f" Train:")
        print(f"   Изображений: {len(image_files)}")
        print(f"   Файлов разметки: {len(label_files)}")

        if len(label_files) != len(image_files):
            print(f"Количество изображений и файлов разметки не совпадает!")

        # Проверяем первые несколько файлов разметки
        print("\nПроверка формата файлов разметки:")
        for i, label_file in enumerate(label_files[:5]):
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                num_points = (len(parts) - 1) // 3
                print(f"   {label_file.name}: {len(parts)} колонок, {num_points} точек")

        return len(label_files) > 0 and len(image_files) > 0

    def train_model(self, yaml_path, epochs=30, imgsz=640, batch_size=16):
        """
        Обучение модели
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ОБУЧЕНИЯ")
        print("=" * 60)

        # Проверяем доступное устройство
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Используется MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Используется CUDA")
        else:
            device = 'cpu'
            print("Используется CPU (будет медленно)")

        print(f"\nПараметры обучения:")
        print(f"   Эпох: {epochs}")
        print(f"   Размер изображений: {imgsz}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Устройство: {device}")

        # Загружаем предобученную модель
        print("\nЗагрузка модели YOLOv8...")
        model = YOLO('yolov8m-pose.pt')

        # Обучаем
        print("\nНачало обучения...\n")

        try:
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                workers=4,
                patience=20,
                save=True,
                project=str(self.output_path),
                name='iss_detector',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                lr0=0.001,
                weight_decay=0.0005,
                warmup_epochs=3,
                cos_lr=True,
                verbose=True
            )

            print("\nОбучение завершено!")
            print(f"Модель сохранена в: {self.output_path / 'iss_detector'}")

            return results

        except Exception as e:
            print(f"\n Ошибка обучения: {e}")
            return None

    def run_full_training(self, epochs=30):
        """
        Запуск полного цикла обучения
        """
        print("\n" + "=" * 70)
        print("ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ НА ТЕКУЩИХ ДАННЫХ")
        print("=" * 70)

        # 1. Подготовка датасета
        dataset_info = self.prepare_dataset()
        if not dataset_info:
            print("Не удалось подготовить датасет")
            return

        # 2. Проверка датасета
        if not self.verify_dataset():
            print("Проблемы с датасетом")
            return

        # 3. Создание YAML
        yaml_path = self.create_dataset_yaml()

        # 4. Обучение
        results = self.train_model(yaml_path, epochs=epochs)

        if results:
            print("\n" + "=" * 70)
            print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print("=" * 70)
            print(f"\nМодель сохранена в: {self.output_path / 'iss_detector'}")

        return 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Обучение на текущих данных')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Количество эпох обучения (по умолчанию 30)')
    parser.add_argument('--quick', action='store_true',
                        help='Быстрое обучение (5 эпох для теста)')

    args = parser.parse_args()

    # Создаём экземпляр класса
    trainer = TrainOnCurrentData(
        results_path="/Users/darya/Desktop/iss_results",
        images_path="/Users/darya/Desktop/archive/train",
        output_path="/Users/darya/Desktop/iss_trained_model"
    )

    if args.quick:
        print("\nЗАПУСК БЫСТРОГО ОБУЧЕНИЯ (5 ЭПОХ)")
        trainer.run_full_training(epochs=5)
    else:
        trainer.run_full_training(epochs=args.epochs)
