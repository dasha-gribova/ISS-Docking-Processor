#!/usr/bin/env python3
"""
СОЗДАНИЕ ПРАВИЛЬНОГО ДАТАСЕТА для обучения YOLO
Конвертирует ваши файлы в корректный формат
"""
from pathlib import Path
import shutil
import random
import yaml
import numpy as np


def verify_and_convert_labels(input_dir, output_dir, num_keypoints=1):
    """
    Проверяет и конвертирует файлы разметки в правильный формат
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    label_files = list(input_dir.glob("*.txt"))
    print(f"📄 Найдено {len(label_files)} файлов разметки")

    valid_count = 0
    invalid_count = 0

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            valid_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Разбираем строку
                parts = line.split()
                if len(parts) < 4:
                    continue

                # Проверяем, что все координаты - числа от 0 до 1
                all_valid = True
                for i, val in enumerate(parts):
                    try:
                        num = float(val)
                        # Для координат (x, y) проверяем диапазон
                        if i % 3 == 1 or i % 3 == 2:  # x или y
                            if num < 0 or num > 1:
                                all_valid = False
                                break
                    except:
                        all_valid = False
                        break

                if all_valid:
                    valid_lines.append(' '.join(parts))

            if valid_lines:
                # Сохраняем только валидные строки
                with open(output_dir / label_file.name, 'w') as f:
                    f.write('\n'.join(valid_lines))
                valid_count += 1
            else:
                invalid_count += 1

        except Exception as e:
            print(f"   Ошибка в {label_file.name}: {e}")
            invalid_count += 1

    print(f"   ✅ Валидных: {valid_count}")
    print(f"   ❌ Невалидных: {invalid_count}")

    return valid_count


def create_dataset_structure():
    """
    Создаёт правильную структуру датасета
    """
    base_dir = Path("/Users/darya/Desktop/iss_training_dataset")

    # Создаём структуру
    dirs = [
        base_dir / 'images' / 'train',
        base_dir / 'images' / 'val',
        base_dir / 'labels' / 'train',
        base_dir / 'labels' / 'val',
    ]

    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        print(f"📁 Создана папка: {d}")

    return base_dir


def copy_images_and_labels():
    """
    Копирует изображения и разметку в правильную структуру
    """
    source_images = Path("/Users/darya/Desktop/archive/train")
    source_labels = Path("/Users/darya/Desktop/iss_results")
    target_base = Path("/Users/darya/Desktop/iss_training_dataset")

    # Получаем все файлы изображений
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(source_images.glob(ext))

    print(f"📸 Найдено {len(image_files)} изображений")

    # Фильтруем только те, для которых есть разметка
    valid_pairs = []
    for img_file in image_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))

    print(f"✅ Найдено {len(valid_pairs)} пар изображение-разметка")

    # Перемешиваем
    random.shuffle(valid_pairs)

    # Разделяем на train (80%) и val (20%)
    split_idx = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"\n📊 Разделение:")
    print(f"   Train: {len(train_pairs)} изображений")
    print(f"   Val: {len(val_pairs)} изображений")

    # Копируем train
    print("\n📋 Копирование train...")
    for img_file, label_file in train_pairs:
        # Копируем изображение
        dst_img = target_base / 'images' / 'train' / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # Копируем разметку
        dst_label = target_base / 'labels' / 'train' / label_file.name
        if not dst_label.exists():
            shutil.copy2(label_file, dst_label)

    # Копируем val
    print("📋 Копирование val...")
    for img_file, label_file in val_pairs:
        # Копируем изображение
        dst_img = target_base / 'images' / 'val' / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # Копируем разметку
        dst_label = target_base / 'labels' / 'val' / label_file.name
        if not dst_label.exists():
            shutil.copy2(label_file, dst_label)

    return target_base, len(train_pairs), len(val_pairs)


def create_yaml(dataset_path, num_keypoints=1):
    """
    Создаёт YAML конфиг для обучения
    """
    yaml_content = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [num_keypoints, 3],  # [количество точек, 3 координаты]
        'names': {
            0: 'ISS'
        }
    }

    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n✅ Создан конфиг: {yaml_path}")
    print("\n📄 Содержимое:")
    with open(yaml_path, 'r') as f:
        print(f.read())

    return yaml_path


def verify_dataset(dataset_path):
    """
    Проверяет созданный датасет
    """
    print("\n" + "=" * 60)
    print("🔍 ПРОВЕРКА ДАТАСЕТА")
    print("=" * 60)

    # Проверяем train
    train_labels = list((dataset_path / 'labels' / 'train').glob("*.txt"))
    train_images = list((dataset_path / 'images' / 'train').glob("*.*"))

    print(f"\n📊 Train:")
    print(f"   Изображений: {len(train_images)}")
    print(f"   Файлов разметки: {len(train_labels)}")

    if len(train_labels) != len(train_images):
        print("⚠️  Количество не совпадает!")

    # Проверяем val
    val_labels = list((dataset_path / 'labels' / 'val').glob("*.txt"))
    val_images = list((dataset_path / 'images' / 'val').glob("*.*"))

    print(f"\n📊 Validation:")
    print(f"   Изображений: {len(val_images)}")
    print(f"   Файлов разметки: {len(val_labels)}")

    # Проверяем несколько файлов разметки
    print("\n📄 Примеры файлов разметки:")
    for i, label_file in enumerate(train_labels[:3]):
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            num_cols = len(parts)
            num_points = (num_cols - 1) // 3 if num_cols > 1 else 0
            print(f"   {label_file.name}: {num_cols} колонок, {num_points} точек")
            print(f"      Пример: {first_line[:100]}...")

    return len(train_labels) > 0 and len(val_labels) > 0


def main():
    print("=" * 70)
    print("🛠  СОЗДАНИЕ КОРРЕКТНОГО ДАТАСЕТА ДЛЯ YOLO")
    print("=" * 70)

    # 1. Создаём структуру папок
    dataset_path = create_dataset_structure()

    # 2. Копируем файлы
    dataset_path, train_count, val_count = copy_images_and_labels()

    # 3. Проверяем и конвертируем train
    print("\n" + "=" * 60)
    print("🔧 ПРОВЕРКА TRAIN РАЗМЕТКИ")
    train_labels_dir = dataset_path / 'labels' / 'train'
    valid_train = verify_and_convert_labels(
        train_labels_dir,
        train_labels_dir  # конвертируем на месте
    )

    # 4. Проверяем и конвертируем val
    print("\n" + "=" * 60)
    print("🔧 ПРОВЕРКА VAL РАЗМЕТКИ")
    val_labels_dir = dataset_path / 'labels' / 'val'
    valid_val = verify_and_convert_labels(
        val_labels_dir,
        val_labels_dir  # конвертируем на месте
    )

    # 5. Проверяем результат
    if valid_train == 0 or valid_val == 0:
        print("\n❌ НЕТ ВАЛИДНЫХ ФАЙЛОВ РАЗМЕТКИ!")
        print("\n💡 Решение: Проверьте содержимое файлов вручную:")
        print("   ls -la /Users/darya/Desktop/iss_results/ | head -5")
        print("   cat /Users/darya/Desktop/iss_results/0.txt")
        return

    # 6. Создаём YAML
    yaml_path = create_yaml(dataset_path)

    # 7. Финальная проверка
    if verify_dataset(dataset_path):
        print("\n" + "=" * 70)
        print("✅ ДАТАСЕТ ГОТОВ К ОБУЧЕНИЮ!")
        print("=" * 70)
        print(f"\n📁 Путь к датасету: {dataset_path}")
        print(f"📄 Конфиг: {yaml_path}")
        print(f"\n🚀 Для обучения выполните:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('yolov8m-pose.pt')")
        print(f"   model.train(data='{yaml_path}', epochs=30, imgsz=640)")
    else:
        print("\n❌ ПРОБЛЕМЫ С ДАТАСЕТОМ!")


if __name__ == "__main__":
    main()