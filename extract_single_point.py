#!/usr/bin/env python3
"""
Извлекает ТОЛЬКО ПЕРВУЮ ТОЧКУ из файлов разметки
Создаёт новый датасет для обучения на стыковочном порте
"""
from pathlib import Path
import shutil
import random
import yaml


def extract_first_point(input_file, output_file):
    """
    Извлекает только первую точку из файла разметки
    Формат: class_id x1 y1 conf1
    """
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:  # минимум class_id + x1 + y1 + conf1
                class_id = parts[0]
                x1 = parts[1]
                y1 = parts[2]
                conf1 = parts[3]

                # Записываем только первую точку
                new_line = f"{class_id} {x1} {y1} {conf1}"
                valid_lines.append(new_line)

        if valid_lines:
            with open(output_file, 'w') as f:
                f.write('\n'.join(valid_lines))
            return True
        return False

    except Exception as e:
        print(f"   Ошибка в {input_file.name}: {e}")
        return False


def create_dataset():
    """
    Создаёт новый датасет с одной точкой
    """
    source_images = Path("/Users/darya/Desktop/archive/train")
    source_labels = Path("/Users/darya/Desktop/iss_results")
    target_base = Path("/Users/darya/Desktop/iss_single_point_dataset")

    print("=" * 70)
    print("🔄 СОЗДАНИЕ ДАТАСЕТА С ОДНОЙ ТОЧКОЙ")
    print("=" * 70)

    # Создаём структуру
    dirs = [
        target_base / 'images' / 'train',
        target_base / 'images' / 'val',
        target_base / 'labels' / 'train',
        target_base / 'labels' / 'val',
    ]

    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        print(f"📁 Создана папка: {d}")

    # Находим все пары изображение-разметка
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(source_images.glob(ext))

    print(f"\n📸 Найдено изображений: {len(image_files)}")

    valid_pairs = []
    for img_file in image_files:
        label_file = source_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))

    print(f"✅ Найдено пар: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("❌ Нет пар изображение-разметка!")
        return None

    # Перемешиваем
    random.shuffle(valid_pairs)

    # Разделяем на train (80%) и val (20%)
    split_idx = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"\n📊 Разделение:")
    print(f"   Train: {len(train_pairs)}")
    print(f"   Val: {len(val_pairs)}")

    # Обрабатываем train
    print("\n📋 Обработка train...")
    converted_train = 0
    for img_file, label_file in train_pairs:
        # Копируем изображение
        dst_img = target_base / 'images' / 'train' / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # Извлекаем первую точку
        dst_label = target_base / 'labels' / 'train' / label_file.name
        if extract_first_point(label_file, dst_label):
            converted_train += 1

    # Обрабатываем val
    print("📋 Обработка val...")
    converted_val = 0
    for img_file, label_file in val_pairs:
        # Копируем изображение
        dst_img = target_base / 'images' / 'val' / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # Извлекаем первую точку
        dst_label = target_base / 'labels' / 'val' / label_file.name
        if extract_first_point(label_file, dst_label):
            converted_val += 1

    print(f"\n✅ Создано файлов разметки:")
    print(f"   Train: {converted_train}")
    print(f"   Val: {converted_val}")

    return target_base


def create_yaml(dataset_path):
    """
    Создаёт YAML конфиг
    """
    yaml_content = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [1, 3],  # 1 точка, [x, y, visibility]
        'names': {
            0: 'docking_port'
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

    # Проверяем val
    val_labels = list((dataset_path / 'labels' / 'val').glob("*.txt"))
    val_images = list((dataset_path / 'images' / 'val').glob("*.*"))

    print(f"\n📊 Validation:")
    print(f"   Изображений: {len(val_images)}")
    print(f"   Файлов разметки: {len(val_labels)}")

    # Показываем пример
    if train_labels:
        print("\n📄 Пример новой разметки:")
        with open(train_labels[0], 'r') as f:
            content = f.read().strip()
            print(f"   {train_labels[0].name}: {content}")

    return len(train_labels) > 0 and len(val_labels) > 0


def create_training_script(dataset_path):
    """
    Создаёт скрипт для обучения
    """
    script_path = dataset_path / 'train.py'

    script_content = f'''#!/usr/bin/env python3
"""
Обучение модели на датасете с одной точкой
"""
from ultralytics import YOLO

# Загружаем модель
model = YOLO('yolov8m-pose.pt')

# Обучаем
results = model.train(
    data='{dataset_path}/dataset.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    device='mps',  # для Mac, можно 'cpu' или 'cuda'
    patience=20,
    save=True,
    project='/Users/darya/Desktop/iss_trained_model',
    name='docking_port_detector',
    exist_ok=True
)

print("✅ Обучение завершено!")
print(f"Модель сохранена в {{results.save_dir}}")
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"\n📝 Создан скрипт обучения: {script_path}")
    return script_path


def main():
    # 1. Создаём датасет
    dataset_path = create_dataset()
    if not dataset_path:
        return

    # 2. Проверяем
    if not verify_dataset(dataset_path):
        print("\n❌ Проблемы с датасетом!")
        return

    # 3. Создаём YAML
    yaml_path = create_yaml(dataset_path)

    # 4. Создаём скрипт обучения
    train_script = create_training_script(dataset_path)

    print("\n" + "=" * 70)
    print("✅ ВСЁ ГОТОВО К ОБУЧЕНИЮ!")
    print("=" * 70)
    print(f"\n📁 Датасет: {dataset_path}")
    print(f"📄 Конфиг: {yaml_path}")
    print(f"📝 Скрипт: {train_script}")
    print(f"\n🚀 Запустите обучение командой:")
    print(f"   python {train_script}")
    print("\n💡 Или вручную:")
    print(f"   cd {dataset_path}")
    print(
        "   python -c \"from ultralytics import YOLO; YOLO('yolov8m-pose.pt').train(data='dataset.yaml', epochs=30)\"")


if __name__ == "__main__":
    main()