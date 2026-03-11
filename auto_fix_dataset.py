#!/usr/bin/env python3
"""
АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ ДАТАСЕТА
Скрипт сам определяет нужный формат и преобразует файлы
"""
from pathlib import Path
import shutil
import yaml


def analyze_label_file(file_path):
    """
    Анализирует один файл разметки и возвращает количество колонок и точек
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return 0, 0

            parts = first_line.split()
            num_cols = len(parts)
            # Для pose estimation: class_id + 3*N точек
            if num_cols >= 4:
                num_points = (num_cols - 1) // 3
                return num_cols, num_points
            return num_cols, 0
    except:
        return 0, 0


def convert_to_format(input_file, output_file, target_points):
    """
    Конвертирует файл в нужное количество точек
    - Если точек太少 - дублирует последнюю
    - Если точек太多 - обрезает
    """
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            class_id = parts[0]
            current_points = (len(parts) - 1) // 3

            # Берем существующие точки
            points = []
            for i in range(current_points):
                idx = 1 + i * 3
                if idx + 2 < len(parts):
                    x = parts[idx]
                    y = parts[idx + 1]
                    conf = parts[idx + 2]
                    points.append((x, y, conf))

            # Если точек太少 - дублируем последнюю
            while len(points) < target_points and points:
                last = points[-1]
                # Добавляем со смещением, чтобы не было точно таких же
                x = str(float(last[0]) + 0.001)
                y = str(float(last[1]) - 0.001)
                points.append((x, y, last[2]))

            # Если точек太多 - обрезаем
            if len(points) > target_points:
                points = points[:target_points]

            # Формируем новую строку
            new_parts = [class_id]
            for x, y, conf in points:
                new_parts.extend([x, y, conf])

            new_lines.append(' '.join(new_parts))

        if new_lines:
            with open(output_file, 'w') as f:
                f.write('\n'.join(new_lines))
            return True
        return False

    except Exception as e:
        print(f"   Ошибка в {input_file.name}: {e}")
        return False


def fix_dataset():
    """
    Главная функция исправления датасета
    """
    source_path = Path("/Users/darya/Desktop/iss_single_point_dataset")
    fixed_path = Path("/Users/darya/Desktop/iss_dataset_fixed_final")

    print("=" * 70)
    print("🔧 АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ ДАТАСЕТА")
    print("=" * 70)

    if not source_path.exists():
        print(f"❌ Папка не найдена: {source_path}")
        return None

    # Анализируем один файл, чтобы понять формат
    sample_labels = list((source_path / 'labels' / 'train').glob("*.txt"))
    if not sample_labels:
        print("❌ Нет файлов разметки для анализа")
        return None

    sample_file = sample_labels[0]
    num_cols, num_points = analyze_label_file(sample_file)
    print(f"\n📊 Анализ образца {sample_file.name}:")
    print(f"   Колонок: {num_cols}")
    print(f"   Точек: {num_points}")

    # Определяем целевое количество точек
    # Пробуем угадать: если 4 колонки (1 + 1*3) -> 1 точка
    # Если 7 колонок (1 + 2*3) -> 2 точки, и т.д.
    if num_cols >= 4:
        target_points = (num_cols - 1) // 3
        print(f"🎯 Целевое количество точек: {target_points}")
    else:
        print("❌ Не могу определить формат")
        return None

    # Создаём структуру для исправленного датасета
    dirs = [
        fixed_path / 'images' / 'train',
        fixed_path / 'images' / 'val',
        fixed_path / 'labels' / 'train',
        fixed_path / 'labels' / 'val',
    ]

    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        print(f"📁 Создана папка: {d}")

    # Копируем изображения
    print("\n📋 Копирование изображений...")
    for split in ['train', 'val']:
        src_img_dir = source_path / 'images' / split
        dst_img_dir = fixed_path / 'images' / split

        if src_img_dir.exists():
            for img_file in src_img_dir.glob("*.*"):
                shutil.copy2(img_file, dst_img_dir / img_file.name)
            print(f"   {split}: скопировано {len(list(src_img_dir.glob('*.*')))} изображений")

    # Конвертируем разметку
    print("\n🔄 Конвертация разметки...")
    converted = {'train': 0, 'val': 0}
    errors = {'train': 0, 'val': 0}

    for split in ['train', 'val']:
        src_label_dir = source_path / 'labels' / split
        dst_label_dir = fixed_path / 'labels' / split

        if src_label_dir.exists():
            label_files = list(src_label_dir.glob("*.txt"))
            for label_file in label_files:
                dst_file = dst_label_dir / label_file.name
                if convert_to_format(label_file, dst_file, target_points):
                    converted[split] += 1
                else:
                    errors[split] += 1

            print(f"\n   {split}:")
            print(f"      Всего файлов: {len(label_files)}")
            print(f"      Сконвертировано: {converted[split]}")
            print(f"      Ошибок: {errors[split]}")

    # Создаём YAML конфиг
    yaml_content = {
        'path': str(fixed_path),
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [target_points, 3],
        'names': {
            0: 'ISS'
        }
    }

    yaml_path = fixed_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n✅ Создан конфиг: {yaml_path}")
    print("\n📄 Содержимое:")
    with open(yaml_path, 'r') as f:
        print(f.read())

    # Создаём финальный скрипт обучения
    train_script = fixed_path / 'train_final.py'
    script_content = f'''#!/usr/bin/env python3
"""
Обучение на исправленном датасете
Количество точек: {target_points}
"""
from ultralytics import YOLO

# Загружаем модель
model = YOLO('yolov8m-pose.pt')

# Параметры обучения
results = model.train(
    data='{fixed_path}/dataset.yaml',
    epochs=30,
    imgsz=640,
    batch=8,  # уменьшенный batch для надежности
    device='mps',  # для Mac
    patience=20,
    save=True,
    project='/Users/darya/Desktop/iss_trained_final',
    name='fixed_experiment',
    exist_ok=True,
    workers=2,  # меньше потоков для стабильности
    deterministic=True  # для воспроизводимости
)

print("\\n✅ Обучение завершено!")
print(f"Модель сохранена в {{results.save_dir}}")

# Проверка на тестовых изображениях
print("\\n🔍 Тестирование модели...")
test_results = model.val()
print(f"   mAP: {{test_results.box.map}}")
'''

    with open(train_script, 'w') as f:
        f.write(script_content)

    print(f"\n📝 Создан скрипт обучения: {train_script}")

    return fixed_path, target_points


def verify_fixed_dataset(fixed_path, target_points):
    """
    Проверяет исправленный датасет
    """
    print("\n" + "=" * 70)
    print("🔍 ПРОВЕРКА ИСПРАВЛЕННОГО ДАТАСЕТА")
    print("=" * 70)

    for split in ['train', 'val']:
        label_dir = fixed_path / 'labels' / split
        if not label_dir.exists():
            continue

        label_files = list(label_dir.glob("*.txt"))
        print(f"\n📊 {split}: {len(label_files)} файлов")

        # Проверяем первые 3 файла
        for i, label_file in enumerate(label_files[:3]):
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                num_cols = len(parts)
                num_points = (num_cols - 1) // 3 if num_cols > 1 else 0
                print(f"   {label_file.name}: {num_cols} колонок, {num_points} точек")

                if num_points != target_points:
                    print(f"      ⚠️ Ожидалось {target_points} точек!")

    return True


def main():
    # Исправляем датасет
    result = fix_dataset()
    if not result:
        return

    fixed_path, target_points = result

    # Проверяем
    verify_fixed_dataset(fixed_path, target_points)

    print("\n" + "=" * 70)
    print("🎯 ДАТАСЕТ ГОТОВ К ОБУЧЕНИЮ!")
    print("=" * 70)
    print(f"\n📁 Исправленный датасет: {fixed_path}")
    print(f"\n🚀 Запустите обучение:")
    print(f"   python {fixed_path}/train_final.py")
    print("\n📌 Важно: Первая эпоха может быть медленной из-за кэширования")


if __name__ == "__main__":
    main()