#!/usr/bin/env python3
"""
ISS Docking Dataset Processor with UI
Точка входа в приложение
"""
import sys
import os
from pathlib import Path

# Добавляем текущую папку в путь поиска
sys.path.append(str(Path(__file__).parent))

try:
    from ui_app import main
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все файлы проекта на месте:")
    print("  - ui_app.py")
    print("  - processor.py")
    print("  - visualizer.py")
    print("  - config.py")
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("ISS Docking Dataset Processor with UI")
    print("=" * 60)
    print("Запуск графического интерфейса...")
    main()
