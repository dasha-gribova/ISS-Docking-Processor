"""
Конфигурация проекта ISS Docking UI
"""
from pathlib import Path

# Ключевые точки МКС
ISS_KEYPOINTS = [
    "docking_port",              # 0: Центр стыковочного порта
    "solar_panel_left_tip",      # 1: Левый край левой панели
    "solar_panel_left_base",     # 2: Основание левой панели
    "solar_panel_right_tip",     # 3: Правый край правой панели
    "solar_panel_right_base",    # 4: Основание правой панели
    "zvezda_module_front",       # 5: Передняя часть "Звезды"
    "zarya_module_rear",         # 6: Кормовая часть "Зари"
    "upper_antenna",             # 7: Верхняя антенна
    "lower_antenna",             # 8: Нижняя антенна
    "service_module_center",     # 9: Центр служебного модуля
    "docking_compartment_1",     # 10: Стыковочный отсек 1
    "docking_compartment_2"      # 11: Стыковочный отсек 2
]

# Цвета для визуализации (RGB)
KEYPOINT_COLORS = [
    (0, 255, 0),      # зеленый
    (255, 0, 0),      # синий
    (255, 165, 0),    # оранжевый
    (0, 0, 255),      # красный
    (255, 255, 0),    # голубой
    (255, 0, 255),    # розовый
    (0, 255, 255),    # желтый
    (128, 0, 128),    # фиолетовый
    (0, 128, 128),    # бирюзовый
    (128, 128, 0),    # оливковый
    (255, 255, 255),  # белый
    (0, 0, 0)         # черный
]

# Модели YOLO для выбора
YOLO_MODELS = [
    "yolov8n-pose.pt",   # nano - самая быстрая
    "yolov8s-pose.pt",   # small
    "yolov8m-pose.pt",   # medium (рекомендуется)
    "yolov8l-pose.pt",   # large
    "yolov8x-pose.pt"    # xlarge - самая точная
]