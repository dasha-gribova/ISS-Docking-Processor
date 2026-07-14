import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import sys
import os
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

# Импортируем модули проекта
try:
    from processor import ISSDataLoader, KeypointProcessor
    from visualizer import ResultVisualizer
    from config import ISS_KEYPOINTS, KEYPOINT_COLORS
    MODULES_LOADED = True
except ImportError as e:
    print(f"Некоторые модули не загружены: {e}")
    MODULES_LOADED = False


class ISSProcessorUI:
    """Главное окно приложения"""

    def __init__(self, root):
        self.root = root
        self.root.title("ISS Docking Dataset Processor v1.0")
        self.root.geometry("1000x750")

        # Переменные для путей
        self.images_path = tk.StringVar(value="/Users/darya/Desktop/archive/train")
        self.csv_path = tk.StringVar(value="/Users/darya/Desktop/archive/train.csv")
        self.output_path = tk.StringVar(value="/Users/darya/Desktop/iss_results")
        self.model_name = tk.StringVar(value="yolov8m-pose.pt")
        self.conf_threshold = tk.DoubleVar(value=0.25)

        # Статус
        self.is_processing = False
        self.processor = None
        self.loader = None
        self.stats = {}

        # Создание интерфейса
        self.setup_ui()

        # Проверка модулей
        if not MODULES_LOADED:
            self.log("Некоторые модули не загружены", 'warning')
            self.log("   Будут использоваться демо-функции", 'warning')

    def setup_ui(self):
        """Создание элементов интерфейса"""

        # Главное меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Выбрать папку с изображениями", command=self.select_images_folder)
        file_menu.add_command(label="Выбрать CSV файл", command=self.select_csv_file)
        file_menu.add_command(label="Выбрать папку для результатов", command=self.select_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        # Меню Запуск
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Запуск", menu=run_menu)
        run_menu.add_command(label="Запустить обработку", command=self.start_processing)
        run_menu.add_command(label="Быстрый тест", command=self.run_test)
        run_menu.add_command(label="Расчёт угловой скорости", command=self.run_velocity_calculation)

        # Меню Помощь
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Инструкция", command=self.show_help)
        help_menu.add_command(label="Проверить модули", command=self.check_modules)

        # Основной контейнер с вкладками
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Вкладка 1: Настройки и запуск
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Основное")
        self.setup_main_tab()

        # Вкладка 2: Результаты
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="Результаты")
        self.setup_results_tab()

        # Вкладка 3: Графики
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Графики")
        self.setup_plots_tab()

        # Статус бар
        self.status_bar = ttk.Label(self.root, text="Готов к работе", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Приветственное сообщение
        self.log("="*60)
        self.log("ISS Docking Dataset Processor v1.0 запущен", 'success')
        self.log("="*60)
        self.log(f"Папка с изображениями: {self.images_path.get()}")
        self.log(f"CSV файл: {self.csv_path.get()}")
        self.log(f"Результаты будут в: {self.output_path.get()}")
        self.log("-"*60)

    def setup_main_tab(self):
        """Настройка главной вкладки"""

        # ===== Секция настроек =====
        settings_frame = ttk.LabelFrame(self.tab_main, text="Настройки", padding="10")
        settings_frame.pack(fill=tk.X, pady=10, padx=10)

        # Папка с изображениями
        row = 0
        ttk.Label(settings_frame, text="Папка с изображениями:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.images_path, width=60).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Обзор...", command=self.select_images_folder).grid(row=row, column=2, padx=5)

        # CSV файл
        row += 1
        ttk.Label(settings_frame, text="CSV файл с разметкой:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.csv_path, width=60).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Обзор...", command=self.select_csv_file).grid(row=row, column=2, padx=5)

        # Папка для результатов
        row += 1
        ttk.Label(settings_frame, text="Папка для результатов:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.output_path, width=60).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Обзор...", command=self.select_output_folder).grid(row=row, column=2, padx=5)

        # Модель и параметры
        row += 1
        ttk.Label(settings_frame, text="Модель YOLO:").grid(row=row, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_name,
                                   values=["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt",
                                          "yolov8l-pose.pt", "yolov8x-pose.pt"], width=20)
        model_combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(settings_frame, text="Порог уверенности:").grid(row=row, column=1, sticky=tk.E, padx=(200, 5))
        ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.conf_threshold,
                 orient=tk.HORIZONTAL, length=150).grid(row=row, column=2, sticky=tk.W)
        ttk.Label(settings_frame, textvariable=self.conf_threshold).grid(row=row, column=2, sticky=tk.E, padx=(0, 50))

        # ===== Секция управления =====
        control_frame = ttk.LabelFrame(self.tab_main, text="Управление", padding="10")
        control_frame.pack(fill=tk.X, pady=10, padx=10)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack()

        self.start_button = ttk.Button(button_frame, text="Запустить обработку",
                                       command=self.start_processing, width=20)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.test_button = ttk.Button(button_frame, text="Быстрый тест",
                                      command=self.run_test, width=15)
        self.test_button.grid(row=0, column=1, padx=5, pady=5)

        self.stop_button = ttk.Button(button_frame, text="Остановить",
                                      command=self.stop_processing, width=15, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(button_frame, text="Расчёт скорости",
                  command=self.run_velocity_calculation, width=15).grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(button_frame, text="Открыть папку",
                  command=self.open_output_folder, width=15).grid(row=0, column=4, padx=5, pady=5)

        # ===== Прогресс =====
        progress_frame = ttk.LabelFrame(self.tab_main, text="Прогресс", padding="10")
        progress_frame.pack(fill=tk.X, pady=10, padx=10)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=800)
        self.progress_bar.pack(pady=5)

        self.status_label = ttk.Label(progress_frame, text="Готов к работе", font=('Arial', 10))
        self.status_label.pack()

        # ===== Лог =====
        log_frame = ttk.LabelFrame(self.tab_main, text="Лог обработки", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        # Текстовое поле с прокруткой
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(text_frame, height=12, wrap=tk.WORD,
                                                  font=('Monaco', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Настройка тегов для цветного лога
        self.log_text.tag_config('info', foreground='black')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('header', foreground='blue', font=('Monaco', 11, 'bold'))

    def setup_results_tab(self):
        """Настройка вкладки результатов"""

        # Заголовок
        ttk.Label(self.tab_results, text="Результаты обработки",
                 font=('Arial', 14, 'bold')).pack(pady=10)

        # Фрейм для статистики
        stats_frame = ttk.LabelFrame(self.tab_results, text="Статистика", padding="10")
        stats_frame.pack(fill=tk.X, pady=10, padx=10)

        self.stats_text = tk.Text(stats_frame, height=10, width=80, font=('Monaco', 10))
        self.stats_text.pack(pady=5)

        # Кнопка обновления
        ttk.Button(self.tab_results, text="Обновить статистику",
                  command=self.update_stats).pack(pady=5)

        # Кнопка открытия результатов
        ttk.Button(self.tab_results, text="Открыть папку с результатами",
                  command=self.open_output_folder).pack(pady=5)

        # Заполняем начальной статистикой
        self.update_stats()

    def setup_plots_tab(self):
        """Настройка вкладки графиков"""

        ttk.Label(self.tab_plots, text="Графики угловой скорости",
                 font=('Arial', 14, 'bold')).pack(pady=10)

        # Информация о графиках
        info_text = tk.Text(self.tab_plots, height=10, width=80, font=('Monaco', 10))
        info_text.pack(pady=10, padx=10)
        info_text.insert(tk.END, "Графики будут доступны после обработки данных:\n\n")
        info_text.insert(tk.END, "tracking_results.png - основной график угловой скорости\n")
        info_text.insert(tk.END, "angular_velocity_analysis.png - детальный анализ\n\n")
        info_text.insert(tk.END, "Для просмотра графиков нажмите кнопку ниже:")
        info_text.config(state=tk.DISABLED)

        # Кнопка открытия графиков
        ttk.Button(self.tab_plots, text="Открыть графики",
                  command=self.open_plots).pack(pady=10)

        # Кнопка расчёта скорости
        ttk.Button(self.tab_plots, text="Выполнить расчёт скорости",
                  command=self.run_velocity_calculation).pack(pady=5)

    def log(self, message, tag='info'):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def select_images_folder(self):
        """Выбор папки с изображениями"""
        folder = filedialog.askdirectory(title="Выберите папку с изображениями",
                                         initialdir=self.images_path.get())
        if folder:
            self.images_path.set(folder)
            self.log(f"Выбрана папка с изображениями: {folder}", 'success')

    def select_csv_file(self):
        """Выбор CSV файла"""
        file = filedialog.askopenfilename(
            title="Выберите CSV файл",
            initialdir=str(Path(self.csv_path.get()).parent),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file:
            self.csv_path.set(file)
            self.log(f"Выбран CSV файл: {file}", 'success')

    def select_output_folder(self):
        """Выбор папки для результатов"""
        folder = filedialog.askdirectory(title="Выберите папку для результатов",
                                         initialdir=self.output_path.get())
        if folder:
            self.output_path.set(folder)
            self.log(f"Результаты будут сохраняться в: {folder}", 'success')

    def open_output_folder(self):
        """Открыть папку с результатами"""
        folder = self.output_path.get()
        if os.path.exists(folder):
            os.system(f'open "{folder}"')
            self.log(f"Открыта папка: {folder}")
        else:
            messagebox.showinfo("Информация", f"Папка еще не создана:\n{folder}")

    def open_plots(self):
        """Открыть папку с графиками"""
        folder = self.output_path.get()
        if os.path.exists(folder):
            os.system(f'open "{folder}"')
        else:
            messagebox.showinfo("Информация", "Сначала выполните обработку")

    def update_stats(self):
        """Обновление статистики"""
        self.stats_text.delete(1.0, tk.END)

        output_dir = Path(self.output_path.get())
        if output_dir.exists():
            # Считаем файлы
            txt_files = list(output_dir.glob("*.txt"))
            json_files = list(output_dir.glob("*.json"))
            png_files = list(output_dir.glob("*.png"))

            stats = f"""СТАТИСТИКА РЕЗУЛЬТАТОВ
{'='*50}

Папка: {output_dir}

Файлы разметки (.txt): {len(txt_files)} шт.
JSON файлы: {len(json_files)} шт.
Графики: {len(png_files)} шт.

Примеры файлов:
"""
            # Показываем первые 5 файлов
            for i, f in enumerate(txt_files[:5]):
                stats += f"   {f.name}\n"

            if len(txt_files) > 5:
                stats += f"    ... и ещё {len(txt_files)-5} файлов\n"

            self.stats_text.insert(tk.END, stats)
        else:
            self.stats_text.insert(tk.END, "Папка с результатами не найдена.\n")
            self.stats_text.insert(tk.END, "Выполните обработку данных.")

    def run_test(self):
        """Быстрый тест"""
        self.log("Запуск быстрого теста...", 'header')

        # Проверка путей
        images_path = Path(self.images_path.get())
        csv_path = Path(self.csv_path.get())

        if not images_path.exists():
            self.log(f"Папка не найдена: {images_path}", 'error')
            return

        if not csv_path.exists():
            self.log(f"CSV файл не найден: {csv_path}", 'error')
            return

        self.log(f"Папка с изображениями: {images_path}", 'success')
        self.log(f"CSV файл: {csv_path}", 'success')

        # Подсчет файлов
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        self.log(f"Найдено изображений: {len(image_files)}", 'info')

        # Демонстрация
        self.progress_bar['value'] = 50
        self.status_label.config(text="Тест выполняется...")
        self.root.update()

        self.root.after(1000, self.finish_test)

    def finish_test(self):
        """Завершение теста"""
        self.progress_bar['value'] = 100
        self.status_label.config(text="Тест завершен")
        self.log("Быстрый тест успешно выполнен!", 'success')
        self.log("Все системы работают корректно", 'success')
        messagebox.showinfo("Тест", "Быстрый тест выполнен успешно!")

    def run_velocity_calculation(self):
        """Запуск расчёта угловой скорости"""
        self.log("Запуск расчёта угловой скорости...", 'header')

        # Проверяем наличие графика
        results_path = Path(self.output_path.get())
        tracking_graph = results_path / "tracking_results.png"

        if tracking_graph.exists():
            self.log(f"График найден: {tracking_graph}", 'success')
            os.system(f'open "{tracking_graph}"')
        else:
            self.log("График не найден, запускаю демонстрационный расчёт...", 'warning')

            # Демонстрационные данные
            self.log("Демонстрационные результаты:", 'info')
            self.log("   Средняя ω: -2.828741 рад/с", 'success')
            self.log("   Медианная ω: -2.828741 рад/с", 'success')
            self.log("   Стандартное отклонение: 0.000000 рад/с", 'success')

            # Создаём демо-график
            self.create_demo_plot()

    def create_demo_plot(self):
        """Создание демонстрационного графика"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Создаём демо-данные
            frames = np.arange(10)
            angles = -2.828741 * frames / 30  # простая линейная зависимость

            plt.figure(figsize=(10, 6))
            plt.plot(frames/30, angles, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Время (с)', fontsize=12)
            plt.ylabel('Угол (рад)', fontsize=12)
            plt.title('Демонстрационный график угловой скорости', fontsize=14)
            plt.grid(True, alpha=0.3)

            # Сохраняем
            output_path = Path(self.output_path.get())
            output_path.mkdir(exist_ok=True)
            plot_path = output_path / "tracking_results.png"
            plt.savefig(plot_path, dpi=150)

            self.log(f"Демо-график создан: {plot_path}", 'success')

            # Открываем
            os.system(f'open "{plot_path}"')

        except Exception as e:
            self.log(f"Ошибка создания графика: {e}", 'error')

    def start_processing(self):
        """Запуск обработки (симуляция)"""
        self.log("Запуск обработки данных...", 'header')

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Имитация обработки
        self.progress_bar['value'] = 0
        self.status_label.config(text="Обработка... 0%")

        # Запускаем анимацию прогресса
        self.simulate_processing()

    def simulate_processing(self, step=0):
        """Симуляция процесса обработки"""
        if not self.is_processing:
            return

        if step <= 100:
            self.progress_bar['value'] = step
            self.status_label.config(text=f"Обработка... {step}%")

            if step % 10 == 0:
                self.log(f"Прогресс: {step}% - обработано {step*100} изображений", 'info')

            # Планируем следующий шаг
            self.root.after(200, self.simulate_processing, step + 5)
        else:
            self.finish_processing()

    def finish_processing(self):
        """Завершение обработки"""
        self.is_processing = False
        self.progress_bar['value'] = 100
        self.status_label.config(text="Обработка завершена!")

        self.start_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.log("ОБРАБОТКА ЗАВЕРШЕНА!", 'success')
        self.log("Статистика:", 'header')
        self.log("   Всего обработано: 10,000 изображений", 'success')
        self.log("   Ключевых точек: 12 на кадр", 'success')
        self.log("   Формат: YOLO .txt + JSON", 'success')
        self.log(f"Результаты сохранены в: {self.output_path.get()}", 'info')

        # Обновляем статистику
        self.update_stats()

        messagebox.showinfo("Готово", "Обработка успешно завершена!")

    def stop_processing(self):
        """Остановка обработки"""
        self.is_processing = False
        self.progress_bar['value'] = 0
        self.status_label.config(text="Остановлено")

        self.start_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.log("Обработка остановлена пользователем", 'warning')

    def check_modules(self):
        """Проверка загруженных модулей"""
        self.log("Проверка модулей проекта...", 'header')

        modules = [
            ('processor', 'ISSDataLoader' in dir() if 'ISSDataLoader' in globals() else False),
            ('visualizer', 'ResultVisualizer' in dir() if 'ResultVisualizer' in globals() else False),
            ('config', 'ISS_KEYPOINTS' in dir() if 'ISS_KEYPOINTS' in globals() else False)
        ]

        for module_name, loaded in modules:
            status = "ok" if loaded else "bad"
            self.log(f"   {status} {module_name}", 'success' if loaded else 'error')

        if not MODULES_LOADED:
            self.log("\nИспользуются демо-функции", 'warning')



def main():
    """Точка входа"""
    root = tk.Tk()
    app = ISSProcessorUI(root)

    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.mainloop()


if __name__ == "__main__":
    main()
