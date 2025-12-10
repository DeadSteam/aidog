"""
Скрипт для разделения Stanford Dogs Dataset на train/val/test (70/15/15)
"""
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images_dir, source_annotations_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Разделяет датасет на обучающую, валидационную и тестовую выборки
    
    Args:
        source_images_dir: Путь к исходным изображениям
        source_annotations_dir: Путь к исходным аннотациям
        output_base_dir: Базовый путь для выходных директорий
        train_ratio: Доля обучающей выборки (по умолчанию 0.7)
        val_ratio: Доля валидационной выборки (по умолчанию 0.15)
        test_ratio: Доля тестовой выборки (по умолчанию 0.15)
    """
    # Проверка соотношения
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Сумма соотношений должна быть равна 1.0"
    
    # Создание выходных директорий
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'annotations', split), exist_ok=True)
    
    # Получение списка классов
    source_images_path = Path(source_images_dir)
    classes = [d.name for d in source_images_path.iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"Найдено классов: {len(classes)}")
    
    total_images = 0
    
    # Обработка каждого класса
    for class_name in classes:
        class_images_dir = source_images_path / class_name
        class_annotations_dir = Path(source_annotations_dir) / class_name
        
        # Получение списка изображений
        images = list(class_images_dir.glob('*.jpg'))
        if not images:
            print(f"Предупреждение: нет изображений в классе {class_name}")
            continue
        
        # Перемешивание
        random.shuffle(images)
        
        # Вычисление индексов разделения
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # n_test = n_total - n_train - n_val (остаток)
        
        # Разделение
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        total_images += n_total
        
        # Копирование файлов для каждого сплита
        for split, split_images in zip(['train', 'val', 'test'], 
                                       [train_images, val_images, test_images]):
            split_images_dir = Path(output_base_dir) / 'images' / split / class_name
            split_annotations_dir = Path(output_base_dir) / 'annotations' / split / class_name
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_annotations_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                # Копирование изображения
                shutil.copy2(img_path, split_images_dir / img_path.name)
                
                # Копирование аннотации (если существует)
                ann_path = class_annotations_dir / img_path.stem
                if ann_path.exists():
                    # Аннотации могут быть файлами без расширения
                    shutil.copy2(ann_path, split_annotations_dir / ann_path.name)
        
        print(f"Класс {class_name}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    print(f"\nВсего обработано изображений: {total_images}")
    print(f"Разделение завершено в директории: {output_base_dir}")

if __name__ == "__main__":
    # Параметры
    source_images = "archive/images/Images"
    source_annotations = "archive/annotations/Annotation"
    output_dir = "archive"
    
    # Установка seed для воспроизводимости
    random.seed(42)
    
    # Разделение датасета
    split_dataset(
        source_images_dir=source_images,
        source_annotations_dir=source_annotations,
        output_base_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\nГотово!")



