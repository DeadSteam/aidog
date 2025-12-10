# Stanford Dogs Dataset - Vision Transformer Classification

Проект для классификации пород собак из Stanford Dogs Dataset с использованием Vision Transformer (ViT).

## Структура проекта

```
aidog/
├── archive/                    # Датасет
│   ├── images/
│   │   ├── Images/            # Исходные изображения
│   │   ├── train/            # Обучающая выборка (70%)
│   │   ├── val/              # Валидационная выборка (15%)
│   │   └── test/             # Тестовая выборка (15%)
│   └── annotations/
│       └── Annotation/        # Исходные аннотации
├── split_dataset.py           # Скрипт разделения датасета
├── model.py                   # Vision Transformer модель
├── train.py                   # Скрипт обучения
├── run_experiments.py         # Скрипт запуска всех экспериментов
├── compare_results.py         # Скрипт сравнения результатов
├── checkpoints/               # Сохраненные модели и результаты
└── requirements.txt           # Зависимости

```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас есть датасет в папке `archive/images/Images` и `archive/annotations/Annotation`

## Использование

### 1. Разделение датасета

Разделите датасет на train/val/test (70/15/15):
```bash
python split_dataset.py
```

### 2. Обучение модели

#### Вариант 1: Дообучение предобученной модели с Adam
```bash
python train.py --pretrained --optimizer adam --epochs 10 --lr 1e-4 --batch_size 32
```

#### Вариант 2: Дообучение предобученной модели с SPAM (Spike-Aware Adam with Momentum Reset)
```bash
python train.py --pretrained --optimizer spam --epochs 10 --lr 1e-3 --batch_size 32
```

#### Вариант 3: Обучение с нуля с Adam
```bash
python train.py --optimizer adam --epochs 15 --lr 1e-4 --batch_size 32
```

### 3. Запуск всех экспериментов

Для автоматического запуска всех трех экспериментов:
```bash
python run_experiments.py
```

### 4. Сравнение результатов

После завершения экспериментов можно сравнить результаты:
```bash
python compare_results.py
```

Этот скрипт выведет сравнительную таблицу всех экспериментов с метриками и найдет лучшие модели.

## Параметры обучения

- `--data_dir`: Директория с данными (по умолчанию: `archive`)
- `--batch_size`: Размер батча (по умолчанию: 32)
- `--epochs`: Количество эпох (по умолчанию: 10)
- `--lr`: Learning rate (по умолчанию: 1e-4)
- `--optimizer`: Оптимизатор - `adam` или `spam` (по умолчанию: `adam`)
- `--pretrained`: Использовать предобученную модель
- `--image_size`: Размер изображения (по умолчанию: 224)
- `--num_classes`: Количество классов (по умолчанию: 120)
- `--save_dir`: Директория для сохранения (по умолчанию: `checkpoints`)

## Эксперименты

Проект включает три основных эксперимента:

1. **Дообучение предобученной модели с Adam**
   - Используется предобученная ViT-base-patch16-224
   - Оптимизатор: Adam
   - Learning rate: 1e-4

2. **Дообучение предобученной модели с SPAM**
   - Используется предобученная ViT-base-patch16-224
   - Оптимизатор: SPAM (Spike-Aware Adam with Momentum Reset)
   - Learning rate: 1e-3

3. **Обучение с нуля с Adam**
   - Модель инициализируется случайным образом
   - Оптимизатор: Adam
   - Learning rate: 1e-4

## Метрики

Модели оцениваются по следующим метрикам:
- **Accuracy** - точность классификации
- **Precision** - точность (macro-averaged)
- **Recall** - полнота (macro-averaged)
- **F1-score** - F1-мера (macro-averaged)

Результаты сохраняются в JSON файлах в директории `checkpoints/`.

## Архитектура модели

Используется Vision Transformer (ViT) с конфигурацией:
- Image size: 224x224
- Patch size: 16x16
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Number of classes: 120 (породы собак)

## Результаты

После обучения результаты сохраняются в:
- `checkpoints/best_model_*.pth` - лучшие модели
- `checkpoints/results_*.json` - метрики и история обучения

## Примечания

- Для предобученной модели используется `google/vit-base-patch16-224` из Hugging Face Transformers
- Модель использует нормализацию ImageNet для изображений
- При обучении применяются аугментации: горизонтальное отражение, поворот, изменение яркости/контраста
- Learning rate scheduler: ReduceLROnPlateau с фактором 0.5 и patience 3

