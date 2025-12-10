"""
Скрипт для обучения Vision Transformer на Stanford Dogs Dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path

from model import VisionTransformer, DogDataset, get_transforms

try:
    # Реальный оптимизатор SPAM: Spike-Aware Adam with Momentum Reset
    from galore_torch import SPAM
except ImportError:
    SPAM = None


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=False, scaler=None):
    """Обучение модели на одной эпохе с поддержкой mixed precision"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass с mixed precision
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass с gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def validate(model, dataloader, criterion, device, use_amp=False):
    """Валидация модели с поддержкой mixed precision"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def test(model, dataloader, device, use_amp=False):
    """Тестирование модели с поддержкой mixed precision"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Обучение Vision Transformer')
    parser.add_argument('--data_dir', type=str, default='archive',
                       help='Директория с данными')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Размер батча (по умолчанию: 64 для GPU, 32 для CPU)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'spam'],
                       help='Оптимизатор (adam или spam - Spike-Aware Adam with Momentum Reset)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Использовать предобученную модель')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Размер изображения')
    parser.add_argument('--num_classes', type=int, default=120,
                       help='Количество классов')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Директория для сохранения моделей')
    parser.add_argument('--use_amp', action='store_true',
                       help='Использовать mixed precision training (AMP) для ускорения на GPU')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Отключить mixed precision training')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Количество воркеров для DataLoader (по умолчанию: 8 для GPU, 0 для CPU)')
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    
    if use_cuda:
        print(f'Используемое устройство: {device}')
        print(f'Название GPU: {torch.cuda.get_device_name(0)}')
        print(f'Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        print(f'Используемое устройство: {device} (CPU)')
    
    # Автоматический выбор batch size для GPU
    if args.batch_size is None:
        args.batch_size = 64 if use_cuda else 32
        print(f'Batch size установлен автоматически: {args.batch_size}')
    
    # Автоматический выбор num_workers
    # Для Windows используем меньше workers, чтобы избежать проблем с памятью и multiprocessing
    if args.num_workers is None:
        import sys
        is_windows = sys.platform == 'win32'
        if use_cuda:
            # Для Windows используем меньше workers из-за проблем с multiprocessing и памятью
            # Для Linux можно использовать больше
            args.num_workers = 2 if is_windows else 4
        else:
            args.num_workers = 0
        print(f'Num workers установлен автоматически: {args.num_workers} (Windows: {is_windows})')
    
    # Mixed precision: по умолчанию включен для GPU
    # Если --no_amp не указан и мы на GPU, включаем AMP по умолчанию
    # argparse устанавливает use_amp=False если не указан флаг, поэтому для GPU устанавливаем True
    if use_cuda:
        # По умолчанию включаем AMP для GPU (если не указан --no_amp)
        # Если указан --no_amp, args.use_amp будет False
        # Если указан --use_amp, args.use_amp будет True
        # Если ничего не указано, args.use_amp будет False, но мы хотим True для GPU
        if not args.use_amp:  # Если False (не указан --use_amp и не указан --no_amp)
            # Проверяем, был ли явно указан --no_amp через sys.argv
            import sys
            if '--no_amp' not in sys.argv:
                args.use_amp = True  # Включаем по умолчанию для GPU
    
    # Mixed precision только для CUDA
    use_amp = args.use_amp and use_cuda
    if use_amp:
        # Проверяем поддержку mixed precision
        if torch.cuda.is_bf16_supported():
            print('Используется Mixed Precision Training (AMP) с bfloat16 для ускорения')
        else:
            print('Используется Mixed Precision Training (AMP) с float16 для ускорения')
    elif args.use_amp and not use_cuda:
        print('Mixed Precision доступен только для GPU, используется FP32')
    elif use_cuda:
        print('Mixed Precision отключен, используется FP32')
    
    # Создание директории для сохранения
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Загрузка данных
    print('\nЗагрузка данных...')
    train_dataset = DogDataset(
        images_dir=os.path.join(args.data_dir, 'images', 'train'),
        transform=get_transforms('train', args.image_size),
        split='train'
    )
    val_dataset = DogDataset(
        images_dir=os.path.join(args.data_dir, 'images', 'val'),
        transform=get_transforms('val', args.image_size),
        split='val'
    )
    test_dataset = DogDataset(
        images_dir=os.path.join(args.data_dir, 'images', 'test'),
        transform=get_transforms('test', args.image_size),
        split='test'
    )
    
    # Настройки загрузчиков данных для GPU
    # Для GPU: используем воркеры и pin_memory для ускорения
    # Для CPU: num_workers=0 чтобы избежать проблем с multiprocessing на Windows
    pin_memory = use_cuda
    # persistent_workers может вызывать проблемы с памятью на Windows
    import sys
    is_windows = sys.platform == 'win32'
    persistent_workers = use_cuda and args.num_workers > 0 and not is_windows
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    # Создание модели
    print(f'\nСоздание модели (pretrained={args.pretrained})...')
    model = VisionTransformer(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        image_size=args.image_size
    ).to(device)
    
    # Loss и оптимизатор
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:  # spam: настоящий SPAM-оптимизатор без разреженных масок
        if SPAM is None:
            raise ImportError(
                "Оптимизатор SPAM не найден. Убедитесь, что в проекте есть пакет galore_torch."
            )
        # Все параметры без 'density', чтобы не создавать 2D-маски и избежать ошибок по shape
        param_groups = [{'params': model.parameters()}]
        optimizer = SPAM(
            param_groups,
            lr=args.lr,
            warmup_steps=150,
            threshold=5000,
            DeltaT=500,
        )
    
    # Scheduler (только для Adam, SPAM имеет свой внутренний warmup)
    if args.optimizer == 'adam':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    else:
        scheduler = None  # SPAM имеет свой внутренний warmup механизм
    
    # Mixed precision scaler для GPU
    scaler = None
    if use_amp:
        # Используем новый API для GradScaler (PyTorch 2.0+)
        scaler = torch.amp.GradScaler('cuda')
    
    # Обучение
    print(f'\nНачало обучения (оптимизатор: {args.optimizer.upper()})...')
    if use_amp:
        print('Mixed Precision Training (AMP) включен')
    best_val_acc = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(1, args.epochs + 1):
        # Обучение
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp, scaler
        )
        
        # Валидация
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device, use_amp
        )
        
        # Scheduler step (только для Adam)
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f'  Learning rate уменьшен: {old_lr:.6f} -> {new_lr:.6f}')
        
        # Сохранение истории
        train_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1
        })
        
        val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1
        })
        
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
              f'Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, '
              f'Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}')
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f'best_model_{args.optimizer}_pretrained{args.pretrained}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
            }, os.path.join(args.save_dir, model_name))
            print(f'  Сохранена лучшая модель (Val Acc: {val_acc:.4f})')
    
    # Тестирование лучшей модели
    print('\nЗагрузка лучшей модели для тестирования...')
    checkpoint = torch.load(os.path.join(args.save_dir, model_name), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device, use_amp)
    
    print(f'\n=== Результаты на тестовой выборке ===')
    print(f'Accuracy:  {test_acc:.4f}')
    print(f'Precision: {test_prec:.4f}')
    print(f'Recall:    {test_rec:.4f}')
    print(f'F1-score:  {test_f1:.4f}')
    
    # Сохранение результатов
    results = {
        'config': vars(args),
        'train_history': train_history,
        'val_history': val_history,
        'test_results': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        }
    }
    
    import json
    results_name = f'results_{args.optimizer}_pretrained{args.pretrained}.json'
    with open(os.path.join(args.save_dir, results_name), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nРезультаты сохранены в {os.path.join(args.save_dir, results_name)}')


if __name__ == '__main__':
    main()

