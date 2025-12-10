"""
Скрипт для генерации графиков обучения из результатов экспериментов
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Поддержка кириллицы
import numpy as np


def load_results(checkpoints_dir='checkpoints'):
    """Загрузить все результаты из директории checkpoints"""
    results = []
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"Директория {checkpoints_dir} не найдена!")
        return None
    
    # Поиск всех JSON файлов с результатами
    result_files = list(checkpoints_path.glob('results_*.json'))
    
    if not result_files:
        print(f"Файлы результатов не найдены в {checkpoints_dir}")
        return None
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'file': result_file.name,
                    'config': data.get('config', {}),
                    'train_history': data.get('train_history', []),
                    'val_history': data.get('val_history', []),
                    'test_results': data.get('test_results', {})
                })
        except Exception as e:
            print(f"Ошибка при загрузке {result_file}: {e}")
    
    return results


def plot_training_curves(results, output_dir='plots'):
    """Построить графики обучения для всех экспериментов"""
    os.makedirs(output_dir, exist_ok=True)
    
    # График 1: Accuracy по эпохам
    plt.figure(figsize=(12, 6))
    for result in results:
        train_history = result['train_history']
        val_history = result['val_history']
        config = result['config']
        
        epochs = [h['epoch'] for h in train_history]
        train_acc = [h['accuracy'] for h in train_history]
        val_acc = [h['accuracy'] for h in val_history]
        
        label = f"{config.get('optimizer', 'unknown').upper()}"
        if config.get('pretrained', False):
            label += " (pretrained)"
        else:
            label += " (scratch)"
        
        plt.plot(epochs, train_acc, 'o-', label=f'Train {label}', linewidth=2, markersize=4)
        plt.plot(epochs, val_acc, 's--', label=f'Val {label}', linewidth=2, markersize=4)
    
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Сравнение Accuracy по эпохам', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Сохранен график: {os.path.join(output_dir, 'accuracy_curves.png')}")
    plt.close()
    
    # График 2: Loss по эпохам
    plt.figure(figsize=(12, 6))
    for result in results:
        train_history = result['train_history']
        val_history = result['val_history']
        config = result['config']
        
        epochs = [h['epoch'] for h in train_history]
        train_loss = [h['loss'] for h in train_history]
        val_loss = [h['loss'] for h in val_history]
        
        label = f"{config.get('optimizer', 'unknown').upper()}"
        if config.get('pretrained', False):
            label += " (pretrained)"
        else:
            label += " (scratch)"
        
        plt.plot(epochs, train_loss, 'o-', label=f'Train {label}', linewidth=2, markersize=4)
        plt.plot(epochs, val_loss, 's--', label=f'Val {label}', linewidth=2, markersize=4)
    
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Сравнение Loss по эпохам', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Сохранен график: {os.path.join(output_dir, 'loss_curves.png')}")
    plt.close()
    
    # График 3: F1-score по эпохам
    plt.figure(figsize=(12, 6))
    for result in results:
        train_history = result['train_history']
        val_history = result['val_history']
        config = result['config']
        
        epochs = [h['epoch'] for h in train_history]
        train_f1 = [h['f1'] for h in train_history]
        val_f1 = [h['f1'] for h in val_history]
        
        label = f"{config.get('optimizer', 'unknown').upper()}"
        if config.get('pretrained', False):
            label += " (pretrained)"
        else:
            label += " (scratch)"
        
        plt.plot(epochs, train_f1, 'o-', label=f'Train {label}', linewidth=2, markersize=4)
        plt.plot(epochs, val_f1, 's--', label=f'Val {label}', linewidth=2, markersize=4)
    
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Сравнение F1-score по эпохам', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Сохранен график: {os.path.join(output_dir, 'f1_curves.png')}")
    plt.close()


def plot_test_results_comparison(results, output_dir='plots'):
    """Построить сравнительную диаграмму тестовых результатов"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Подготовка данных
    labels = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for result in results:
        config = result['config']
        test = result['test_results']
        
        label = f"{config.get('optimizer', 'unknown').upper()}"
        if config.get('pretrained', False):
            label += "\n(pretrained)"
        else:
            label += "\n(scratch)"
        
        labels.append(label)
        accuracies.append(test.get('accuracy', 0))
        precisions.append(test.get('precision', 0))
        recalls.append(test.get('recall', 0))
        f1_scores.append(test.get('f1', 0))
    
    # График сравнения метрик
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-score', alpha=0.8)
    
    ax.set_xlabel('Эксперимент', fontsize=12)
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.set_title('Сравнение тестовых метрик по экспериментам', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Добавление значений на столбцы
    for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
        ax.text(i - 1.5*width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Сохранен график: {os.path.join(output_dir, 'test_metrics_comparison.png')}")
    plt.close()


def main():
    """Главная функция"""
    print("Загрузка результатов...")
    results = load_results()
    
    if not results:
        print("Не удалось загрузить результаты.")
        return
    
    print(f"Загружено {len(results)} экспериментов")
    
    print("\nГенерация графиков обучения...")
    plot_training_curves(results)
    
    print("\nГенерация сравнительных графиков...")
    plot_test_results_comparison(results)
    
    print("\nВсе графики сохранены в директории 'plots/'")
    print("Графики можно добавить в отчет ОТЧЕТ.md")


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print("Ошибка: matplotlib не установлен. Установите: pip install matplotlib")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()











