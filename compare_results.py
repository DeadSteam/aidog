
"""
Скрипт для сравнения результатов различных экспериментов
"""
import json
import os
from pathlib import Path
import pandas as pd


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
                    'test_results': data.get('test_results', {}),
                    'best_val_acc': max([h['accuracy'] for h in data.get('val_history', [])], default=0.0)
                })
        except Exception as e:
            print(f"Ошибка при загрузке {result_file}: {e}")
    
    return results


def print_comparison(results):
    """Вывести сравнение результатов"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("="*80)
    
    # Подготовка данных для таблицы
    comparison_data = []
    
    for result in results:
        config = result['config']
        test = result['test_results']
        
        comparison_data.append({
            'Эксперимент': result['file'],
            'Предобученная': 'Да' if config.get('pretrained', False) else 'Нет',
            'Оптимизатор': config.get('optimizer', 'unknown').upper(),
            'Epochs': config.get('epochs', 'N/A'),
            'LR': config.get('lr', 'N/A'),
            'Test Accuracy': f"{test.get('accuracy', 0):.4f}",
            'Test Precision': f"{test.get('precision', 0):.4f}",
            'Test Recall': f"{test.get('recall', 0):.4f}",
            'Test F1': f"{test.get('f1', 0):.4f}",
            'Best Val Acc': f"{result['best_val_acc']:.4f}"
        })
    
    # Вывод таблицы
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Нахождение лучших результатов
    print("\n" + "="*80)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    
    best_acc = max(results, key=lambda x: x['test_results'].get('accuracy', 0))
    best_prec = max(results, key=lambda x: x['test_results'].get('precision', 0))
    best_rec = max(results, key=lambda x: x['test_results'].get('recall', 0))
    best_f1 = max(results, key=lambda x: x['test_results'].get('f1', 0))
    
    print(f"\nЛучшая Accuracy:  {best_acc['file']}")
    print(f"  Accuracy:  {best_acc['test_results'].get('accuracy', 0):.4f}")
    print(f"  Precision: {best_acc['test_results'].get('precision', 0):.4f}")
    print(f"  Recall:    {best_acc['test_results'].get('recall', 0):.4f}")
    print(f"  F1:        {best_acc['test_results'].get('f1', 0):.4f}")
    
    print(f"\nЛучшая Precision: {best_prec['file']}")
    print(f"  Precision: {best_prec['test_results'].get('precision', 0):.4f}")
    
    print(f"\nЛучшая Recall:    {best_rec['file']}")
    print(f"  Recall:    {best_rec['test_results'].get('recall', 0):.4f}")
    
    print(f"\nЛучшая F1-score:  {best_f1['file']}")
    print(f"  F1:        {best_f1['test_results'].get('f1', 0):.4f}")
    
    print("\n" + "="*80)
    
    # Сравнение предобученных vs с нуля
    pretrained_results = [r for r in results if r['config'].get('pretrained', False)]
    scratch_results = [r for r in results if not r['config'].get('pretrained', False)]
    
    if pretrained_results and scratch_results:
        print("\nСРАВНЕНИЕ: ПРЕДОБУЧЕННАЯ vs С НУЛЯ")
        print("="*80)
        
        avg_pretrained = {
            'accuracy': sum([r['test_results'].get('accuracy', 0) for r in pretrained_results]) / len(pretrained_results),
            'precision': sum([r['test_results'].get('precision', 0) for r in pretrained_results]) / len(pretrained_results),
            'recall': sum([r['test_results'].get('recall', 0) for r in pretrained_results]) / len(pretrained_results),
            'f1': sum([r['test_results'].get('f1', 0) for r in pretrained_results]) / len(pretrained_results)
        }
        
        avg_scratch = {
            'accuracy': sum([r['test_results'].get('accuracy', 0) for r in scratch_results]) / len(scratch_results),
            'precision': sum([r['test_results'].get('precision', 0) for r in scratch_results]) / len(scratch_results),
            'recall': sum([r['test_results'].get('recall', 0) for r in scratch_results]) / len(scratch_results),
            'f1': sum([r['test_results'].get('f1', 0) for r in scratch_results]) / len(scratch_results)
        }
        
        print(f"\nПредобученная модель (среднее):")
        print(f"  Accuracy:  {avg_pretrained['accuracy']:.4f}")
        print(f"  Precision: {avg_pretrained['precision']:.4f}")
        print(f"  Recall:    {avg_pretrained['recall']:.4f}")
        print(f"  F1:        {avg_pretrained['f1']:.4f}")
        
        print(f"\nС нуля (среднее):")
        print(f"  Accuracy:  {avg_scratch['accuracy']:.4f}")
        print(f"  Precision: {avg_scratch['precision']:.4f}")
        print(f"  Recall:    {avg_scratch['recall']:.4f}")
        print(f"  F1:        {avg_scratch['f1']:.4f}")
        
        print(f"\nРазница:")
        print(f"  Accuracy:  {avg_pretrained['accuracy'] - avg_scratch['accuracy']:.4f}")
        print(f"  Precision: {avg_pretrained['precision'] - avg_scratch['precision']:.4f}")
        print(f"  Recall:    {avg_pretrained['recall'] - avg_scratch['recall']:.4f}")
        print(f"  F1:        {avg_pretrained['f1'] - avg_scratch['f1']:.4f}")


def main():
    """Главная функция"""
    results = load_results()
    
    if results:
        print_comparison(results)
    else:
        print("Не удалось загрузить результаты. Убедитесь, что эксперименты завершены.")


if __name__ == '__main__':
    try:
        main()
    except ImportError:
        print("Ошибка: pandas не установлен. Установите: pip install pandas")
    except Exception as e:
        print(f"Ошибка: {e}")



