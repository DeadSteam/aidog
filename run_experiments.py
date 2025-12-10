"""
Скрипт для запуска всех экспериментов
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Запустить команду и вывести результат"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Команда: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"ОШИБКА при выполнении: {description}")
        return False
    else:
        print(f"✓ Успешно: {description}")
        return True


def main():
    """Запуск всех экспериментов"""
    
    print("="*60)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ ПО ОБУЧЕНИЮ VISION TRANSFORMER")
    print("="*60)
    
    # Проверка наличия разделенного датасета
    if not os.path.exists('archive/images/train'):
        print("\nРазделение датасета на train/val/test...")
        if not run_command('python split_dataset.py', 'Разделение датасета'):
            print("Ошибка при разделении датасета!")
            sys.exit(1)
    else:
        print("\nДатасет уже разделен, пропускаем split_dataset.py")
    
    # Параметры экспериментов
    experiments = [
        {
            'name': 'Эксперимент 1: Дообучение предобученной модели с Adam',
            'cmd': 'python train.py --pretrained --optimizer adam --epochs 30 --lr 1e-4 --batch_size 32',
            'save_name': 'exp1_pretrained_adam'
        },
        {
            'name': 'Эксперимент 2: Дообучение предобученной модели с SPAM',
            'cmd': 'python train.py --pretrained --optimizer spam --epochs 30 --lr 1e-3 --batch_size 32',
            'save_name': 'exp2_pretrained_spam'
        },
        {
            'name': 'Эксперимент 3: Обучение с нуля с Adam',
            'cmd': 'python train.py --optimizer adam --epochs 40 --lr 1e-4 --batch_size 32',
            'save_name': 'exp3_from_scratch_adam'
        },
    ]
    
    results_summary = []
    
    # Запуск экспериментов
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*60}")
        print(f"ЭКСПЕРИМЕНТ {i}/{len(experiments)}")
        print(f"{'#'*60}")
        
        success = run_command(exp['cmd'], exp['name'])
        results_summary.append({
            'name': exp['name'],
            'success': success
        })
    
    # Вывод итогов
    print("\n\n" + "="*60)
    print("ИТОГИ ЭКСПЕРИМЕНТОВ")
    print("="*60)
    
    for result in results_summary:
        status = "✓ Успешно" if result['success'] else "✗ Ошибка"
        print(f"{status}: {result['name']}")
    
    print("\n" + "="*60)
    print("Для сравнения результатов проверьте файлы в директории checkpoints/")
    print("="*60)


if __name__ == '__main__':
    main()



