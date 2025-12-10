@echo off
echo ========================================
echo Установка PyTorch с поддержкой CUDA
echo ========================================
echo.
echo Текущая версия PyTorch:
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.
echo Удаление старой версии PyTorch (CPU)...
pip uninstall torch torchvision torchaudio -y
echo.
echo Установка PyTorch с CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo Проверка установки:
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.
echo ========================================
echo Установка завершена!
echo ========================================
pause










