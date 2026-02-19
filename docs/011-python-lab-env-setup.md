# 011 - Python Lab Env Setup

Цель задачи: Подготовить полноценное окружение для Python-лаборатории (python_lab), покрывающее все этапы: от обработки Parquet-данных до обучения LiT-модели, поиска гиперпараметров (Optuna) и экспорта в ONNX. Мы используем современный стек 2026 года с упором на скорость (Polars) и удобство (PyTorch Lightning).

Файлы для изменения/создания:

python_lab/requirements.txt
python_lab/.gitignore
python_lab/README.md
Инструкции для Gemini:

python_lab/requirements.txt:

# Data Engineering (Polars > Pandas)
polars>=1.9.0
pyarrow>=16.0.0
numpy>=2.0.0

# Deep Learning & Training
torch>=2.4.0
pytorch-lightning>=2.2.0
tensorboard>=2.16.0

# Optimization & Analysis
optuna>=3.6.0
shap>=0.45.0
scikit-learn>=1.5.0

# Export & Verification
onnx>=1.16.0
onnxruntime>=1.18.0

# Utils & R&D
tqdm>=4.66.0
matplotlib>=3.9.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
jupyterlab>=4.2.0
python_lab/.gitignore:

venv/
.venv/
__pycache__/
.ipynb_checkpoints/
data/
models/
runs/
logs/
*.parquet
*.onnx
*.pt
*.pth
python_lab/README.md:

# Python Lab for Neirobot LiT

1. Создать виртуальное окружение:
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate    # Windows

2. Установить зависимости:
   pip install -r requirements.txt

3. Запустить среду для экспериментов:
   jupyter lab
Технические требования:

Использовать pytorch-lightning для автоматизации циклов обучения и интеграции с Optuna.
onnxruntime автоматически подхватит GPU, если установлены драйверы CUDA.
polars обеспечит мгновенную загрузку Parquet-файлов, созданных в Rust.
Почему это важно: Этот набор библиотек гарантирует, что мы сможем не только обучить модель, но и проверить её качество (F1/MCC через scikit-learn), интерпретировать её решения (SHAP) и корректно подготовить для инференса в Rust через ONNX.