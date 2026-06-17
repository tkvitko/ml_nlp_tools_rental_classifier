# Система классификации обращений клиентов

## Инсталляция
```
pip install -r requirements.txt
```

## Структура репозитория
### training
- step_1_dataset_analyse.ipynb - подготовка базового датасета
- step_2_dataset_extend.ipynb - набор дополнительных примеров через LLM
- step_3_train_on_base_dataset.ipynb - тренировка моделей на базовом датасете
- step_4_train_on_externded_dataset.ipynb - тренировка моделей на расширенном датасете
- step_5_train_tuning_hyperparams.ipynb - подбор гиперпараметров
### models
- обученные модели (NDA) 
### dataset
- texts_from_avito.csv - исходный датасет (NDA)
- extended_by_deepseek_3.csv - расширенный датасет (NDA)
- labels.csv - итоговые метки для обучения
### config
- HF_TOKEN.txt - huggingface api token
- deepseek_api_key.txt - deepseek api key


## Скрипты
1. Тестовый запуск модели для ввода текста и получения результата классификации
```
python test_model
```

2. Тестовый запуск обращения к deepseek
```
python test_deepseek_request
```
