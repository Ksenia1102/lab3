# ЛР №3. Линейные модели

## Содержимое:

- linear_models.py — решения: собственная линейная регрессия, собственная логистическая регрессия, сравнение с готовыми моделями sklearn
- run_experiments.py — подготовка данных, эксперименты, запись experiment_results.json
- experiment_results.json — измеренные метрики (линейная регрессия и 4 классификатора)
- REPORT.md — отчёт: данные, гипотезы, метрики, выводы
- Student_Performance.csv, german.csv — данные
- grader_contracts/ — датаклассы

## Быстрый старт

Установка зависимостей

```
pip install -r requirements.txt
```

## Проверка:

1. Запуск экспериментов (печатает все числа из отчёта и перезаписывает experiment_results.json)

```
python run.py
```

2. Сверка результатов — числа в консоли должны совпасть с experiment_results.json и таблицами в REPORT.md