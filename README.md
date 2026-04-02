# Tolstoy-RNN: Character-level Language Model ✍️

Deep learning проект, в котором реализована и обучена рекуррентная нейронная сеть _(RNN/LSTM/GRU)_ для генерации текста в стиле Льва Толстого. Все написано с нуля с использованием PyTorch.

## Краткое описание

Реализован полный цикл (end-to-end pipeline) обучения посимвольной языковой модели (Character-level Language Model). Вместо предсказания слов, модель предсказывает следующий символ в последовательности. Несмотря на простоту, такой подход все же позволяет выучить грамматику, пунктуацию и стилистические паттерны Толстого из сырого текста.

### Особенности

* **Гибкая архитектура**: Поддержка стандартных RNN, GRU и многослойных LSTM архитектур через единый класс модели (`Dispatch Table` паттерн).
* **Train Loop**: Есть отсечение градиентов (gradient clipping), есть кастомная инициализация весов (Orthogonal + Forget Gate Bias для LSTM), и, конечно, есть регуляризация Dropout.
* **Трекинг экспериментов**: Интеграция с **Weights & Biases (W&B)** для мониторинга кривых обучения (Loss) и системных метрик в реальном времени.

## Быстрый старт

### 1. Установка
В проекте используется пакетный менеджер `uv` для быстрого управления зависимостями.

Однако если у вас его нет, можно установить через `pip`.

```bash
# Клонирование репозитория
git clone https://github.com/financier-py/rnn-text-gen.git
cd rnn-text-gen

# Установка зависимостей
uv sync
```

### 2. Подготовка данных

Очищаем текст и генерируем словарь символов. Результат сохраняется в `data/processed/` $:)$

```bash
uv run src/preprocess.py
```

### 3. Обучение модели

Тут вы можете настроить любые свои гиперпараметры в словаре `CONFIG` внутри самого скрипта.

```bash
uv run src/train.py
```

### 4. Генерация текста

После заверщения обучения, все веса сохраняются в `checkpoints/`. Вот они и будут использоваться далее для генерации как раз. Тут вы еще можете поиграться с температурой и с начальной фразой.

```bash
uv run src/generate.py
```

## Архитектура модели

Мной были выбраны следующие слои:

1. **Embedding Layer**: Преобразует индексы символов в плотные векторы.
2. **RNN/GRU/LSTM Layer**: Тот самый рекуррентный слой, который и обрабатывает последовательности символов.
3. **Dropout Layer**: Для регуляризации $:)$
4. **Linear Layer**: Проецирует выход RNN обратно в пространство словаря, выдавая логиты!

## Пример работы

Начальная фраза была: 

```prince andrew```

Выдало: 
```prince andrew. "why do you know it was all right?"

"yes, that's not such a thing."

"oh, you're sure that the count will imagined this to the name of the condition of the enemy's, the enthusiasm that presented the strength of the beloved body and dangerovs, went on."
```

## Стек технологий

* Python 3.11
* PyTorch
* Weights & Biases
* Uv

## Какие-то графики с W&B:

[![W&amp;amp;B Chart 02.04.2026, 17 15 41](https://s3.firstvds.ru/fotohosting/2026/04/02/WB-Chart-02.04.2026-17_15_41.png)](https://fotohosting.pro/i/WB-Chart-02.04.2026%2C-17-15-41.78X65Y)

[![W&amp;amp;B Chart 02.04.2026, 17 15 54](https://s3.firstvds.ru/fotohosting/2026/04/02/WB-Chart-02.04.2026-17_15_54.png)](https://fotohosting.pro/i/WB-Chart-02.04.2026%2C-17-15-54.78XC8q)

## Мои графики:

[![image](https://s3.firstvds.ru/fotohosting/2026/04/02/imagede320e917562096b.png)](https://fotohosting.pro/i/image.78an6Q)

[![image](https://s3.firstvds.ru/fotohosting/2026/04/02/image95628fce0b69226b.png)](https://fotohosting.pro/i/image.78arJ2)

[![image](https://s3.firstvds.ru/fotohosting/2026/04/02/image2b7b6977c3d395db.png)](https://fotohosting.pro/i/image.78aV8o)