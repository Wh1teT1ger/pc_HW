# Домашнее задание по параллельным вычислениям
## Поиск прообраза методом перебора для нерегулярной композиции хэш-функций
Выполнили:
* Артамонова Илюза (ИУ8-113)
* Артамонов Марк (ИУ8-115)
* Бурылов Денис (ИУ8-115)

## Установка
1. Клонирование репозитория:
```bash
git clone https://github.com/Wh1teT1ger/pc_HW.git
cd pc_HW
```
2. Сборка проекта:
```bash
mkdir build
cd build
cmake ..
make
```

## Запуск
Поиск прообраза методом перебора:
```bash
./bf_hash  --target_hash VAR --hash_sequence VAR... [--charset VAR] [--max_length VAR] [--gpu]
```
### Аргументы
* `target_hash`: Хеш в виде шестнадцатеричной строки (например, `5e884898da28047151d0e56f8dc62927`).
* `hash_sequence`: Последовательность хешей (например, `md5 sha1 sha256`).
* `charset`: Набор символов для генерации строк-кандидатов (по умодчанию: `abcdefghijklmnopqrstuvwxyz`).
* `max_length`: Максимальная длина строки-кандидата (по умодчанию: `6`).
* `gpu`: Запуск перебора на gpu (по умолчанию на cpu).
### Пример
Запуск для хеша `5882f83b17620d005f291df57d46c53f` и композиции `sha256 sha1 md5`:
```bash
./bf_hash -t 5882f83b17620d005f291df57d46c53f -s sha256 sha1 md5 -m 6 --gpu
```
Вывод:
```bash
Target hash: 5882f83b17620d005f291df57d46c53f
Hash sequence: sha256 sha1 md5 
Start on gpu...
Match found: qwerty
Execution time: 32759 ms
```
## Получение хеша
```bash
./hash --target VAR --hash_sequence VAR...
```
### Аргументы
* `target_hash`: Прообраз  хеша (например, `qwerty`).
* `hash_sequence`: Последовательность хешей (например, `md5 sha1 sha256`).
### Пример
Запуск для хеша `5882f83b17620d005f291df57d46c53f` и композиции sha256 → sha1 → md5:
```bash
./hash -t qwerty -s sha256 sha1 md5
```
Вывод:
```bash
5882f83b17620d005f291df57d46c53f
```

## Сравнение времени перебора на CPU и GPU
Сравнение прозводительности на CPU Intel Core i5 6500 и GPU Nvidia RTX 4070

| Длина слова | CPU       | GPU      |
|:-----------:|:---------:|:--------:|
| 4           | 680 ms    | 69 ms    |
| 5           | 16580 ms  | 187 ms   |
| 6           | 428474 ms | 2961 ms  |

![График](/images/compere.png "Сравнение времени перебора на CPU и GPU")