from apriori import apriori, OrderedType
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from typing import List, Tuple, Dict, Set, FrozenSet


TEST_DATA_FILENAME: str = "baskets.csv"


def print_apriori_result(item_sets: List[Dict[FrozenSet, float]]):
    """Выводит результат алгоритма Априори.

    Аргументы:
        item_sets (List[Dict[FrozenSet, float]]): Результат работы алгоритма Априори.
    """
    for i, level in enumerate(item_sets):
        print(f"Частота предметов из набора длины {i+1}:")
        for item_set, support in level.items():
            print(f"    Предметы: {set(item_set)}, Поддержка: {round(support, 8):.8f}")


def parse_csv_file(filename: str) -> List[Set[str]]:
    """Парсит файл csv и возвращает список транзакций.

    Аргументы:
        filename (str): Имя файла csv.

    Возвращает:
        List[Set[str]]: Список транзакций.
    """
    csv_file = pd.read_csv(filename, encoding="windows-1251")
    data = []
    for _, row in csv_file.iterrows():
        data_row = []
        for _, item in enumerate(row):
            if pd.isna(item):
                continue
            data_row.append(str(item))
        data.append(set(data_row))
    return data


def run_apriori_file(
    filename: str, min_support: float
) -> Tuple[List[Dict[FrozenSet, float]], float]:
    """Запускает алгоритм Априори для файла и возращает результат и время выполнения.

    Аргументы:
        filename (str): Имя csv-файла.
        min_support (float): Порог поддержки.

    Возвращает:
        Tuple[List[Dict[FrozenSet, float]], float]: Результат работы алгоритма Априори и время выполнения.
    """
    transactions = parse_csv_file(filename)

    start = time()
    frequent_item_sets = apriori(transactions, min_support)
    end = time()

    return frequent_item_sets, end - start


def make_figure_bar(
    x: List[float], y: List[float], x_label: str, y_label: str, title: str
):
    """Создает график бара.

    Аргументы:
        x (List[float]): Координаты центров бара по оси X.
        y (List[float]): Высоты бара.
        x_label (str): Метка для оси X.
        y_label (str): Метка для оси Y.
        title (str): Заголовок графика.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(
        x,
        y,
        width=0.015,
        alpha=0.5,
        edgecolor="k",
        linewidth=1,
        color="b",
    )
    plt.gcf().canvas.manager.set_window_title(title)
    plt.title(f"{title} (файл {TEST_DATA_FILENAME})")
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def first_task():
    print("Задание №1. Демонстрация алгоритма Априори для небольшого набора данных.")

    transactions: List[Set[str]] = [
        {"хлеб", "молоко"},
        {"хлеб", "печенье", "пиво", "яйца"},
        {"молоко", "печенье", "пиво", "кола"},
        {"хлеб", "молоко", "печенье", "пиво"},
        {"хлеб", "молоко", "печенье", "кола"},
    ]

    min_support: float = 0.5

    frequent_item_sets: List[Set[FrozenSet[str]]] = apriori(
        transactions, min_support, OrderedType.LEXICAL_ASCENDING
    )

    print("Тестовый набор:")
    for transaction in transactions:
        print(f"Транзакция: {transaction}")
    print(f"Минимальная поддержка: {min_support}")
    print_apriori_result(frequent_item_sets)
    print("", end="\n")


def second_task():
    print("Задание №2. Экспериментальная проверка алгоритма Apriori.")

    frequent_item_sets, calc_time = run_apriori_file(TEST_DATA_FILENAME, 0.01)

    print_apriori_result(frequent_item_sets)

    print(f"Время работы алгоритма: {round(calc_time, 3):.3f} секунд", end="\n\n")


def third_task():
    print("Задание №3. Визуализация результатов алгоритма Apriori.")

    theresholds: List[float] = [0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15]

    execution_times: List[float] = []
    frequent_item_sets_list: List[List[Set[OrderedType]]] = []
    for threshold in theresholds:
        print(f"Старт алгоритма с порогом поддержки: {threshold}...")
        item_sets, calc_time = run_apriori_file(TEST_DATA_FILENAME, threshold)
        frequent_item_sets_list.append(item_sets)
        execution_times.append(calc_time)
        print(f"Время работы алгоритма: {round(calc_time, 3):.3f} секунд")
    print("", end="\n")

    freq_cnts: List[int] = [
        sum(len(item_set) for item_set in level) for level in frequent_item_sets_list
    ]

    make_figure_bar(
        theresholds,
        execution_times,
        "Порог поддержки",
        "Время работы алгоритма (секунды)",
        "Зависимость времени работы алгоритма от порога поддержки",
    )

    make_figure_bar(
        theresholds,
        freq_cnts,
        "Порог поддержки",
        "Количество частых наборов",
        "Зависимость количества частых наборов от порога поддержки",
    )

    plt.show()


def main():
    first_task()
    second_task()
    third_task()


if __name__ == "__main__":
    main()
