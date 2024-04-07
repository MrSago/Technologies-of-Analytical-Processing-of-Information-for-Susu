from apriori import apriori, OrderedType
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
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
        table = PrettyTable(["Предметы", "Поддержка"])
        table.align["Предметы"] = "l"
        table.align["Поддержка"] = "l"
        for item_set, support in level.items():
            table.add_row([set(item_set), round(support, 6)])
        print(table)


def print_rules(rules: List[Tuple[FrozenSet, FrozenSet, float]]):
    """Выводит результат алгоритма Априори.

    Аргументы:
        rules (List[Tuple[FrozenSet, FrozenSet, float]]): Результат работы алгоритма Априори.
    """
    table = PrettyTable(["Предпосылка", "Заключение", "Достоверность"])
    table.align["Предпосылка"] = "l"
    table.align["Заключение"] = "l"
    table.align["Достоверность"] = "l"
    for rule in rules:
        antecedent = set(rule[0])
        consequent = set(rule[1])
        confidence = rule[2]
        table.add_row([antecedent, consequent, round(confidence, 6)])
    print(table)


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
    filename: str, min_support: float, confidence_threshold: float
) -> Tuple[
    List[Dict[FrozenSet, float]], List[Tuple[FrozenSet, FrozenSet, float]], float
]:
    """Запускает алгоритм Априори для файла и возращает результат и время выполнения.

    Аргументы:
        filename (str): Имя csv-файла.
        min_support (float): Порог поддержки.

    Возвращает:
        Tuple[List[Dict[FrozenSet, float]], List[Tuple[FrozenSet, FrozenSet, float]], float]:
            Результат работы алгоритма Априори и время выполнения.
    """
    transactions = parse_csv_file(filename)

    start = time()
    frequent_item_sets, rules = apriori(transactions, min_support, confidence_threshold)
    end = time()

    return frequent_item_sets, rules, end - start


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
    print(
        "Задание №1. Демонстрация алгоритма Априори для небольшого набора данных.",
        end="\n\n",
    )

    transactions: List[Set[str]] = [
        {"хлеб", "молоко"},
        {"хлеб", "печенье", "пиво", "яйца"},
        {"молоко", "печенье", "пиво", "кола"},
        {"хлеб", "молоко", "печенье", "пиво"},
        {"хлеб", "молоко", "печенье", "кола"},
    ]

    min_support: float = 0.1
    confidence_threshold: float = 0.1

    frequent_item_sets, rules = apriori(
        transactions, min_support, confidence_threshold, OrderedType.LEXICAL_ASCENDING
    )

    print("Тестовый набор:")
    for transaction in transactions:
        print(f"Транзакция: {transaction}")
    print("", end="\n\n")

    print(f"Минимальная поддержка: {min_support}")
    print(f"Порог достоверности: {confidence_threshold}", end="\n\n")

    print_apriori_result(frequent_item_sets)
    print_rules(rules)
    print("", end="\n")


def second_task():
    print("Задание №2. Экспериментальная проверка алгоритма Apriori.", end="\n\n")

    min_support: float = 0.01
    confidence_threshold: float = 0.3

    frequent_item_sets, rules, calc_time = run_apriori_file(
        TEST_DATA_FILENAME, min_support, confidence_threshold
    )

    print(f"Файл с тестовыми данными: {TEST_DATA_FILENAME}")
    print(f"Минимальная поддержка: {min_support}")
    print(f"Порог достоверности: {confidence_threshold}", end="\n\n")

    print_apriori_result(frequent_item_sets)
    print("\nПравила:", end="\n")
    print_rules(rules)

    print(f"Время работы алгоритма: {round(calc_time, 3):.3f} секунд", end="\n\n")


def third_task():
    print("Задание №3. Визуализация результатов алгоритма Apriori.", end="\n\n")

    min_support: float = 0.01
    confidence_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]

    frequent_item_sets_list: List[List[Set[OrderedType]]] = []
    rules_list: List[List[Tuple[FrozenSet, FrozenSet, float]]] = []
    execution_times: List[float] = []
    for threshold in confidence_thresholds:
        print(f"Старт алгоритма с порогом достоверности: {threshold}...")
        item_sets, rules, calc_time = run_apriori_file(
            TEST_DATA_FILENAME, min_support, threshold
        )
        frequent_item_sets_list.append(item_sets)
        rules_list.append(rules)
        execution_times.append(calc_time)
        print(f"Время работы алгоритма: {round(calc_time, 3):.3f} секунд")
    print("", end="\n")

    make_figure_bar(
        confidence_thresholds,
        execution_times,
        "Порог достоверности",
        "Время работы алгоритма (секунды)",
        "Зависимость времени работы алгоритма от порога достоверности",
    )

    make_figure_bar(
        confidence_thresholds,
        [len(r) for r in rules_list],
        "Порог достоверности",
        "Количество правил",
        "Зависимость количества найденных правил от порога достоверности",
    )

    plt.show()


def main():
    # first_task()
    # second_task()
    third_task()


if __name__ == "__main__":
    main()
