from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)
from enum import Enum


class OrderedType(int, Enum):
    SUPPORT_ASCENDING = 1
    SUPPORT_DESCENDING = 2
    LEXICAL_ASCENDING = 3
    LEXICAL_DESCENDING = 4


def _calc_support(item_set: Set[int], data_set: List[Set[int]]) -> float:
    """Вычисляет поддержку для набора элементов.

    Аргументы:
        item_set (Set[int]): Набор элементов для вычисления поддержки.
        data_set (List[Set[int]]): Набор данных для вычисления поддержки в.

    Возвращает:
        float: Поддержка набора элементов.
    """
    return sum([item_set.issubset(transaction) for transaction in data_set]) / len(
        data_set
    )


def _calc_support_candidate(
    data_set: List[Set[int]], candidate: Set[Tuple[int]], min_support: int
) -> Dict[Tuple[int], float]:
    """Вычисляет поддержку для наборов элементов-кандидатов.

    Аргументы:
        data_set (List[Set[int]]): Набор данных, для которого вычисляется поддержка.
        candidate (Set[Tuple[int]]): Наборы элементов-кандидатов, для которого вычисляется поддержка.
        min_support (int): Минимальная поддержка, необходимая для того, чтобы набор элементов считался часто встречающимся.

    Возвращает:
        Dict[Tuple[int], float]: Словарь, содержащий поддержку для каждого набора элементов-кандидата, где ключи - наборы элементов, а значения - их поддержки.
    """
    return {
        item: _calc_support(item, data_set)
        for item in candidate
        if (_calc_support(item, data_set) >= min_support)
    }


def _apriori_gen(item_set: Set[Tuple[int]], length: int) -> Set[Tuple[int]]:
    """Генерирует наборы элементов конкретной длины.

    Аргументы:
        item_set (Set[Tuple[int]]): Набор наборов элементов, на основе которого генерируются кандидаты.
        length (int): Длина генерируемых наборов элементов.

    Возвращает:
        Set[Tuple[int]]: Набор созданных наборов элементов.
    """
    return set(
        [i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length]
    )


def _sort_items(
    dict_items: Dict[Any, float], ordered: "OrderedType"
) -> Dict[Any, float]:
    """Сортирует элементы в словаре по поддержке или лексикографическому порядку.

    Аргументы:
        dict_items (Dict[Any, float]): Словарь для сортировки.
        ordered (OrderedType): Тип порядка сортировки.

    Возвращает:
        Dict[Any, float]: Сorted словарь.
    """
    if ordered == OrderedType.SUPPORT_ASCENDING:
        return {k: v for k, v in sorted(dict_items.items(), key=lambda item: item[1])}
    elif ordered == OrderedType.SUPPORT_DESCENDING:
        return {
            k: v
            for k, v in sorted(
                dict_items.items(), key=lambda item: item[1], reverse=True
            )
        }
    elif ordered == OrderedType.LEXICAL_ASCENDING:
        return dict(sorted(dict_items.items(), key=lambda item: tuple(sorted(item[0]))))
    elif ordered == OrderedType.LEXICAL_DESCENDING:
        return dict(
            sorted(
                dict_items.items(),
                key=lambda item: tuple(sorted(item[0], reverse=True)),
                reverse=True,
            )
        )


def apriori(
    data_set: List[Set[Any]],
    min_support: int,
    order: Optional[OrderedType] = None,
) -> List[Dict[FrozenSet[Any], float]]:
    """Алгоритм Априори для поиска часто встречающихся множеств элементов.

    Аргументы:
        data_set (List[Set[Any]]): Входные данные.
        min_support (int): Порог поддержки.
        order (Optional[OrderedType]): Порядок сортировки.

    Возвращает:
        List[Dict[FrozenSet[Any], float]]: Список часто встречающихся множеств элементов и их поддержки.
    """
    # Генерируем начальные кандидаты для часто встречающихся множеств (C1)
    C1: Set[FrozenSet[Any]] = {
        frozenset([item]) for transaction in data_set for item in transaction
    }

    # Вычисляем поддержку для каждого кандидата в C1
    L1: Dict[FrozenSet[Any], float] = _calc_support_candidate(data_set, C1, min_support)
    if order:
        L1 = _sort_items(L1, order)

    # Сохраняем часто встречающиеся множества и их поддержку
    L: List[Dict[FrozenSet[Any], float]] = [L1]
    k: int = 2
    while True:
        # Генерируем новое множество кандидатов
        Ck: Set[FrozenSet[Any]] = _apriori_gen(L[k - 2], k)
        # Вычисляем поддержку для каждого кандидата в Ck
        Lk: Dict[FrozenSet[Any], float] = _calc_support_candidate(
            data_set, Ck, min_support
        )
        if not Lk:
            break
        if order:
            Lk = _sort_items(Lk, order)
        L.append(Lk)
        k += 1

    return L
