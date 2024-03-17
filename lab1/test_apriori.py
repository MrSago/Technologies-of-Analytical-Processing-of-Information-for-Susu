import unittest
from apriori import (
    OrderedType,
    _calc_support,
    _calc_support_candidate,
    _apriori_gen,
    _sort_items,
    apriori,
)


class TestAprioriAlgorithm(unittest.TestCase):
    def test_calc_support(self):
        data_set = [{"a", "b"}, {"b", "c", "a"}, {"a", "c"}, {"a", "b", "c", "e"}]

        item_set = frozenset(["a"])
        excepted = 1.0
        self.assertEqual(_calc_support(item_set, data_set), excepted)

        item_set = frozenset(["b", "c"])
        excepted = 0.5
        self.assertEqual(_calc_support(item_set, data_set), excepted)

    def test_calc_support_candidate(self):
        data_set = [{"a", "b"}, {"b", "c", "a"}, {"a", "c"}, {"a", "b", "c", "e"}]
        candidate = {
            frozenset(["a"]),
            frozenset(["b"]),
            frozenset(["c"]),
            frozenset(["a", "b"]),
        }
        min_support = 0.75
        expected = {
            frozenset(["a"]): 1.0,
            frozenset(["b"]): 0.75,
            frozenset(["c"]): 0.75,
            frozenset(["a", "b"]): 0.75,
        }
        self.assertEqual(
            _calc_support_candidate(data_set, candidate, min_support), expected
        )

    def test_apriori_gen(self):
        item_set = {frozenset(["a"]), frozenset(["b"]), frozenset(["c"])}
        length = 2
        expected = {frozenset(["a", "b"]), frozenset(["a", "c"]), frozenset(["b", "c"])}
        self.assertEqual(_apriori_gen(item_set, length), expected)

    def test_sort_items_support_ascending(self):
        input_dict = {"a": 2, "b": 1, "c": 3}
        expected = {"b": 1, "a": 2, "c": 3}
        self.assertEqual(
            _sort_items(input_dict, OrderedType.SUPPORT_ASCENDING), expected
        )

    def test_sort_items_support_descending(self):
        input_dict = {"a": 2, "b": 1, "c": 3}
        expected = {"c": 3, "a": 2, "b": 1}
        self.assertEqual(
            _sort_items(input_dict, OrderedType.SUPPORT_DESCENDING), expected
        )

    def test_sort_items_lexical_ascending(self):
        input_dict = {("b",): 2, ("a",): 1, ("c",): 3}
        expected = {("a",): 1, ("b",): 2, ("c",): 3}
        self.assertEqual(
            _sort_items(input_dict, OrderedType.LEXICAL_ASCENDING), expected
        )

    def test_sort_items_lexical_descending(self):
        input_dict = {("b",): 2, ("a",): 1, ("c",): 3}
        expected = {("c",): 3, ("b",): 2, ("a",): 1}
        self.assertEqual(
            _sort_items(input_dict, OrderedType.LEXICAL_DESCENDING), expected
        )

    def test_apriori(self):
        data_set = [
            {"хлеб", "молоко"},
            {"хлеб", "печенье", "пиво", "яйца"},
            {"молоко", "печенье", "пиво", "кола"},
            {"хлеб", "молоко", "печенье", "пиво"},
            {"хлеб", "молоко", "печенье", "кола"},
        ]
        min_support = 0.5
        expected = [
            {
                frozenset(["пиво"]): 0.6,
                frozenset(["хлеб"]): 0.8,
                frozenset(["печенье"]): 0.8,
                frozenset(["молоко"]): 0.8,
            },
            {
                frozenset(["хлеб", "печенье"]): 0.6,
                frozenset(["хлеб", "молоко"]): 0.6,
                frozenset(["пиво", "печенье"]): 0.6,
                frozenset(["печенье", "молоко"]): 0.6,
            },
        ]
        self.assertEqual(
            apriori(data_set, min_support, OrderedType.LEXICAL_ASCENDING), expected
        )


if __name__ == "__main__":
    unittest.main()
