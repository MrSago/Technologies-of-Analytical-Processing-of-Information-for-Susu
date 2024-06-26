import apyori
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
        confidence_threshold = 0.5

        expected_supports = [
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
        expected_rules = frozenset(
            [
                (frozenset({"печенье"}), frozenset({"молоко"}), 0.7499999999999999),
                (frozenset({"молоко"}), frozenset({"печенье"}), 0.7499999999999999),
                (frozenset({"хлеб"}), frozenset({"молоко"}), 0.7499999999999999),
                (frozenset({"молоко"}), frozenset({"хлеб"}), 0.7499999999999999),
                (frozenset({"пиво"}), frozenset({"печенье"}), 1.0),
                (frozenset({"печенье"}), frozenset({"пиво"}), 0.7499999999999999),
                (frozenset({"хлеб"}), frozenset({"печенье"}), 0.7499999999999999),
                (frozenset({"печенье"}), frozenset({"хлеб"}), 0.7499999999999999),
            ]
        )

        supports, rules = apriori(
            data_set,
            min_support,
            confidence_threshold,
            OrderedType.LEXICAL_ASCENDING,
        )
        rules = frozenset(rules)

        self.assertEqual(supports, expected_supports)
        self.assertEqual(rules, expected_rules)

    def test_apriori_with_apyori(self):
        data_set = [
            {"хлеб", "молоко"},
            {"хлеб", "печенье", "пиво", "яйца"},
            {"молоко", "печенье", "пиво", "кола"},
            {"хлеб", "молоко", "печенье", "пиво"},
            {"хлеб", "молоко", "печенье", "кола"},
        ]
        min_support = 0.5
        confidence_threshold = 0.5

        apriori_supports, apriori_rules = apriori(
            data_set, min_support, confidence_threshold, OrderedType.LEXICAL_ASCENDING
        )
        apyori_supports = apyori.apriori(data_set, min_support=min_support)
        apyori_supports_converted = self._convert_apyori_result(apyori_supports)

        self.assertEqual(apriori_supports, apyori_supports_converted)

    def _convert_apyori_result(self, apyori_result):
        apyori_adapted = []
        k = 1
        transaction = {}
        for rule in apyori_result:
            if len(rule.items) > k:
                apyori_adapted.append(transaction)
                transaction = {}
                k += 1
            transaction[rule.items] = rule.support
        if transaction:
            apyori_adapted.append(transaction)
        return apyori_adapted


if __name__ == "__main__":
    unittest.main()
