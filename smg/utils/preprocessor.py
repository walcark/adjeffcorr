from collections import defaultdict
import numpy as np


class Preprocessor:
    """
    Preprocesses a collection of namedtuple or dict data points by grouping them according to a specified key.
    Returns groups as lists of the original tuple instances, with the grouping key field set to the average raw value of the group.

    Example usage:
        from collections import namedtuple
        Atmosphere = namedtuple('Atmosphere', ['aer_type', 'aer_profile', 'tau_aer', 'tau_ray', 'hmin'])
        data = [
            Atmosphere('type1', 'profile1', 0.2, 0.01, 102),
            Atmosphere('type2', 'profile1', 0.3, 0.02, 148),
            # ...
        ]
        # Group by 'hmin' with discretization step of 50
        prep = Preprocessor(tuple_class=Atmosphere, group_key='hmin', delta=50)
        groups = prep.get_groups(data)
        # groups = [
        #     [Atmosphere('type1','profile1',0.2,0.01,125), Atmosphere('type2','profile1',0.3,0.02,125)],
        #     ...
        # ]
    """
    def __init__(self, tuple_class, group_key: str, delta: float | None = None):
        self.tuple_class = tuple_class
        self.group_key = group_key
        self.delta = delta
        if group_key not in tuple_class._fields:
            raise KeyError(
                f"group_key '{group_key}' not in tuple fields {tuple_class._fields}"
            )

    def _get_value(self, point, key: str):
        """
        Retrieves a field value from a namedtuple or dict point.
        """
        if isinstance(point, dict):
            return point[key]
        return getattr(point, key)

    def _set_value(self, point, key: str, value):
        """
        Returns a new point with the specified key set to the given value.
        """
        if isinstance(point, dict):
            new_point = point.copy()
            new_point[key] = value
            return new_point
        return point._replace(**{key: value})

    def _discretize(self, value: float) -> float:
        """
        Discretizes a numeric value to the nearest multiple of delta.
        """
        return round(value / self.delta) * self.delta

    def group(self, data: list) -> dict:
        """
        Groups data points by the (optionally discretized) grouping key.

        :param data: list of namedtuple or dict points
        :return: dict mapping each discretized group key to list of original points
        """
        grouped = defaultdict(list)
        for point in data:
            raw = self._get_value(point, self.group_key)
            # Apply discretization if needed
            key = (
                self._discretize(raw)
                if self.delta is not None and isinstance(raw, (int, float))
                else raw
            )
            grouped[key].append(point)
        return grouped

    def get_groups(self, data: list) -> list:
        """
        Returns grouped data as a list of lists of tuple_class instances,
        each with their grouping key set to the average raw value of the group.

        :param data: list of namedtuple or dict points
        :return: list of lists, each inner list contains points sharing the same group (discretized) key
        """
        grouped = self.group(data)
        result = []
        for discretized_key, points in grouped.items():
            # compute average raw value
            raw_values = [self._get_value(pt, self.group_key) for pt in points]
            avg_value = np.round(sum(raw_values) / len(raw_values), 5)
            # set each point's grouping field to the average
            normalized = [self._set_value(pt, self.group_key, avg_value) for pt in points]
            result.append(normalized)
        return result


# Demonstration block (optional)
if __name__ == '__main__':
    from collections import namedtuple
    Atmosphere = namedtuple('Atmosphere', ['aer_type', 'aer_profile', 'tau_aer', 'tau_ray', 'hmin'])
    data = [
        Atmosphere('type1', 'profile1', 0.2, 0.01, 120),
        Atmosphere('type1', 'profile2', 0.25, 0.015, 130),
        Atmosphere('type2', 'profile1', 0.3, 0.02, 175),
        Atmosphere('type2', 'profile2', 0.35, 0.025, 180),
        Atmosphere('type2', 'profile3', 0.4, 0.03, 178),
    ]
    prep = Preprocessor(tuple_class=Atmosphere, group_key='hmin', delta=50)
    groups = prep.get_groups(data)
    for group in groups:
        avg = group[0].hmin
        print(f"Group average hmin = {avg}")
        for item in group:
            print("  ", item)
