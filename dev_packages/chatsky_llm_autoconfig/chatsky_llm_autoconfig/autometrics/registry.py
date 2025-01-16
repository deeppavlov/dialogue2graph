"""
Algorithm Registry.
--------------------------
This module contains an AlgorithmRegistry class that serves for registering
algorithms for automatic metrics gathering.

"""


class AlgorithmRegistry:
    _algorithms = {}

    @classmethod
    def register(cls, name=None, input_type=None, path_to_result=None, output_type=None):
        def decorator(func):
            algorithm_name = name or func.__name__
            cls._algorithms[algorithm_name] = {"type": func, "input_type": input_type, "path_to_result": path_to_result, "output_type": output_type}
            return func

        return decorator

    @classmethod
    def get_all(cls):
        return cls._algorithms

    @classmethod
    def get_by_category(cls, category):
        return {name: data for name, data in cls._algorithms.items() if data["category"] == category}
