from typeguard import typechecked


@typechecked
def check_params(q: float, redundancy: float, stages: int):
    if q < 1:
        raise ValueError("q must be greater or equal 1!")
    if redundancy <= 1:
        raise ValueError("The redundancy must be strictly greater than 1")
    if stages < 1:
        raise ValueError("stages must be a positive integer!")
