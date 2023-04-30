import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(self, x):
        try:
            if ((isinstance(x, list) and len(x) > 0) or
                    (isinstance(x, ndarray) and x.size > 0)):
                return fun(self, x)
        except Exception as e:
            print(e)
    return wrapper


def typechecker2(fun):
    def wrapper(self, x, p):
        try:
            if ((isinstance(x, list) and len(x) > 0) or
                (isinstance(x, ndarray) and x.size > 0)
                ) and isinstance(p, (int, float)) and 0 <= p <= 100:
                return fun(self, x, p)
        except Exception as e:
            print(e)
    return wrapper


class TinyStatistician:

    def __mean(x, size: int) -> float:
        it = iter(x)
        sum = next(it)
        for i in it:
            sum += i
        return sum / size

    def __percentile(x: list | ndarray, p: int | float) -> float:
        size = len(x)
        q = p * (size - 1) / 100
        index = int(q)
        ret = x[index] * (1.0 - q + index)
        if q - index != 0:
            ret += x[index + 1] * (q - index)
        return ret

    @typechecker
    def mean(self, x: list | ndarray) -> float | None:
        return TinyStatistician.__mean(x, len(x))

    @typechecker
    def median(self, x: list | ndarray) -> float | None:
        return TinyStatistician.__percentile(sorted(x), 50)

    @typechecker
    def quartile(self, x: list | ndarray) -> list[float] | None:
        x = sorted(x)
        return [TinyStatistician.__percentile(x, 25),
                TinyStatistician.__percentile(x, 75)]

    @typechecker2
    def percentile(self, x: list | ndarray, p: float) -> float | None:
        return TinyStatistician.__percentile(sorted(x), p)

    @typechecker
    def var(self, x: list | ndarray) -> float | None:
        len_ = len(x)
        return (TinyStatistician.__mean(map(lambda x: x**2, x), len_) -
                TinyStatistician.__mean(x, len_) ** 2)

    def std(self, x: list | ndarray) -> float | None:
        if (ret := self.var(x)) != None:
            return ret ** (1/2)


if __name__ == '__main__':
    tstat = TinyStatistician()

    a = [1, 42, 300, 10, 59]
    # a = [1]
    a_ndarray = np.array(a)
    print(tstat.mean(a))
    print(tstat.median(a))
    print(tstat.quartile(a))
    print(tstat.percentile(a, 10))
    print(tstat.percentile(a, 15))
    print(tstat.percentile(a, 20))
    print(tstat.percentile(a, 100))
    print(tstat.var(a))
    print(tstat.std(a))
    print()
    print(np.mean(a_ndarray))
    print(np.median(a_ndarray))
    print([np.percentile(a_ndarray, 25), np.percentile(a_ndarray, 75)])
    print(np.percentile(a_ndarray, 10))
    print(np.percentile(a_ndarray, 15))
    print(np.percentile(a_ndarray, 20))
    print(np.percentile(a_ndarray, 100))
    print(np.var(a_ndarray))
    print(np.std(a_ndarray))
