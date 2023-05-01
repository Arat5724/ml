class Matrix:

    def __init__(self, obj):
        if isinstance(obj, list):
            self.shape = (len(obj), len(obj[0]))
            for row in obj:
                if isinstance(row, list) and (len(row) == self.shape[1]):
                    for col in row:
                        if not isinstance(col, (int, float, bool)):
                            raise Exception
                else:
                    raise Exception
            self.data = obj
        elif isinstance(obj, tuple):
            m, n = obj
            self.shape = (m, n)
            if m > 0 and n >= 0:
                self.data = [[0.0 for _ in range(n)] for __ in range(m)]
            else:
                raise Exception
        else:
            raise Exception

    def T(self):
        m, n = self.shape
        return type(self)([[self.data[j][i] for j in range(m)] for i in range(n)])

    def issameshape(fun):
        def wrapper(self, obj):
            if not isinstance(obj, Matrix) or self.shape != obj.shape:
                raise TypeError("only matrices of same dimensions")
            return fun(self, obj)
        return wrapper

    @issameshape
    def __add__(self, obj):
        res = [[c1 + c2 for c1, c2 in zip(r1, r2)]
               for r1, r2 in zip(self.data, obj.data)]
        if type(self) == Vector or type(obj) == Vector:
            return Vector(res)
        return Matrix(res)

    @issameshape
    def __radd__(self, obj):
        return obj.__add__(self)

    @issameshape
    def __sub__(self, obj):
        res = [[c1 - c2 for c1, c2 in zip(r1, r2)]
               for r1, r2 in zip(self.data, obj.data)]
        if type(self) == Vector or type(obj) == Vector:
            return Vector(res)
        return Matrix(res)

    @issameshape
    def __rsub__(self, obj):
        return obj.__rsub__(self)

    def isscalar(fun):
        def wrapper(self, obj):
            if not isinstance(obj, (int, float, bool)):
                raise TypeError("only scalars")
            return fun(self, obj)
        return wrapper

    @isscalar
    def __truediv__(self, obj):
        return type(self)([[c / obj for c in r] for r in self.data])

    def __rtruediv__(self, obj):
        raise NotImplementedError(
            "Division of a scalar by a Matrix is not defined here.")

    def issvm(i, j):
        def wrapper1(fun):
            def wrapper2(self, obj):
                if isinstance(obj, Matrix):
                    if self.shape[i] != obj.shape[j]:
                        raise TypeError("invalid shape")
                elif not isinstance(obj, (int, float, bool)):
                    raise TypeError("scalars, vectors and matrices")
                return fun(self, obj)
            return wrapper2
        return wrapper1

    @issvm(1, 0)
    def __mul__(self, obj):
        if isinstance(obj, (int, float, bool)):
            return [[c * obj for c in r] for r in self.data]
        a, b = self.shape
        _, c = obj.shape

        def _(i, j):
            r = 0
            for k in range(b):
                r += self.data[i][k] * obj.data[k][j]
            return r
        res = [[_(i, j) for j in range(c)] for i in range(a)]
        if type(self) == Vector or type(obj) == Vector:
            return Vector(res)
        return Matrix(res)

    @issvm(0, 1)
    def __rmul__(self, obj):
        return obj.__mul__(self)

    def __str__(self):
        res = f'{type(self).__name__}\n['
        for r in self.data:
            res += str(r) + '\n'
        res = res[:-1] + ']'
        return res

    def __repr__(self):
        return self.__str__()


class Vector(Matrix):
    def __init__(self, obj):
        super().__init__(obj)
        m, n = self.shape
        if not (m == 1 or n == 1):
            raise TypeError("data must be a row or column vector")
