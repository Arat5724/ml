from matrix import Matrix, Vector

v1 = Vector([[1, 2, 3]])  # create a row vector
v2 = Vector([[1], [2], [3]])  # create a column vector
# v3 = Vector([[1, 2], [3, 4]])  # return an error

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
print(m1.T())
print(m1.T().shape)

m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
print(m1.T())
print(m1.T().shape)

m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
             [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0],
             [6.0, 7.0]])
print(m1 * m2)


m1 = Matrix([[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]])

print(1 / m1)

m1 = Matrix([[0.0, 1.0, 2.0],
             [0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1)

v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
print(v1.dot(v2))
