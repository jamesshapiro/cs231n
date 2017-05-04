#!/usr/bin/env python3 -tt

import numpy as np

a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a))           # Prints "<type 'numpy.ndarray'>"
print(a.shape)           # Prints "(3,)"
print(a[0], a[1], a[2])  # Prints "1 2 3"
a[0] = 5                 # Change an element of the array
print(a)                 # Prints "[5 2 3]"

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b.shape)                    # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])  # Prints "1 2 4"

a = np.zeros((2,2))
print(a)
'''
[[ 0.  0.]
 [ 0.  0.]]
'''
b = np.ones((1,2))
print(b)
'''
[[ 1.  1.]]
'''
type(b[0][0])
# <class 'numpy.float64'>


c = np.full((2,2), 7)
print(c)
'''
[[7 7]
 [7 7]]
'''
type(c[0][0])
# <class 'numpy.int64'>

d = np.eye(2)
print(d)
'''
[[ 1.  0.]
 [ 0.  1.]]
'''

e = np.random.random((2,2))
print(e)
'''
[[ 0.30872625  0.65263754]
 [ 0.75989337  0.49955559]]
'''

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
'''
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
'''
print(a[1:2,1:3])
# [[6 7]]
print(a[1:,:3])
'''
[[ 5  6  7]
 [ 9 10 11]]
'''
print(a[0, 1])
# 2
b = a[:2, 1:3]
b[0, 0] = 77
print(a[0, 1])
# 77

#============================
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]
print(row_r1)
# [5 6 7 8]
row_r2 = a[1:2, :]
# [[5 6 7 8]]

print(row_r1, row_r1.shape)
# [5 6 7 8] (4,)

print(row_r2, row_r2.shape)
# [[5 6 7 8]] (1, 4)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]

print(col_r1, col_r1.shape)
#[ 2  6 10] (3,)

print(col_r2, col_r2.shape)
'''
[[ 2]
 [ 6]
 [10]] (3, 1)
Interesting!
'''

#============================

a = np.array([[1,2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])
# [1 4 5]
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
# [1 4 5]

print(a[[0, 0], [1, 1]])
# [2 2]

print(np.array([a[0, 1], a[0, 1]]))
# [2 2]

print([a[0, 1], a[0, 1]])
# [2, 2]. Note, this is a list of numpy.int64s, not an np.array
nums = [a[0, 1], a[0, 1]]
type(nums)
# <class 'list'>
type(nums[0])
# <class 'numpy.int64'>

#============================
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)
b = np.array([0, 2, 0, 1])
arange = a[np.arange(4), b]
print(arange, type(arange))
# [ 1  6  7 11] <class 'numpy.ndarray'>

a[np.arange(4), b] += 10
a
'''
array([[11,  2,  3],
       [ 4,  5, 16],
       [17,  8,  9],
       [10, 21, 12]])'''

#============================
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 3)
print(bool_idx)
'''
[[False False]
 [False  True]
 [ True  True]]
'''

print(a > 3)
'''
[[False False]
 [False  True]
 [ True  True]]
'''
print(a[bool_idx])
# [4 5 6]
print(type(a[bool_idx]))
# <class 'numpy.ndarray'>

type(bool_idx)
#<class 'numpy.ndarray'>

type(a > 3)
#<class 'numpy.ndarray'>
type((a > 3)[0][0])
# <class 'numpy.bool_'>

b = np.array([[-1,2], [3, -4], [-5, 6]])
a > b
'''
array([[ True, False],
       [False,  True],
       [ True, False]], dtype=bool)
'''
