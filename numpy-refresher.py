#!/usr/bin/env python3 -tt

import numpy as np

#==================== Arrays ====================

a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a))           # Prints "<type 'numpy.ndarray'>"
print(a.shape)           # Prints "(3,)"
print(a[0], a[1], a[2])  # Prints "1 2 3"
a[0] = 5                 # Change an element of the array
print(a)                 # Prints "[5 2 3]"

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b.shape)                    # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])  # Prints "1 2 4"

#================ Array Creation ================

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

#================ Array Indexing ================


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

#================ Array Slicing =================

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

#============ Integer Array Indexing ============

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

#====== Integer Array Indexing + Mutating =======

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

#============ Boolean Array Indexing ============

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
(a > b).all()
# False

(a > b).any()
# True

#================== Datatypes ===================

x = np.array([1, 2])
print(x.dtype)

# int64

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1, 2], dtype=np.int64)
print(x.dtype)

x = np.array([1.0, 2.5], dtype=np.int64)
print(x)
# [1 2]
print(x.dtype)
# int64

#================== Array Math ==================

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))
'''
Same result:
[[  6.   8.]
 [ 10.  12.]]
'''
print(x - y)
print(np.subtract(x, y))
'''
Same result:
[[-4. -4.]
 [-4. -4.]]
'''
print(x * y)
print(np.multiply(x, y))
'''
Same result:
[[  5.  12.]
 [ 21.  32.]]

# Note that * is element-wise multiplication,
# not matrix multiplication (which it would
# be in say, MATLAB)
# For matrix multiplication, we use "dot"
# See next section for examples.

'''


print(x / y)
print(np.divide(x, y))
'''
Same result:
[[ 0.2         0.33333333]
 [ 0.42857143  0.5       ]]
'''
print(np.sqrt(x))
'''
[[ 1.          1.41421356]
 [ 1.73205081  2.        ]]
'''

#================= Dot Product ==================

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
# 219
print(np.dot(v, w))
# 219

print(x.dot(v))
print(np.dot(x, v))
# [29 67]
type(np.dot(x, v))
# <class 'numpy.ndarray'>
print(x.dot(y))
print(np.dot(x, y))
'''
[[19 22]
 [43 50]]
'''
print(np.outer(w, v))
'''
[[ 99 110]
 [108 120]]
'''
print(np.outer(v, w))
'''
[[ 99 108]
 [110 120]]
'''

#===================== Sum ======================

x = np.array([[1,2],[3,4]])

print(np.sum(x))
# 10
print(np.sum(x, axis=0))
# [4 6]
print(np.sum(x, axis=1))
# [3 7]

#================== Transpose ===================

# Note: It's interesting how vectors are treated
# as vectors (as opposed to nx1 or 1xn matrices)

x = np.array([[1,2], [3,4]])
print(x)
'''
[[1 2]
 [3 4]]
'''
print(x.T)
'''
[[1 3]
 [2 4]]
'''

v = np.array([1,2,3])
vT = v.T
np.dot(v, v.T)
# 14
np.dot(v.T, v)
# 14
print(v)
# [1 2 3]
print(v.T)
# [1 2 3]
v.T.shape
# (3,)
v.shape
# (3,)

v.dot(np.ones((3,3)))
# array([ 6.,  6.,  6.])
np.ones((3,3)).dot(v)
# array([ 6.,  6.,  6.])

#================ Broadcasting ==================

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])

y = np.empty_like(x)
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)
'''
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
'''

''' More efficient alternative: '''
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))
print(vv)
'''
[[1 0 1]
 [1 0 1]
 [1 0 1]
 [1 0 1]]
'''

y = x + vv
print(y)
'''
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
'''

# Most efficient: just use broadcasting!
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v
print(y)
'''
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
'''
v = np.array([1, 0])
y = x + v
# ValueError: operands could not be broadcast together with shapes (4,3) (2,)

#========== Broadcasting Applications ===========

v = np.array([1,2,3])
w = np.array([4,5])

print(np.reshape(v, (3, 1)) * w)
'''
[[ 4  5]
 [ 8 10]
 [12 15]]
'''

x = np.array([[1,2,3], [4,5,6]])
print(x + v)
'''
[[2 4 6]
 [5 7 9]]
'''

print((x.T + w).T)
'''
[[ 5  6  7]
 [ 9 10 11]]
'''

print(x + np.reshape(w, (2, 1)))
'''
[[ 5  6  7]
 [ 9 10 11]]
'''
print(x * 2)
'''
[[ 2  4  6]
 [ 8 10 12]]
'''
