"""NUMPY

Creating numpy array
"""

import numpy as np
a=np.array([1,2,4]) #1D array
print(type(a))
print(a)

import numpy as np
a=np.array([[1,2,3],[1,2,3]]) #2D array using list os lists
print(a,"\n")

b=np.array([(1,2,3),(4,5,6)]) #2D array using list if tuples
print(b,"\n")

c=np.array([range(i,i+3) for i in [2,3,5]])  #2D array using list comprehension
print(c,"\n")

d=np.array([[i for i in range(3)] for j in range(4)]) #2D array using list comprehension
print(d)

"""Various ways to create numpy array"""

a=np.zeros((3,3),dtype=int)  #numpy array consisting of 0s
print(a,"\n")

b=np.ones(10,dtype=int)      #numpy array consisting of 1s
print(b,"\n")

c=np.full((3,5),3)           #numpy array consisting of specified number
print(c,"\n")

d=np.arange(1,10,2)          # 1D numpy array consisting of numbers
                                #specified in the range(start,end,steps) without including the end
print(d,"\n")

e=np.linspace(0, 1, 5)       #gives equally sized array between given range
print(e,"\n")

f=np.random.random((2,2))       #numpy array consisting of numbers between 0 and 1 
print(f,"\n")

g=np.random.randint(1,2,(3,3)) #gives numpy array of random numbers bw two given numbers
print(g,"\n")

h=np.eye(5)                    #identity matrix of given degree
print(h,"\n")

i=np.empty((2,2,2))            #numpy array of garbage values of given degree
print(i)

"""Numpy attributes"""

import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(6,10,5)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
#print(x1)
#print(x2)
print(x3)
print("x3 dim:",x3.ndim) #ndim gives the number of dimensions(axes) of the ndarray
print("x3 shape:",x3.shape) #shape is a tuple of integers representing the size of the ndarray in each dimension
print("x3 size:",x3.size) #size gives the total number of elements in the ndarray
print("x3 dtype:",x3.dtype) #dtype tells the data type of the elements of a NumPy array
print("itemsize:", x3.itemsize, "bytes") #itemsize returns the size (in bytes) of each element of a NumPy array.
print("nbytes:", x3.nbytes, "bytes") #nbytes returns the total size of a NumPy array.

"""Reshaping of np arrays"""

import numpy as np
a=np.random.randint(1,10,(3,4))
print("before reshaping: \n",a)
print("\n")
print("after reshaping: \n",a.reshape(2,6))

import numpy as np
a=np.random.randint(1,10,(2,3,4))
print("before using ravel function:\n",a)
print("\n")
print("After using ravel function:\n",a.ravel())  #this will convert all the rows in one row

# addition, subtraction, multiplication, division of arrays
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[1,2,3],[4,5,6]])

print(a+b)
print("")
print(a-b)
print(end='')
print(a*b) #this is not matrix multiplication
print('end')
print(a/b)

"""Transpose of an array"""

import numpy as np
a=np.array([[1,2,3],[1,2,3]])
print("a:",a,"\n")
print("Transpose of a is \n",a.T)

# max, min and sum

a=np.array([1,2,3,4,5])
print(a.max())
print(a.min())
print(a.sum())

# rows are called axis=0, and cols are called axis=1
import numpy as np
a=np.array([[1,2,3],[4,5,6]])
print(a.max(axis=0)) #this will give the max of each col

print(a.max(axis=1)) #this will give the max of each row

print(a.min(axis=0)) #this will give the min of each col

print(a.min(axis=1)) #this will give the min of each row

print(a.sum(axis=0)) #this will give the sum of each col

print(a.sum(axis=1)) #this will give the sum of each row

# square root and standard deviation in numpy

# standar deviation is how much each element deviates/varies from the mean or something like that

import numpy as np

a=np.array([4,9,16,25,36,49])
print(np.sqrt(a)) #will print the square root of each element

a=np.array([1,2,3,4,5])
print(np.std(a))  #will print the standard deviation

#concatenation in arrays
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[7,8,9],[10,11,12]])
c=np.array([[13,14,15],[16,17,18]])

print(np.vstack((a,b,c)))  #vertical stack
print('')
print(np.hstack((a,b,c)))  #horizontal stack

"""Numpy occupies less memory than list"""

import numpy as np
import time
import sys
a=[i for i in range(10)]
print(type(a))
print(a)
print(sys.getsizeof(1)*len(a))
b=np.arange(10)
print(type(b))
print(b)
print(b.itemsize*b.size)

"""Numpy is faster than lists"""

import numpy as np
import time
size=10000000
l1=range(size)
l2=range(size)
a1=np.arange(size)
a2=np.arange(size)
start=time.time()
result=[(x+y) for x,y in zip(l1,l2)]
#print(result)
print((time.time()-start))
start=time.time()
result=a1+a2
print((time.time()-start))
