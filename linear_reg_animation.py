

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.linalg import inv


a = 3
b = 2
N = 100
# y = ax+b
x = np.reshape(range(N),(N,1))
y = a*x + b + 10*np.random.randn(N,1)

# Animation: MSE gradient descent
fig1 = plt.figure()

def init():
    line.set_data([], [])
    return line,

def update_w(i):
    global w
    off = 2*a*X.T.dot((X.dot(w)-y))
    w = w - off
    line.set_data(x,X.dot(w))
    print(i)
    return line,

X = np.hstack((np.ones((N,1)),x))
w = np.random.rand(X.shape[1],1)
ax = plt.axes(xlim=(-20, 120), ylim=(-50, 350))
line, = ax.plot([], [], lw=2)

a = 0.00000001

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Fitted model','Input'])
plt.title('Plot of $y = 3x+2 + 10*\eta (0,1)$')

# for i in range(98):
#     off = 2*a*X.T.dot((X.dot(w)-y))
#     if np.sum(np.square(off)) < .000001:
#         break
#     w = w - off
#     print(off)
# print(w)
# line, = plt.plot(x, X.dot(w))

line_ani = animation.FuncAnimation(fig1, update_w,init_func=init, frames=100, interval=25, blit=True)
plt.show()