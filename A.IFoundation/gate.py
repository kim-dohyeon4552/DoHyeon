# 논리회로 And , OR , NANE

#  w1 * x1 + w2 * x2 + b > 0  흐른다. 
#  w1 * x1 + w2 * x2 + b <= 0  흐르지 않는다. 

import numpy as np


def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0 #흐르지 않는다.
    else:
        return 1 # 흐른다.
    
    
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0 #흐르지 않는다.
    else:
        return 1 # 흐른다.
    
def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0 #흐르지 않는다.
    else:
        return 1 # 흐른다.

print("AND")
for i in [(0,0),(1,0),(0,1),(1,1)]:
    y  = AND(i[0],i[1])
    print(str(i) + " -> " + str(y))
print()
print("OR")
for i in [(0,0),(1,0),(0,1),(1,1)]:
    y  = OR(i[0],i[1])
    print(str(i) + " -> " + str(y))
print()
print("NAND")
for i in [(0,0),(1,0),(0,1),(1,1)]:
    y  = NAND(i[0],i[1])
    print(str(i) + " -> " + str(y))
print()
print("XOR")
for i in [(0,0), (1, 0), (0,1),(1,1)]:
    s1 = NAND(i[0], i[1])
    s2 = OR(i[0], i[1])
    y = AND(s1, s2)
    
    print(str(i) + "->",  y)