#!/usr/bin/env python
# coding: utf-8

# In[152]:


# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
r = []
d = []
result =[]
man = []
man_average = []
weight_r, height_r,weight_d,height_d=0,0,0,0
a,m,k,cnt = 0,0,0,0
#r,b로 남자와여자 랜덤으로 50명
for i in range(50):
    r.append([random.randint(40,70), random.randint(140, 180)])
    d.append([random.randint(60,90), random.randint(160, 200)])
result = r+ d
#2명의 사람 변수
for i in range(0,2):
    man.append([random.randint(50,80),random.randint(150,180)])
# 초기값
plt.plot([man[0][0],man[1][0]],[man[0][1],man[1][1]])

#직선의 수직
a = (man[1][1] - man[0][1])/(man[1][0]-man[0][0]) # a = (y2-y1)/(x2-x1)
m = -1/a # m = -1/a
k = ((man[0][1]+man[1][1])/2 - m * (man[0][0]+man[1][0]) /2) # b = (y1+y2)/2 - a *(x1+x2)/2
plt.plot([40,100],[40*m+k,100*m+k])
while(cnt < 30):
    
    for i in range(0,100):
        if(result[i][1]-(m*result[i][0]) > k): # k > b
            plt.plot(result[i][0],result[i][1],'ro')
        elif(result[i][1]-(m*result[i][0]) < k): # k < b
            plt.plot(result[i][0],result[i][1],'bo')

    for i in range(0,50):
        weight_r += r[i][0]
        height_r += r[i][1]
        weight_d += d[i][0]
        height_d += d[i][1]
    
    man_average.append([weight_r/50,height_r/50])
    man_average.append([weight_d/50,height_d/50])
    plt.plot(man_average[0][0],man_average[0][1],'bx')
    plt.plot(man_average[1][0],man_average[1][1],'rx')
    plt.show()
    cnt += 1
    a = (man_average[1][1] - man_average[0][1])/(man_average[1][0]-man_average[0][0]) # a = (y2-y1)/(x2-x1)
    m = -1/a # m = -1/a
    k = ((man_average[0][1]+man_average[1][1])/2 - m * (man_average[0][0]+man_average[1][0]) /2)

