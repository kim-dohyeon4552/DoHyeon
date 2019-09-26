#!/usr/bin/env python
# coding: utf-8

# # 1. 로또 추첨기

# In[53]:


import random
r = random.sample(range(1,46),6)    
print(sorted(r))


# # 2. 구구단
#     - 3단, 6단 제외한 구구단
#     - 반복문, 조건문사용
#     - 문자열 포맷팅을 활용해 2 X 2 = 4 형식으로 출력
#     

# In[58]:


count = 2
for j in range(0,8):
    if(count != 3 and count != 6):
        print()
        print('{0} 단'.format(count))
        for i in range(1,10):
            print('{0} X {1} = {2}'.format(count,i,count*i))
    count += 1
        


# # 3. 리스트 정렬
#     - a = [1,1,1,4,2,5,6,7,3,5,6,7,6,7,8,4,3,5,6,7,2,6,2,4,2,9,8,6,7,4,5,3,2]
#     - 중복제거 역순으로 정렬

# In[76]:


a = [1,1,1,4,2,5,6,7,3,5,6,7,6,7,8,4,3,5,6,7,2,6,2,4,2,9,8,6,7,4,5,3,2]
a = list(set(a))
a.reverse()
print(a)


# # 4. 동물 분류기
#     - 특징 5개 ex) 평균 크기,평균 무게,육식 여부(채식0,잡식1,육식2),사는 곳(물0,물땅1,땅2), 번식(알0,새끼1) 등)
#     - 분류할 동물은 최소 4마리 이상 (코끼리,상어,악어,돌고래,개)
#     - KNN 혹은 kmeans 알고리즘을 활용

# In[127]:


import numpy as np
from random import * 
import matplotlib.pyplot as plt
def distance(x,y):
    return np.sqrt(pow((x[0]-y[0]),2)+pow((x[1]-y[1]),2)+pow((x[2]-y[2]),2)+pow((x[3]-y[3]),2)+pow((x[4]-y[4]),2))
elephant = []
shark = []
alligator = []
dolphin = []
dog = []
salamander = []
animal = []
result=[]
lists = []
ok =[]
for i in range(50):
    elephant.append([uniform(3.2,4), uniform(4700, 6048),0,2,1,0])
    shark.append([uniform(3.4,4.9), uniform(522, 1110),2,0,0,1])
    alligator.append([uniform(3.5,6), uniform(200, 1000),2,1,0,2])
    dolphin.append([uniform(1.4,8), uniform(45, 6000),1,0,1,3])
    dog.append([uniform(0.1,2.1), uniform(2.75, 150),1,2,1,4])
p = elephant + shark + alligator + dolphin + dog
k = input("3,5,7,9,11 중에 넣으시요 제발")
for i in range(0,5):
    animal.append([uniform(0.1,8),uniform(2.75,6048),randint(0,2),randint(0,2),randint(0,1)])
for j in range(0,5):
    for i in range(0,250):
        result.append([distance(animal[j],p[i]),p[i][5]])
    lists.append(sorted(result))
for j in range(0,5):
    a,b,o,g,w = 0,0,0,0,0
    for i in range(0,int(k)):
        if (lists[j][i][1] == 0):
            a += 1
        elif (lists[j][i][1] == 1):
            b += 1
        elif (lists[j][i][1] == 2):
            o += 1
        elif (lists[j][i][1] == 3):
            g += 1
        elif (lists[j][i][1] == 4):
            w += 1
    ok.append([a,b,o,g,w])
print(ok)
for i in range(0,5):
    if (max(ok[i]) == ok[i][0]):
        print("elephant")
    elif(max(ok[i]) == ok[i][1]):
        print("shark")
    elif(max(ok[i]) == ok[i][2]):
        print("alligator")
    elif(max(ok[i]) == ok[i][3]):
        print("dolphin")
    elif(max(ok[i]) == ok[i][4]):
        print("dog")
    else:
        print("구분이 안되요")


# # 5. 수열 프로그램
#     - 100번까지 진행되는 피보나치 수열 프로그램

# In[120]:


r = [1,1]
for i in range(1,100): 
    r.append(r[-1+i]+r[0+i])
print(r)

