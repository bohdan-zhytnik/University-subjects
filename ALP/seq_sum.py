nums = list(map(int, input().split()))
list1=nums.copy()
list2=[]
list3=[]
list4=[]
def prime(l):
    m=True
    p=2
    k=abs(l)
    while p<k:
        if k%p==0:
            break
        p+=1
    if p<k or k<=1:
       m=True
    else:
        m=False
    return m

b=len(list1)
a=0
for i in list1:
    c = len(list2)
    d = len(list3)
    f = len(list4)
    if b==1:
        l=list1[0]
        if prime(l)==True:
            list4=list1.copy()
    if a>=b-1:
        break
    if list1[a]>list1[a+1] and prime(list1[a])==True and prime(list1[a+1])==True :
        list2.append(list1[a])
        if a >=1:
            list2.remove(list1[a])
        list2.append(list1[a+1])
        a += 1
    else:
        if c>f:
            list4=list2.copy()
        elif d>f:
            list4=list3.copy()
        else:
            list4 = list4.copy()
        list3 = list2.copy()
        list2.clear()
        # l = list1[a+1]
        if prime(list1[a+1])==True:
            list2.append(list1[a+1])
        else:
            if prime(list1[a])==True:
                list2.append(list1[a])
        a += 1
    c = len(list2)
    d = len(list3)
    f = len(list4)
    if c>f:
        # if sum(list2) > sum(list4):
        list4 = list2.copy()
    elif c==f:
        if sum(list2) > sum(list4):
            list4 = list2.copy()
# print(list2)
# print(list3)
# print(list4)
delka=len(list4)
suma=sum(list4)
print(delka)
print(suma)