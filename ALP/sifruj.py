import sys
a=[]
# data='ahoj, jak se mate?'
data=str(input())
for char in data:
    a.append(char)
Len=(len(a)%4)
LEN=Len
# print(a)
# print(Len)
aa=[]
# print(a)
if LEN !=0:
    while Len >0:
        aa.append(a.pop(-Len))
        Len-=1
# print(aa)
lenAA=len(aa)
AA=[]
if LEN!=0:
    for i in range(lenAA):
        AA.append(ord(aa[i]))
    for i in range(4-len(AA)):
        AA.append(0)
# print(AA)
# print('AA',(( (AA[0]*256 + AA[1])*256 + AA[2] )*256 + AA[3]))
# k=(( (AA[0]*256 + AA[1])*256 + AA[2] )*256 + AA[3])
# print('k',k)
# print(a)


# for  i in range(4-len(a)%4):
#     a.append(' ')
n=len(a)
b=[]
for i in range(0,len(a),4):
    # print(((ord(a[i]) * 256 + ord(a[i+1])) * 256 + ord(a[i+2])) * 256 + ord(a[i+3]))
    b.append(((ord(a[i]) * 256 + ord(a[i+1])) * 256 + ord(a[i+2])) * 256 + ord(a[i+3]))
if LEN!=0:
    b.append(( (AA[0]*256 + AA[1])*256 + AA[2] )*256 + AA[3])
# print(b)
def FastExp(a, k,n):
    if k == 0: return 1
    if k == 1: return a
    if k % 2 == 0:
        i = (FastExp(a, k / 2,n))
        return (i * i)%n
    else:
        return (a * FastExp(a, k - 1,n))%n
k=int(sys.argv[2])
n=int(sys.argv[1])
for i in range(len(b)-1):
    print((FastExp(b[i],k,n)),end=' ')
print(FastExp(b[-1],k,n))
# print(ord('.'))