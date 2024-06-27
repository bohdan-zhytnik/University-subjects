numberOfLines = int(input())
lc=[]
def mindel (a,b):
    t=0
    while b!=0:
        t = b
        b=a%b
        a=t
    return a

for i in range(numberOfLines):
    nums = list(map(int, input().split()))
    a = mindel(nums[0],nums[1])
    lc+=chr(a)

print(*lc,sep='')
