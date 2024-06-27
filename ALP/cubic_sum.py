a=int(input())

sum=0

for i in range(1,a+1,1):
    sum+=i**3
b=int((a*(a+1)/2)**2)
print(sum)
print(b)