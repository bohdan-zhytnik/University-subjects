import time
start_time = time.time()
n=int(input())
x=float(input())
results = {}
def vzorec (n,x):
    if n in results.keys():
        return results[n]

    if n ==0:
        output = -1
    elif n== 1:
        output = x
    elif n == 2:
        output = (-((x+1)/3))
    else:
        output = ((n/x)*(vzorec(n-1,x)))+((-1)**n)*((n+1)/(n-1))*vzorec(n-2,x)+((n-1)/(2*x))*vzorec(n-3,x)
    results[n] = output
    print(results)
    return output

# while :
#     kk = []
#     for k in range(n+1):
#         kk += [binomial(k, n)]
print(vzorec(n,x))
print("--- %s seconds ---" % (time.time() - start_time))
print(results)