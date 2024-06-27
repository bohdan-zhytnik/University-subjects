

rad1 = list(map(float, input().split()))
rad2 = list(map(float, input().split()))
def function(x,y):
    f=(1/2)*(x*x)*((1-y)*(1-y))+((x-2)**3)-2*y+x
    return f
size1=len(rad1)
size2=len(rad2)
if size1!=size2:
    print('ERROR')
    quit()
VYSLFinal=0
less=0
index2=0
index=0
for i in range(size1):
    vysl = function(rad1[i],rad2[i])
    # print('f',vysl)
    if vysl>VYSLFinal:
        VYSLFinal=vysl
        index=i
    if vysl<0:
        # print('<0','  ',vysl)
        less=less+1
    function2=vysl*(rad1[i]+2)*(rad2[i]-2)
    # print('f2',function2)
    if  i == 0:
        VYSLFinal2=function2
    if function2<VYSLFinal2:
        VYSLFinal2=function2
        index2=i
# print(size1)
# print(size2)
print(index,less,index2)
# print(rad1[0],rad2[0])
# print(rad2[0]^2)
