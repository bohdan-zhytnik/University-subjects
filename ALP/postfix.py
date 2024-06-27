import sys
# def PrtMat(mat):
#     for line in mat:
#         print(line)
mat=[]
f=open(sys.argv[1],'r')
for line in f:
    mat.append(list(map(str, line.split())))

m=len(mat)
end=(sys.argv[2])
c=0
# print(end)
n=len(end)
ks=0
sl=[]
# print(mat)
# for i in range(m-1,-1,-1):
#     print(mat[i])
#     print(mat[i][0][2])

for i in mat :
    # print(mat[i])
    # print(mat[i][1])
    # print(i[0])
    a=0
    b=0
    for q in range(1,n+1):
        a = 0
        b = 0
        # print(a,b)
        #
        # print()
        # print(q,-q)
        a=i[0][-q]
        b=end[-q]
        # print(a,b)
        if a==b:
            # print(a, b)
            c+=1
        else:
            a=0
            b=0
            c=0
            break
        # print(c)
        if c==n:
            # print(i[0])
            ks+=1

            c=0
            if len(sl)==0:
                sl=i[0]
                # print()
                # print(sl)
                # print()
            if len(sl) > len(i[0]):
            # print('.')
                sl = i[0]
            # print(sl)

# print()
print(ks)
if len(sl) !=0:
    sl=str(sl)
    print(sl)
else:
    print('None')


count=0
sl1=''
sl2=''

def Postfix(mat, end):
    for i in mat:
        a=i[0][-1]
        b=end[-1]
        if a == b:






#
#

