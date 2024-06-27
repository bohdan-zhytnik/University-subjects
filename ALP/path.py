import sys
b = []
f = open(sys.argv[1], 'r')
for line in f:
    b.append(list(map(int, line.split())))
# b.append(list(map(int, f[0].split())))
# print(b)
LEN_I=b[0][0]
LEN_J = b[0][1]
LEN_K=b[0][2]
a=b[1:]
# print(a)
mat1=[]
mat=[]
# print(a.pop(0))
for i in range(LEN_K):
    # mat1.clear()
    for ii in range(LEN_J):
        mat1.append(a.pop(0))
        # print(mat1)
    mat.append(mat1)
    mat1=[]
    # print('mat',mat)
print(mat)

# for i in range(m):
#     for ii in range(n):
#
#         if b[i][ii] == 2:
#             start.append(i)
#             start.append((ii))
#         if b[i][ii] == 4:
#             end.append(i)
#             end.append(ii)
#     if len(start) != 0 and len(end) != 0:
#         break
start=[]
end=[]
for i in range(LEN_K):
    for ii in range(LEN_J):
        for iii in range(LEN_I):
            if mat[i][ii][iii] == 2:
                start.append(iii)
                start.append(ii)
                start.append(i)
            if mat[i][ii][iii] == 4:
                end.append(iii)
                end.append(ii)
                end.append(i)
    if len(start)!=0 and len(end)!=0:
        break
print(start)
print(end)


def inside(s, r, m, matice):
    return 0 <= r < len(matice) and 0 <= s < len(matice[0]) and
