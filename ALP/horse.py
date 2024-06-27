import sys
import time
b = []
f = open(sys.argv[1], 'r')
for line in f:
    b.append(list(map(int, line.split())))
m = len(b)
n = len(b[0])
start = []
end = []
for i in range(m):
    for ii in range(n):

        if b[i][ii] == 2:
            start.append(i)
            start.append((ii))
        if b[i][ii] == 4:
            end.append(i)
            end.append(ii)
    if len(start) != 0 and len(end) != 0:
        break
def inside(r, s, matice):
        return 0 <= r < len(matice) and 0 <= s < len(matice[0])

def move(r,s,b):
    c=[]
    dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8 = [],[],[],[],[],[],[],[]
    if inside(r-2,s-1,b) and b[r-2][s-1]!=1:
        dir1 = [r-2, s-1]
        c+=[dir1]
    if inside(r - 2, s + 1, b) and b[r - 2][s + 1] != 1:
        dir2=[r - 2, s + 1]
        c+=[dir2]
    if inside(r - 1, s + 2, b) and b[r - 1][s + 2] != 1:
        dir3=[r - 1, s + 2]
        c+=[dir3]
    if inside(r + 1, s + 2, b) and b[r + 1][s + 2] != 1:
        dir4=[r + 1, s + 2]
        c+=[dir4]
    if inside(r + 2, s + 1, b) and b[r + 2][s + 1] != 1:
        dir5=[r + 2, s + 1]
        c+=[dir5]
    if inside(r + 2, s - 1, b) and b[r + 2][s - 1] != 1:
        dir6=[r + 2, s - 1]
        c+=[dir6]
    if inside(r + 1, s - 2, b) and b[r + 1][s - 2] != 1:
        dir7=[r + 1, s - 2]
        c+=[dir7]
    if inside(r - 1, s - 2, b) and b[r - 1][s - 2] != 1:
        dir8=[r - 1, s - 2]
        c+=[dir8]

    return c
def BFS():
    queue = [(start,[start])]
    Path=[]
    Path+=start[:]
    visited=[]
    sequence={}
    while queue:
        actual,path=queue.pop(0)
        visited.append(actual)
        for next in move(actual[0],actual[1],b):
            if next == end:
                path.append(next)
                return path
            if next not in path and next not in visited:
                path1=path.copy()
                path1.append(next)
                queue.append((next,path1))
                visited.append(next)
    if len(queue)==0:
        print('NEEXISTUJE')
        quit()
path=BFS()
sequence=[]
seq=[]
sequence.extend(path[1:])
for ans in sequence:
    for answ in ans:
        seq.append(answ)
for i in range(len(seq)-1):
    print(seq[i],end=' ')
print(seq[-1])