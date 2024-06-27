import sys
def PrtMat(mat):
    for line in mat:
        print(line)
mat=[]
f=open(sys.argv[1],'r')
for line in f:
    mat.append(list(map(int, line.split())))
# print(mat)
# PrtMat(mat)
p=0
radProv=0
radProv1=0
l=-1

m=len(mat)
n=len(mat[0])
Lvp1=[]
Lvp2=[]
vpIr=0
vpIs=0
Lvl1=[]
Lvl2=[]
vlIr=0
vlIs=0
Vysl=[]
VIr=0
VIs=0
# print(n)
for rad in range(m):
    for i in range(n):
        # radProv= rad+1
        if i+1<n:
            p=i+1
        else:
            p=0
        if rad+1<m:
            radProv = rad + 1
        else:
            radProv=0
        # if i>0:
        #     l=i-1
        if mat[rad][i]%2==0:
            Lvp1.append(mat[rad][i])

            # print(radProv)
            # print(rad,i)
            # print(radProv,p)

            # print(p)
            # print(Lvp1)
            if p !=0 and radProv !=0 :
                for vp in range(m-rad):
                    # print(radProv,p)
                    # print(m,n)
                    if (mat[radProv][p])%2==0:
                        # print(radProv)
                        # print(p)

                        Lvp1.append(mat[radProv][p])
                        radProv+=1
                        p+=1
                        # print(Lvp1)
                    else:
                        break
                    if p==n or radProv==m:
                        break
            if len(Lvp1)>len(Lvp2):
                Lvp2=Lvp1.copy()
                vpIr=rad
                vpIs=i
                # print(Lvp1,'\n')
                Lvp1.clear()
            else:
                Lvp1.clear()
        # print()


    for i in range(n):
        # radProv= rad+1
        if rad+1<m:
            radProv1 = rad + 1
        else:
            radProv1 = 0
        if i>=1:
            l=i-1
        else:
            l=-1
        if mat[rad][i]%2==0:
            Lvl1.append(mat[rad][i])
            # print(radProv1)
            # print(rad,i)
            # print(l)
            # print(Lvl1)
            # print()
            if l >=0 and radProv1 !=0 :
                for vp in range(m-rad):
                    # print(radProv1,l)
                    if (mat[radProv1][l])%2==0:
                        # print(radProv1)
                        # print(l)

                        Lvl1.append(mat[radProv1][l])
                        # print(Lvl1)
                        radProv1+=1
                        l=l-1
                    else:
                        break
                    if l<0 or radProv1==m:
                        break
            if len(Lvl1)>len(Lvl2):
                Lvl2=Lvl1.copy()
                vlIr=rad
                vlIs=i
                # print(Lvp1,'\n')
                Lvl1.clear()
            else:
                Lvl1.clear()
            # print(Lvl2)
            # print()
if len(Lvp2) > len(Lvl2):
    Vysl=Lvp2.copy()
    VIr=vpIr
    VIs=vpIs
else:
    Vysl=Lvl2.copy()
    VIr=vlIr
    VIs=vlIs

# print(Vysl)
# print(len(Vysl))
# print(VIr)
# print(VIs)
print(VIr,VIs,len(Vysl))
# print()
# print(Lvp2)
# print(vpIr)
# print(vpIs)
# print(len(Lvp2))
# print(Lvl2)
# print(vlIr)
# print(vlIs)
# print(len(Lvl2))
#




