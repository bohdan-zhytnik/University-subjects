import sys
def PrtMat(mat):
    for line in mat:
        print(line)
mat=[]
f=open(sys.argv[1],'r')
for line in f:
    mat.append(list(map(str, line.split())))
# PrtMat(mat)
ST=0
# for i in mat:
#     print(i)
p=0
radProv=0
radProv1=0
l=-1
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
CX=0
CO=0
m=len(mat)
n=len(mat[0])
def VkolX(mat):
    CountX=0
    CountO = 0
    m = len(mat)
    n = len(mat[0])
    FM=[]
    kolO = 0
    kolX = 0
    for q in range(m):
        FM=mat[q]
        CountX1=FM.count('x')
        CountX=CountX1+CountX
        CountX1=0
        CountO=FM.count('o')+CountO
    return CountX,CountO

def PPV (mat,rad,i,Prov):              #исследую вправо вниз
    coordR=0
    coordS=0
    count=0

    Lvp1 = 0
    Lvp2 = 0
    m = len(mat)
    n = len(mat[0])
    if i + 1 < n:
        p = i + 1
    else:
        p = 0
    if rad + 1 < m:
        radProv = rad + 1
    else:
        radProv = 0

    if (mat[rad][i] == Prov) or((mat[rad][i]=='.')):
        Lvp1 += 1
        if mat[rad][i]=='.':
            count=1
            coordR = rad
            coordS = i

        # print(radProv)
        # print(rad,i)
        # print(radProv,p)

        # print(p)
        # print(Lvp1)
        if p != 0 and radProv != 0:
            for vp in range(m - rad):
                # print(radProv,p)
                # print(m,n)
                if ((mat[radProv][p]) == Prov) or (((mat[radProv][p]) == '.') and count == 0) :
                    # print(radProv)
                    # print(p)
                    if (mat[radProv][p]) == '.':
                        count=1
                        coordR = radProv
                        coordS = p
                    Lvp1 += 1
                    radProv += 1
                    p += 1
                    # print(Lvp1)
                # if (mat[radProv][p]) == '.':

                else:
                    break
                if p == n or radProv == m:
                    break
        if Lvp1 > Lvp2:
            Lvp2 = Lvp1

            # print(Lvp1,'\n')
            Lvp1=0
        else:
            Lvp1=0
    return Lvp2,coordR,coordS
def PLV(mat,rad,i,Prov):            #исследую вниз слева
    coordR=0
    coordS=0
    count=0
    Lvl1=0
    Lvl2=0
    m = len(mat)
    n = len(mat[0])
    if rad+1<m:
        radProv1 = rad + 1
    else:
        radProv1 = 0
    if i>=1:
        l=i-1
    else:
        l=-1
    if (mat[rad][i]==Prov) or((mat[rad][i]=='.')):
        Lvl1+=1
        if mat[rad][i]=='.':
            count=1
            coordR = rad
            coordS = i
        # print(radProv1)
        # print(rad,i)
        # print(l)
        # print(Lvl1)
        # print()
        if l >=0 and radProv1 !=0 :
            for vp in range(m-rad):
                # print(radProv1,l)
                if ((mat[radProv1][l])==Prov) or (((mat[radProv][p]) == '.') and count == 0):
                    if (mat[radProv][p]) == '.':
                        count=1
                        coordR = radProv
                        coordS = p
                    # print(radProv1)
                    # print(l)

                    Lvl1+=1
                    # print(Lvl1)
                    radProv1+=1
                    l=l-1
                else:
                    break
                if l<0 or radProv1==m:
                    break
        if Lvl1>Lvl2:
            Lvl2=Lvl1
            # print(Lvp1,'\n')
            Lvl1=0
        else:
            Lvl1=0
        # print(Lvl2)
        # print()
    return Lvl2,coordR,coordS


def PPP(mat, rad, i, Prov):  # исследую вправо
    coordR = 0
    pp=0
    count=0
    coordS = 0
    Lpp1 = 0
    Lpp2 = 0
    m = len(mat)
    n = len(mat[0])
    # if rad+1<m:
    #     radProv1 = rad + 1
    # else:
    #     radProv1 = 0
    if i + 1 < n:
        pp = i + 1
    else:
        p = 0
    if (mat[rad][i] == Prov) or ((mat[rad][i] == '.')):
        Lpp1 += 1
        if mat[rad][i] == '.':
            # print('WoW')
            # print(rad,i)
            count = 1
            coordR = rad
            coordS = i
        # print(radProv1)
        # print(rad,i)
        # print(l)
        # print(Lvl1)
        # print()
        if pp !=0:
            for vp in range(n - i - 1):
                # print(radProv1,l)
                if ((mat[rad][pp]) == Prov) or (((mat[rad][pp]) == '.') and count == 0):
                    if (mat[rad][pp]) == '.':
                        # print('WoW')
                        # print(Lpp1)
                        # print(vp)
                        # print(rad,pp,i)
                        count = 1
                        coordR = rad
                        coordS = pp
                    # print(radProv1)
                    # print(l)

                    Lpp1 += 1
                    # print(Lvl1)

                    pp += 1
                else:
                    break
                if pp == n:
                    break
        if Lpp1 > Lpp2:
            Lpp2 = Lpp1
            # print(Lvp1,'\n')
            Lvl1 = 0
        else:
            Lvl1 = 0
        # print(Lvl2)
        # print()
    return Lpp2, coordR, coordS


def PVV(mat, rad, i, Prov):  # исследую вправо вниз
    coordR = 0
    coordS = 0
    count = 0

    Lvv1 = 0
    Lvv2 = 0
    m = len(mat)
    n = len(mat[0])
    # if i + 1 < n:
    #     p = i + 1
    # else:
    #     p = 0
    if rad + 1 < m:
        radProv = rad + 1
    else:
        radProv = 0

    if (mat[rad][i] == Prov) or ((mat[rad][i] == '.')):
        Lvv1 += 1
        if mat[rad][i] == '.':
            count = 1
            coordR = rad
            coordS = i

        # print(radProv)
        # print(rad,i)
        # print(radProv,p)

        # print(p)
        # print(Lvp1)
        if radProv != 0:
            for vp in range(m - rad):
                # print(radProv,p)
                # print(m,n)
                if ((mat[radProv][i]) == Prov) or (((mat[radProv][i]) == '.') and count == 0):
                    # print(radProv)
                    # print(p)
                    if (mat[radProv][i]) == '.':
                        # print('WoW')
                        # print(Lvv1)
                        # print(vp)
                        # print(radProv,i)
                        count = 1
                        coordR = radProv
                        coordS = i
                    Lvv1 += 1
                    radProv += 1
                    # p += 1
                    # print(Lvp1)
                # if (mat[radProv][p]) == '.':

                else:
                    break
                if radProv == m:
                    break
        if Lvv1 > Lvv2:
            Lvv2 = Lvv1

            # print(Lvp1,'\n')
            Lvv1 = 0
        else:
            Lvv1 = 0
    return Lvv2, coordR, coordS

CX,CO=VkolX(mat)
if CO>=CX:
    ST=0
if CO==CX+1:
    ST=1
P1=0
P2=0
P3=0
P4=0

# if CX>CO:
#     ST=1


m = len(mat)
n = len(mat[0])
Prov='0000'
# print(CX,CO,ST)
if ST==1:
    Prov='x'
else:
    Prov='o'
# print(Prov)                                             #!!!!!!!!!!!!!!!!!!!!!!!!

for rad in range(m):
    # print(Prov)
    if P1==5 or P2==5 or P3==5 or P4==5:
    # if P2 == 5 :
        break
    for i in range(n):
        P2,coordR,coordS = PPV(mat, rad, i, Prov)
        if P2==5:
            # print('WoW')

            print(coordR,coordS)
            # print(P2)
            # print()
            break
        # print(P2)
        P1,coordR,coordS = PLV(mat, rad, i, Prov)
        if P1==5:
            # print('WoW')

            print(coordR,coordS)
            # print(P2)
            # print()
            break
        P3,coordR,coordS = PPP(mat, rad, i, Prov)
        # print(P3)
        # print(coordR, coordS)
        if P3==5:
            # print('WoW')

            print(coordR,coordS)
            # print(P3)
            # print()
            break
        P4,coordR,coordS = PVV(mat, rad, i, Prov)
        if P4==5:
            # print('WoW')

            print(coordR,coordS)
            # print(P4)
            # print()
            break
        # print(P1)

# print()
# print()
# for rad in range(m):
#     if P1==5:
#         break
#     for i in range(n):
#         P1,coordR,coordS = PLV(mat, rad, i, Prov)
#         if P1==5:
#             # print('WoW')
#
#             print(coordR,coordS)
#             # print(P2)
#             # print()
#             break
#         # print(P1)
#
# for rad in range(m):
#     if P3==5:
#         break
#     for i in range(n):
#         P3,coordR,coordS = PPP(mat, rad, i, Prov)
#         if P3==5:
#             # print('WoW')
#
#             print(coordR,coordS)
#             # print(P3)
#             # print()
#             break
#         # print(P2)
# print()
# print()
#
#
# for rad in range(m):
#     if P4==5:
#         break
#     for i in range(n):
#         P4,coordR,coordS = PVV(mat, rad, i, Prov)
#         if P4==5:
#             # print('WoW')
#
#             print(coordR,coordS)
#             # print(P4)
#             # print()
#             break

        # print(P2)
# print()
# print(P1,P2,P3,P4)
if (P1!=5) and (P2!=5) and (P3!=5) and (P4!=5):
    ST=(ST+1)%2
    if ST == 1:
        Prov = 'x'
    else:
        Prov = 'o'
    # for rad in range(m):
    #     if P1 == 5:
    #         break
    #     for i in range(n):
    #         P2, coordR, coordS = PPV(mat, rad, i, Prov)
    #         if P2 == 5:
    #             # print('WoW')
    #
    #             print(coordR, coordS)
    #             # print(P2)
    #             # print()
    #             break
    #         # print(P2)
    # # print()
    # # print()
    # for rad in range(m):
    #     if P2 == 5:
    #         break
    #     for i in range(n):
    #         P1, coordR, coordS = PLV(mat, rad, i, Prov)
    #         if P1 == 5:
    #             # print('WoW')
    #
    #             print(coordR, coordS)
    #             # print(P2)
    #             # print()
    #             break
    #         # print(P1)
    #
    # for rad in range(m):
    #     if P3 == 5:
    #         break
    #     for i in range(n):
    #         P3, coordR, coordS = PPP(mat, rad, i, Prov)
    #         if P3 == 5:
    #             # print('WoW')
    #
    #             print(coordR, coordS)
    #             # print(P3)
    #             # print()
    #             break
    #         # print(P2)
    # # print()
    # # print()
    #
    # for rad in range(m):
    #     if P4 == 5:
    #         break
    #     for i in range(n):
    #         P4, coordR, coordS = PVV(mat, rad, i, Prov)
    #         if P4 == 5:
    #             # print('WoW')
    #
    #             print(coordR, coordS)
    #             # print(P4)
    #             # print()
    #             break
    for rad in range(m):
        if P1 == 5 or P2 == 5 or P3 == 5 or P4 == 5:
            break
        for i in range(n):
            P2, coordR, coordS = PPV(mat, rad, i, Prov)
            if P2 == 5:
                # print('WoW')

                print(coordR, coordS)
                # print(P2)
                # print()
                break
            # print(P2)
            P1, coordR, coordS = PLV(mat, rad, i, Prov)
            if P1 == 5:
                # print('WoW')

                print(coordR, coordS)
                # print(P2)
                # print()
                break
            P3, coordR, coordS = PPP(mat, rad, i, Prov)
            if P3 == 5:
                # print('WoW')

                print(coordR, coordS)
                # print(P3)
                # print()
                break
            P4, coordR, coordS = PVV(mat, rad, i, Prov)
            if P4 == 5:
                # print('WoW')

                print(coordR, coordS)
                # print(P4)
                # print()
                break
            # print(P2)
    # print()
    # print()
# print(P2)
# print(Prov)