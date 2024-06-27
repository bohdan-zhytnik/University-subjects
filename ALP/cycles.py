import sys

def PrtMat(mat):
    for line in mat:
        print(line)

def FindNext(mat,i,q,bukva):
    samekolor = 0
    a=mat[i][q]
    c=[]
    b=mat[i][q][bukva]  #буква это индекс(позиция) линии откуда ищем такойже цвет в этом квадрате
    x = bukva
    if b in a[:x]:
        c = a[:x]
        samekolor = c.index(b)
    else:
        c = a[:x] + 'o' + a[x + 1:]
        samekolor = c.index(b)
    return samekolor

def Perechod (next1,rad,sl,prov,n,m,b):
    radok=0
    prov1=1
    sloup=0
    next2=(next1+2)%4
    kontrl=b[rad][sl][next1]
    if next1==0:
        radok=rad
        sloup=sl-1
        if sloup<0:
            prov1=0
    elif next1==1:
        radok=rad-1
        if radok<0:
            # print('No')
            prov1=0
        sloup=sl
    elif next1==2:
        radok=rad
        sloup=sl+1
        if sloup>n-1:
            # print()
            # print(sloup, n)
            sloup=sloup-1
            prov1=0
    elif next1==3:
        radok=rad+1
        if radok>m-1:
            radok=radok-1
            prov1=0
        sloup=sl
    kontrl1=b[radok][sloup][next2]

    if b[radok][sloup]=='none':
        prov1 = 0
    # print(type(next2), type(radok), type(sloup), type(prov))
    if kontrl==kontrl1:
        return next2,radok,sloup,prov1
    else:
        prov1 = 0
        return next2,radok,sloup,prov1


b = []
f = open(sys.argv[1],"r")
for line in f:
    b.append( line.strip().split() )
f.close()
skok=0
Mega=[]
MegaProv=[]
LengthD=0
AmountD=0
LengthL=0
AmountL=0
LengthDFinal=0
LengthLFinal=0

# PrtMat(b)
m=len(b)
n=len(b[0])
for i in range(m-1):
    for q in range(n-1):
        prov=1
        # print(True)
        if b[i][q]=='none':
            continue
        start=b[i][q][2]
        rad=i

        if q<n-1:
            sl=q+1
        else:
            break
        if b[rad][sl][0]==start and start == b[i][q][3] :
            Mega=[i,q,2]
            if Mega in MegaProv:
                # print()
                continue
            # print(i,q)
            MegaProv.append([i,q,3])
            next1=FindNext(b,i,sl,0)
            MegaProv.append([rad, sl,next1])
            if b[i][q][3] =='d':
                LengthD+=1
            else:
                LengthL+=1
            while prov == 1:
                # print('DO')
                # print(rad,sl)
                next2,rad,sl,prov=Perechod(next1,rad,sl,prov,n,m,b)
                if b[i][q][3] == 'd':
                    LengthD += 1
                else:
                    LengthL += 1
                MegaProv.append([rad, sl,next2])
                # print()
                # print(rad,sl,next2)
                # print()
                # print(prov)
                # print(rad)
                if prov==0:
                    if b[i][q][3] == 'd':
                        LengthD =0
                    else:
                        LengthL =0
                    # print("wow")
                    break



                next1=FindNext(b,rad,sl,next2)
                MegaProv.append([rad, sl,next1])
                if (rad == i) and (sl ==q):
                    skok+=1
                    # print(i,q)
                    if  b[i][q][3] =='d':
                        # print(i,q,'d')
                        AmountD+=1
                        if LengthDFinal<LengthD:
                            LengthDFinal=LengthD
                            # print('d',LengthD)
                            LengthD=0
                        else:
                            LengthD=0
                    elif  b[i][q][3] =='l':
                        # print(i, q, 'l')
                        AmountL+=1
                        if LengthLFinal<LengthL:
                            LengthLFinal=LengthL
                            LengthL=0
                        else:
                            LengthL=0
                    # print(1)
                    break
# print(skok)
print(AmountL,AmountD,LengthLFinal,LengthDFinal)
# print(AmountD)
# print(LengthLFinal)
# print(LengthDFinal)
# print(MegaProv)


