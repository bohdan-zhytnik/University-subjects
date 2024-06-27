import sys

b = []
f = open(sys.argv[1], 'r')
for line in f:
    b.extend(list(map(str, line.split())))
slovo = ''
slovoContl = ''
word = ''
for word1 in range(len(b[0])):
    prov = True
    word = b[0][word1]
    for line in b:
        if word in line:
            continue
        else:
            prov = True
            break
    if prov is True:
        slovo = word
        for i in b[0][word1 + 1:]:
            prov1 = 1
            word += i
            for line in b:
                if word in line:
                    continue
                else:
                    word = ''
                    prov1 = 0
                    break
            if prov1 == 1:
                slovo = word
            else:
                break
        if len(slovo) > len(slovoContl):
            slovoContl = slovo
            slovo = ''
    else:
        slovo = ''
        word = ''
        continue
if len(slovoContl) == 0:
    print('NEEXISTUJE')
else:
    print(slovoContl)





