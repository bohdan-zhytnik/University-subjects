import draw as Drawer
import sys
import random
import copy
import base as Base

# implement you player here. If you need to define some classes, do it also here. Only this file is to be submitted to Brute.
# define all functions here

class Player(Base.BasePlayer):
    def __init__(self, board, name, color):
        Base.BasePlayer.__init__(self,board, name, color)
        self.algorithmName = "MarkLox"
        self.BigControleForMove=1


    def __str__(self):
        return f"{self.board}({self.myColor})"



    def AnalysisTheBoard(self):
        OccupiedSquare=[]
        for rad in range(len(self.board)):      #analysis board
            for sl in range(len(self.board[rad])):
                if self.board[rad][sl]=="none":
                    continue
                else:
                    OccupiedSquare.append([rad,sl])
        return OccupiedSquare   #list of occupied square

    def FindNext(self, rad, sl, BukvaIndex):     # look up the same color in the squere
        samekolor = 0
        a = self.board[rad][sl]
        c = []
        b = self.board[rad][sl][BukvaIndex]
        x = BukvaIndex
        if b in a[:x]:
            c = a[:x]
            samekolor = c.index(b)
        else:
            c = a[:x] + 'o' + a[x + 1:]
            samekolor = c.index(b)
        return samekolor

    def AnalysisSquare(self,OneOfOccupiedSquare):            #Nachozdenie indexov linij mojeho cveta
        LinePosition={}                                 #writes the indices of my color under the square coordinate key, koordinaty ne v list a v (rad,sl)
        # for i in range(len(OneOfOccupiedSquare)):
        rad,sl=OneOfOccupiedSquare
        ColorPosition1=self.board[rad][sl].index(self.myColor)

        ColorPosition2=self.FindNext(rad,sl,ColorPosition1)
        LinePosition[rad,sl]=[ColorPosition1,ColorPosition2]
        return LinePosition #rabotajet po proverke kvadrata

    def SuperPerechodThroughOneSquare(self, ColorPosition, rad, sl):
        m = len(self.board)
        n = len(self.board[0])
        AnotherColor = ''
        if self.myColor == 'l':
            AnotherColor = 'd'
        else:
            AnotherColor = 'l'
        PermissionToUseSquare = 1
        # VynucenyTah=0
        GoingOverTheBoard = 0
        sloup = 0
        ColorPositionNeighbor = (ColorPosition + 2) % 4
        # kontrl = b[rad][sl][ColorPosition]
        if ColorPosition == 0:
            radok = rad
            sloup = sl - 1
            if sloup < 0:
                PermissionToUseSquare = 1
                GoingOverTheBoard = 1
        elif ColorPosition == 1:
            radok = rad - 1
            if radok < 0:
                # print('No')
                PermissionToUseSquare = 1
                GoingOverTheBoard = 1
            sloup = sl
        elif ColorPosition == 2:
            radok = rad
            sloup = sl + 1
            if sloup > n - 1:
                # print()
                # print(sloup, n)
                sloup = sloup
                PermissionToUseSquare = 1
                GoingOverTheBoard = 1
        else:
            radok = rad + 1
            if radok > m - 1:
                radok = radok
                PermissionToUseSquare = 1
                GoingOverTheBoard = 1
            sloup = sl
        # print('GOTB',GoingOverTheBoard)
        if GoingOverTheBoard == 0:
            kontrl1 = self.board[radok][sloup][ColorPositionNeighbor]
        # if self.myColor==kontrl1:
        # VynucenyTah=1
        if GoingOverTheBoard == 1:
            # VynucenyTah=0
            return PermissionToUseSquare
        if GoingOverTheBoard == 0:
            if self.board[radok][sloup] == 'none':
                PermissionToUseSquare = 1

                return PermissionToUseSquare
            elif AnotherColor == kontrl1:
                PermissionToUseSquare = 0
                return PermissionToUseSquare

    def Perechod(self, ColorPosition, rad, sl):
        m = len(self.board)
        n = len(self.board[0])
        PossibleDrawingPoint = [0, 1, 2, 3]
        SuperPossibleDrawingPoint = []
        PermissionToUseSquare = 1
        sloup = 0
        ColorPositionNeighbor = (ColorPosition + 2) % 4
        kontrl = self.board[rad][sl][ColorPosition]
        if ColorPosition == 0:
            radok = rad
            sloup = sl - 1
            if sloup < 0:
                PermissionToUseSquare = 0
                # print(rad,sl,'dou')
        elif ColorPosition == 1:
            radok = rad - 1
            if radok < 0:
                # print('No')
                PermissionToUseSquare = 0
            sloup = sl
        elif ColorPosition == 2:
            radok = rad
            sloup = sl + 1
            if sloup > n - 1:
                # print()
                # print(sloup, n)
                sloup = sloup
                PermissionToUseSquare = 0
        else:
            radok = rad + 1
            if radok > m - 1:
                radok = radok
                PermissionToUseSquare = 0
            sloup = sl
        if PermissionToUseSquare != 0:
            kontrl1 = self.board[radok][sloup][ColorPositionNeighbor]

            if self.board[radok][sloup] == 'none':
                # PermissionToUseSquare = 1
                PossibleDrawingPoint.remove(ColorPositionNeighbor)
                for i in PossibleDrawingPoint:
                    SuperPermissionToUseSquare = self.SuperPerechodThroughOneSquare(i, radok, sloup)
                    if SuperPermissionToUseSquare == 1:
                        SuperPossibleDrawingPoint.append(i)
                    else:
                        continue
                if len(SuperPossibleDrawingPoint) == 0:
                    PermissionToUseSquare = 0
                return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint  # I want to make dictionary[from which point I can driwe a line]=to whith point I can drive a line
            if kontrl == kontrl1:
                PermissionToUseSquare = 0
                return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint
        elif PermissionToUseSquare == 0:
            return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint

    def SuperVynucenyTah(self, ColorPosition, rad, sl):
        m = len(self.board)
        n = len(self.board[0])
        AnotherColor = ''
        # if self.myColor == 'l':
        #     AnotherColor='d'
        # else:
        #     AnotherColor='l'
        # AnotherColor=self.myColor
        PermissionToUseSquare = 1
        # VynucenyTah=0
        GoingOverTheBoard = 0
        sloup = 0
        ColorPositionNeighbor = (ColorPosition + 2) % 4
        # kontrl = b[rad][sl][ColorPosition]
        if ColorPosition == 0:
            radok = rad
            sloup = sl - 1
            if sloup < 0:
                PermissionToUseSquare = 0
                GoingOverTheBoard = 1
        elif ColorPosition == 1:
            radok = rad - 1
            if radok < 0:
                # print('No')
                PermissionToUseSquare = 0
                GoingOverTheBoard = 1
            sloup = sl
        elif ColorPosition == 2:
            radok = rad
            sloup = sl + 1
            if sloup > n - 1:
                # print()
                # print(sloup, n)
                sloup = sloup
                PermissionToUseSquare = 0
                GoingOverTheBoard = 1
        else:
            radok = rad + 1
            if radok > m - 1:
                radok = radok
                PermissionToUseSquare = 0
                GoingOverTheBoard = 1
            sloup = sl
        # print('GOTB',GoingOverTheBoard)
        if GoingOverTheBoard == 0:
            kontrl1 = self.board[radok][sloup][ColorPositionNeighbor]
        # if self.myColor==kontrl1:
        # VynucenyTah=1
        if GoingOverTheBoard == 1:
            # VynucenyTah=0
            return PermissionToUseSquare
        if GoingOverTheBoard == 0:
            if self.board[radok][sloup] == 'none':
                PermissionToUseSquare = 0

                return PermissionToUseSquare
            elif self.myColor == kontrl1:
                PermissionToUseSquare = 1
                return PermissionToUseSquare
    def PoverkaNaDveCesty(self,radok,sloup,ColorPosition,rad,sl):
        VseVarianty=[0,1,2,3]
        # global BigControleForMove
        # self.BigControleForMove=1
        Prov=[]
        Prov0=self.board[rad][sl][ColorPosition]
        Prov.extend(Prov0)
        VseVarianty.remove(ColorPosition)
        if 1 in VseVarianty:
            # print('1')
            if radok==len(self.board)-1:
                return 1
            elif radok <0 or radok>len(self.board)-1:
                # print('q')
                return 0
            else:
                # print('q')
                Prov.extend(self.board[radok+1][sloup][1])
        if 2 in VseVarianty:
            # print('2')
            if sloup==0:
                return 1
            elif sloup<=len(self.board[0])-1 and sloup>0:
                Prov.extend(self.board[radok][sloup - 1][2])
            else:
                # print('no')
                return 0
        if 3 in VseVarianty:
            # print('3')
            if radok==0:
                return 1
            elif radok<=len(self.board)-1 and radok >0:
                Prov.extend(self.board[radok-1][sloup][3])
            else:
                return 0

        if 0 in VseVarianty:
            # print('0')
            if sloup==len(self.board[0])-1:
                return 1
            elif sloup<0 or sloup>len(self.board[0])-1:
                return 0
            else:
                Prov.extend(self.board[radok][sloup+1][0])
        # print('Prov',Prov)
        L=Prov.count('l')
        D=Prov.count('d')
        # if self.board[rad][sl][ColorPosition] == 'l':
        #     L+=1
        # if self.board[rad][sl][ColorPosition]=='d':
        #     D+=1
        # print('radok,sloup',radok,sloup)
        # print('L D',L,D)
        if L>2 or D>2 :
            self.BigControleForMove=0
            # print('BigControleForMove!!!!!!!!!',self.BigControleForMove)
            return  0

        else:
            return 1

        # Prov1=self.board[radok][sloup-1][2]
        # Prov2=self.board[radok-1][sloup][3]
        # Prov3=self.board[radok][sloup+1][0]
        # Prov4=self.board[radok+1][sloup][1]

    def VynucenyTah(self, ColorPosition, rad, sl):
        m = len(self.board)
        n = len(self.board[0])
        PossibleDrawingPoint = [0, 1, 2, 3]
        SuperPossibleDrawingPoint = []
        PermissionToUseSquare = 1
        sloup = 0
        ColorPositionNeighbor = (ColorPosition + 2) % 4
        # kontrl = self.board[rad][sl][ColorPosition]
        kontrl=self.myColor
        # print(rad,sl,kontrl)
        if ColorPosition == 0:
            radok = rad
            sloup = sl - 1
            if sloup < 0:
                PermissionToUseSquare = 0
                # print(rad,sl,'dou')
        elif ColorPosition == 1:
            radok = rad - 1
            if radok < 0:
                # print('No')
                PermissionToUseSquare = 0
            sloup = sl
        elif ColorPosition == 2:
            radok = rad
            sloup = sl + 1
            if sloup > n - 1:
                # print()
                # print(sloup, n)
                sloup = sloup
                PermissionToUseSquare = 0
            # print(radok,sloup)
        else:
            radok = rad + 1
            if radok > m - 1:
                radok = radok
                PermissionToUseSquare = 0
            sloup = sl
        if PermissionToUseSquare != 0:
            kontrl1 = self.board[radok][sloup][ColorPositionNeighbor]
            if kontrl == kontrl1:
                # print('wtf')
                PermissionToUseSquare = 0
                return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint
            if self.board[radok][sloup] == 'none':
                PoverkaNaDveCesty = self.PoverkaNaDveCesty(radok, sloup, ColorPosition, rad, sl)          #Important
                if PoverkaNaDveCesty==0:
                    PermissionToUseSquare=0
                    return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint      #Important
                # print(PoverkaNaDveCesty)
                # PermissionToUseSquare = 1
                PossibleDrawingPoint.remove(ColorPositionNeighbor)
                for i in PossibleDrawingPoint:
                    SuperPermissionToUseSquare = self.SuperVynucenyTah(i, radok, sloup)
                    if SuperPermissionToUseSquare == 1:
                        SuperPossibleDrawingPoint.append(i)
                    else:
                        continue
                if len(SuperPossibleDrawingPoint) == 0:
                    PermissionToUseSquare = 0
                return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint  # I want to make dictionary[from which point I can driwe a line]=to whith point I can drive a line
            # if kontrl == kontrl1:
            #     # print('wtf')
            #     PermissionToUseSquare = 0

            # print('wtf')
        elif PermissionToUseSquare == 0:
            return ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint
        # print(rad,sl)
        # print(self.myColor)
        # print(kontrl1)
        # return ColorPositionNeighbor, radok, sloup, 0, []

    def AnalysisAllBoard(self):
        # TestAnalysisTheBoard = []
        SolutionForVynucenyTah={}
        if self.myColor == 'l':
            AnotherColor = 'd'
        else:
            AnotherColor = 'l'
        PieceForAll = [AnotherColor, AnotherColor, AnotherColor, AnotherColor]
        PieceForAllForSome = []
        PieceForAllForSome1 = PieceForAllForSome.copy()
        PieceForAll1 = PieceForAll.copy()
        # FinalAnalysisAllBoard = {}
        # TrueFinalAnalysisAllBoard = {}
        TrueFinalAnalysisAllBoard=[]
        SuperPossibleDrawingPoint = []
        TestAnalysisTheBoard = self.AnalysisTheBoard()
        # while True:
        for i in TestAnalysisTheBoard:
            TestAnalysisSquare = self.AnalysisSquare(i)
            # print('TestAnalysisSquare',TestAnalysisSquare)
            # print(type(TestAnalysisSquare))
            rad, sl = i
            for OneOfLinePoint in TestAnalysisSquare[rad, sl]:

                PieceForAllForSome.clear()
                # print((rad,sl),OneOfLinePoint)
                ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint = self.Perechod(OneOfLinePoint, rad, sl) #ja poluchaju tolko odin variant v SuperPossibleDrawingPoint

                # print('P',ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint)
                # print(rad, sl, '(', OneOfLinePoint, ')', radok, sloup, '(', SuperPossibleDrawingPoint, ')')
                if PermissionToUseSquare == 0:
                    # print('continue')
                    continue
                if PermissionToUseSquare == 1:
                    # if (rad, sl) not in FinalAnalysisAllBoard:
                    # FinalAnalysisAllBoard[rad, sl] = [[(radok, sloup), OneOfLinePoint, SuperPossibleDrawingPoint]]
                    # for i in range(len(SuperPossibleDrawingPoint)):

                    PieceForAll[(OneOfLinePoint + 2) % 4] = self.myColor
                    PieceForAll[SuperPossibleDrawingPoint[0]] = self.myColor
                    PieceForAll = ''.join(PieceForAll)
                    # print(PieceForAll)
                    # PieceForAllForSome.append(PieceForAll)
                    # print(rad,sl,PieceForAllForSome)

                        # PieceForAllForSome=PieceForAllForSome1.copy()
                    # print((rad,sl),(radok, sloup), PieceForAllForSome)
                    # print('[(radok, sloup), PieceForAll]',[(radok, sloup), PieceForAll])
                    if [(radok, sloup), PieceForAll] not in TrueFinalAnalysisAllBoard:
                    # if (radok,sloup) not in list(SolutionForVynucenyTah.keys()):
                        TrueFinalAnalysisAllBoard.append([(radok, sloup), PieceForAll])
                        PieceForAll = PieceForAll1.copy()
                    # PieceForAllForSome = PieceForAllForSome1.copy()
                    # if len(TrueFinalAnalysisAllBoard)!=0:                       #Important
                        keySolution1=(OneOfLinePoint + 2) % 4
                        keySolution2=SuperPossibleDrawingPoint[0]
                        keySolutionlit=[keySolution1,keySolution2]
                        keySolutionlit.sort()
                        SolutionForVynucenyTah[(radok, sloup)]=keySolutionlit           #Important
                        # SolutionForVynucenyTah[(radok, sloup)] = [keySolution1]
                    # if (radok,sloup) in list(SolutionForVynucenyTah.keys()):

                    PieceForAll = PieceForAll1.copy()                # print(TrueFinalAnalysisAllBoard,'   qweqeqe   ',SolutionForVynucenyTah)
        return TrueFinalAnalysisAllBoard,SolutionForVynucenyTah
        # print(TrueFinalAnalysisAllBoard, '   qweqeqe   ', SolutionForVynucenyTah)
                        # print()
                        # print(TrueFinalAnalysisAllBoard)
                        # # PieceForAllForSome.clear()
                        # # print(TrueFinalAnalysisAllBoard)
                        # print()
                    # else:
                    #     FinalAnalysisAllBoard[rad, sl].append(
                    #         [(radok, sloup), OneOfLinePoint, SuperPossibleDrawingPoint])
                    #     # PieceForAllForSome.clear()
                    #     for i in range(len(SuperPossibleDrawingPoint)):
                    #         PieceForAll[(OneOfLinePoint + 2) % 4] = self.myColor
                    #         PieceForAll[SuperPossibleDrawingPoint[i]] = self.myColor
                    #         PieceForAll = ''.join(PieceForAll)
                    #         PieceForAllForSome.append(PieceForAll)
                    #         PieceForAll = PieceForAll1.copy()
                    #     TrueFinalAnalysisAllBoard[rad, sl].append([(radok, sloup), PieceForAllForSome])
                    #     PieceForAllForSome = PieceForAllForSome1.copy()
                    #     # PieceForAllForSome.clear()

        # return TrueFinalAnalysisAllBoard

    def FinalVynucenyTahMyColor(self,TrueFinalAnalysisAllBoard,SolutionForVynucenyTah):
        FinalVynucenyTahMyColor = []
        # FinalVynucenyTahAnotherColor=[]

        if self.myColor == 'l':
            AnotherColor = 'd'
        else:
            AnotherColor = 'l'
        PieceForAll = [AnotherColor, AnotherColor, AnotherColor, AnotherColor]
        # PieceForAllAnotherColor=[self.myColor,self.myColor,self.myColor,self.myColor]
        # PieceForAllAnotherColor1=PieceForAllAnotherColor.copy()
        # print(PieceForAll)
        PieceForAll1 = PieceForAll.copy()
        # TestAnalysisTheBoard = []
        # TestAnalysisTheBoard = self.AnalysisTheBoard()
        #
        # TestAnalysisSquare = {}
        # TestAnalysisSquare = self.AnalysisSquare(TestAnalysisTheBoard)
        # print(TestAnalysisSquare)
        SuperPossibleDrawingPoint = []
        # for koord in TestAnalysisTheBoard:
        #     rad, sl = koord
        rad,sl=TrueFinalAnalysisAllBoard[0]
        for OneOfLinePoint in SolutionForVynucenyTah[rad, sl]:
            # print((rad,sl),OneOfLinePoint)

            ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint = \
                self.VynucenyTah(OneOfLinePoint, rad, sl)
            # print('P',ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint)
            # print(rad,sl,'(',OneOfLinePoint,')',radok,sloup,'(',SuperPossibleDrawingPoint,')')
            if len(SuperPossibleDrawingPoint) != 0:
                PieceForAll[(OneOfLinePoint + 2) % 4] = self.myColor
                PieceForAll[SuperPossibleDrawingPoint[0]] = self.myColor
                PieceForAll = ''.join(PieceForAll)
                if [radok, sloup, PieceForAll] not in FinalVynucenyTahMyColor:
                    FinalVynucenyTahMyColor.append([(radok, sloup), PieceForAll])
                    keySolution1 = (OneOfLinePoint + 2) % 4
                    keySolution2 = SuperPossibleDrawingPoint[0]
                    keySolutionlit = [keySolution1, keySolution2]
                    keySolutionlit.sort()
                    FinalVynucenyTahMyColor.append(keySolutionlit)
                PieceForAll = PieceForAll1.copy()
    # print()
        return FinalVynucenyTahMyColor

    def FinalVynucenyTahAnotherColor(self,TrueFinalAnalysisAllBoard,SolutionForVynucenyTah):
        FinalVynucenyTahAnotherColor = []

        if self.myColor == 'l':
            AnotherColor = 'd'
        else:
            AnotherColor = 'l'
        self.myColor, AnotherColor = AnotherColor, self.myColor
        PieceForAll = [AnotherColor, AnotherColor, AnotherColor, AnotherColor]
        # PieceForAllAnotherColor=[self.myColor,self.myColor,self.myColor,self.myColor]
        # PieceForAllAnotherColor1=PieceForAllAnotherColor.copy()
        # print(PieceForAll)
        PieceForAll1 = PieceForAll.copy()
        # TestAnalysisTheBoard = []
        # TestAnalysisTheBoard = self.AnalysisTheBoard()
        # print(TestAnalysisTheBoard)

        # TestAnalysisSquare = {}
        # TestAnalysisSquare = self.AnalysisSquare(TestAnalysisTheBoard)
        # print(TestAnalysisSquare)
        SuperPossibleDrawingPoint = []
        # for koord in TestAnalysisTheBoard:
        #     rad, sl = koord
        SolutionForVynucenyTahAnother={}
        rad, sl = TrueFinalAnalysisAllBoard[0]
        SolutionForVynucenyTahAnother[rad,sl] = [0,1,2,3]
        # print('1',SolutionForVynucenyTah[rad, sl],(rad,sl))
        SolutionForVynucenyTahAnother[rad,sl].remove(SolutionForVynucenyTah[rad, sl][0])
        SolutionForVynucenyTahAnother[rad, sl].remove(SolutionForVynucenyTah[rad, sl][1])
        # remove1=SolutionForVynucenyTah[rad, sl][0]
        # remove2=SolutionForVynucenyTah[rad, sl][1]
        # print('r',remove1,remove2)
        # print('SolutionForVynucenyTahAnother',SolutionForVynucenyTahAnother)
        # ToCoPakZaberu=SolutionForVynucenyTahAnother[rad, sl]
        # print('ToCoPakZaberu',ToCoPakZaberu)
        for OneOfLinePoint in SolutionForVynucenyTahAnother[rad, sl]:
                # print((rad,sl),OneOfLinePoint)
                ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint = \
                    self.VynucenyTah(OneOfLinePoint, rad, sl)
                # print('P',ColorPositionNeighbor, radok, sloup, PermissionToUseSquare, SuperPossibleDrawingPoint)
                # print(rad,sl,'(',OneOfLinePoint,')',radok,sloup,'(',SuperPossibleDrawingPoint,')')
                if len(SuperPossibleDrawingPoint) != 0:
                    PieceForAll[(OneOfLinePoint + 2) % 4] = self.myColor
                    PieceForAll[SuperPossibleDrawingPoint[0]] = self.myColor
                    PieceForAll = ''.join(PieceForAll)
                    if [radok, sloup, PieceForAll] not in FinalVynucenyTahAnotherColor:
                        FinalVynucenyTahAnotherColor.append([(radok, sloup), PieceForAll])
                        keySolution1 = (OneOfLinePoint + 2) % 4
                        keySolution2 = SuperPossibleDrawingPoint[0]
                        keySolutionlit = [keySolution1, keySolution2]
                        # print('keySolution1, keySolution2',keySolution1, keySolution2)
                        keySolutionlit.sort()
                        # print('keySolutionlit',keySolutionlit)
                        keySolutionlit1=[0,1,2,3]
                        keySolutionlit1.remove(keySolutionlit[0])
                        keySolutionlit1.remove(keySolutionlit[1])
                        FinalVynucenyTahAnotherColor.append(keySolutionlit1)
                    PieceForAll = PieceForAll1.copy()
        self.myColor, AnotherColor = AnotherColor, self.myColor
        # print(self.myColor)
        return FinalVynucenyTahAnotherColor

    def move(self):
        # Perevernut=0
        # """ return list of moves:
        #     []  ... if the player cannot move
        #     [ [r1,c1,piece1], [r2,c2,piece2] ... [rn,cn,piece2] ] -place tiles to positions (r1,c1) .. (rn,cn)
        # """
        # """if self.tournament:
        # #hard strategy
        #    else:
        # """
        # rmove = []
        # TestBoard = copy.deepcopy(self.board)
        AllSquere=self.AnalysisTheBoard()
        # print(AllSquere)
        # print(len(self.board))
        # print(len(self.board[0]))
        # if len(AllSquere)==(len(self.board))*(len(self.board[0])) and len(A_LotOf_TrueFinalAnalysisAllBoard)==0:
        #     print('Wow')
        #     return []
        A_LotOf_TrueFinalAnalysisAllBoard,SolutionForVynucenyTah = self.AnalysisAllBoard()
        if len(AllSquere)==(len(self.board))*(len(self.board[0])) and len(A_LotOf_TrueFinalAnalysisAllBoard)==0:
            # print('Wow')
            return []
        if len(AllSquere)!=(len(self.board))*(len(self.board[0])) and len(A_LotOf_TrueFinalAnalysisAllBoard)==0:
            if self.myColor == 'l':
                AnotherColor = 'd'
            else:
                AnotherColor = 'l'
            self.myColor, AnotherColor = AnotherColor, self.myColor
            A_LotOf_TrueFinalAnalysisAllBoard, SolutionForVynucenyTah = self.AnalysisAllBoard()
            Perevernut = 1

            # self.myColor, AnotherColor = AnotherColor, self.myColor
        # print('A_LotOf_TrueFinalAnalysisAllBoard',A_LotOf_TrueFinalAnalysisAllBoard)
        c = []
        c = copy.deepcopy(self.board)
        for TrueFinalAnalysisAllBoard in A_LotOf_TrueFinalAnalysisAllBoard:
            rmove = []
            self.board = copy.deepcopy(c)
            # print('Start__TrueFinalAnalysisAllBoard',TrueFinalAnalysisAllBoard)
            Stop = 0
            self.BigControleForMove = 1
        # a = ((list(TrueFinalAnalysisAllBoard.keys())[0]))
            rad, sl = TrueFinalAnalysisAllBoard[0]
            piece1 = TrueFinalAnalysisAllBoard[1]
            # rmove.append([rad, sl, piece1])
            # print('rmove',rmove)
            # c = []
            # c = copy.deepcopy(self.board)
            rmoveForLoop = []
            SolutionForVynucenyTahAnother={}
            # TrueFinalAnalysisAllBoardAnother=TrueFinalAnalysisAllBoard.copy()
            # SolutionForVynucenyTahAnother[rad, sl]=SolutionForVynucenyTah[rad, sl]
            ForLoopTrueFinalAnalysisAllBoard=[]
            ForLoopTrueFinalAnalysisAllBoard.append(TrueFinalAnalysisAllBoard)
            # print('ForLoopTrueFinalAnalysisAllBoard',ForLoopTrueFinalAnalysisAllBoard)
            ForLoopSolutionForVynucenyTah={}
            ForLoopSolutionForVynucenyTah[rad, sl] = SolutionForVynucenyTah[rad, sl]
            # print('ForLoopSolutionForVynucenyTah',ForLoopSolutionForVynucenyTah)
            ForLoopInLoopSolutionForVynucenyTah={}
            ForLoopControle=[]
            r1,s1=TrueFinalAnalysisAllBoard[0]
            t=TrueFinalAnalysisAllBoard[1]
            ForLoopControle.append([r1,s1,t])
            # ForLoopControle.append(TrueFinalAnalysisAllBoard)
            ForLoopTrueFinalAnalysisAllBoardNaPotom=[]
            while True:
                if Stop==1:
                    break
                # self.board[rad][sl] = piece1
                for i in range(len(ForLoopTrueFinalAnalysisAllBoard)):
                    rad1,sl1=ForLoopTrueFinalAnalysisAllBoard[i][0]
                    piece111=ForLoopTrueFinalAnalysisAllBoard[i][1]
                    self.board[rad1][sl1] = piece111
                    # print('i',i)
                    # print('ForLoopTrueFinalAnalysisAllBoardNachaloCykla',ForLoopTrueFinalAnalysisAllBoard)
                    ForLoopInLoopSolutionForVynucenyTah[ForLoopTrueFinalAnalysisAllBoard[i][0]]=ForLoopSolutionForVynucenyTah[ForLoopTrueFinalAnalysisAllBoard[i][0]]
                    # print('ForLoopTrueFinalAnalysisAllBoard',ForLoopTrueFinalAnalysisAllBoard[i])
                    # print('ForLoopInLoopSolutionForVynucenyTah',ForLoopInLoopSolutionForVynucenyTah)
                    # print('ForLoopTrueFinalAnalysisAllBoard[i]',ForLoopTrueFinalAnalysisAllBoard[i])
                    FinalVynucenyTah = self.FinalVynucenyTahMyColor(ForLoopTrueFinalAnalysisAllBoard[i],ForLoopInLoopSolutionForVynucenyTah)
                    # print('FinalVynucenyTah',FinalVynucenyTah)
                    FinalVynucenyTahAnotherColor = self.FinalVynucenyTahAnotherColor(ForLoopTrueFinalAnalysisAllBoard[i],ForLoopInLoopSolutionForVynucenyTah)
                    # print('FinalVynucenyTahAnotherColor',FinalVynucenyTahAnotherColor)
                    # print('BigControleForMove',self.BigControleForMove)
                    # ForLoopTrueFinalAnalysisAllBoard.remove(ForLoopTrueFinalAnalysisAllBoard[i])
                    del ForLoopInLoopSolutionForVynucenyTah[((list(ForLoopInLoopSolutionForVynucenyTah.keys())[0]))]
                    if self.BigControleForMove == 0 :
                        # print('!!!!!!!!!!!Stop')
                        Stop=1
                        break
                    # print('ForLoopInLoopSolutionForVynucenyTahDel', ForLoopInLoopSolutionForVynucenyTah)
                    # print('ForLoopTrueFinalAnalysisAllBoard', ForLoopTrueFinalAnalysisAllBoard)
                    # print(self.myColor)
                    if len(FinalVynucenyTah) != 0:
                        for ii in range(0,len(FinalVynucenyTah)-1,2):
                        # rmoveForLoop.append(FinalVynucenyTah[0])
                            radok,sloup=FinalVynucenyTah[ii][0]
                            piece2=FinalVynucenyTah[ii][1]
                            if [radok,sloup,piece2] not in ForLoopControle:
                                # rmoveForLoop.append(FinalVynucenyTah[0])
                                ForLoopControle.append([radok,sloup,piece2])
                                # rmoveForLoop.append([(radok,sloup),piece2])
                                ForLoopTrueFinalAnalysisAllBoardNaPotom.append([(radok,sloup),piece2])
                                ForLoopSolutionForVynucenyTah[radok, sloup] = FinalVynucenyTah[ii+1]
                        # print('ForLoopControle', ForLoopControle)
                        # print('ForLoopTrueFinalAnalysisAllBoardNaPotom!!!My', ForLoopTrueFinalAnalysisAllBoardNaPotom)
                        # print('ForLoopSolutionForVynucenyTah!!!My',ForLoopSolutionForVynucenyTah)
                        # del SolutionForVynucenyTah[((list(SolutionForVynucenyTah.keys())[0]))]

                        # print('SolutionForVynucenyTah',SolutionForVynucenyTah)
                        # for i in FinalVynucenyTah:
                        #     radok, sloup, piece2 = i
                        #     self.board[radok][sloup] = piece2
                    if len(FinalVynucenyTahAnotherColor) != 0:
                        for ii in range(0, len(FinalVynucenyTahAnotherColor) - 1, 2):
                        # print('rmoveForLoop',rmoveForLoop)
                            radok1,sloup1=FinalVynucenyTahAnotherColor[ii][0]
                            piece3=FinalVynucenyTahAnotherColor[ii][1]
                            if [radok1,sloup1,piece3] not in ForLoopControle:
                                # rmoveForLoop.append(FinalVynucenyTahAnotherColor[0])
                                ForLoopControle.append([radok1,sloup1,piece3])
                                ForLoopSolutionForVynucenyTah[radok1, sloup1] = FinalVynucenyTahAnotherColor[ii+1]
                                # print('ForLoopControle',ForLoopControle)
                                # rmoveForLoop.append([(radok1,sloup1),piece3])
                                ForLoopTrueFinalAnalysisAllBoardNaPotom.append([(radok1,sloup1),piece3])
                        # ForLoopTrueFinalAnalysisAllBoard.append([(radok1,sloup1),piece3])
                        # print('ForLoopTrueFinalAnalysisAllBoardNaPotom!!!An', ForLoopTrueFinalAnalysisAllBoardNaPotom)
                        # print('ForLoopControle', ForLoopControle)
                        # print('ForLoopSolutionForVynucenyTah!!!Ay', ForLoopSolutionForVynucenyTah)
                        # print('TrueFinalAnalysisAllBoardAnother',TrueFinalAnalysisAllBoardAnother)
                        # del SolutionForVynucenyTah[((list(SolutionForVynucenyTah.keys())[0]))]
                        # ForLoopSolutionForVynucenyTah[radok1,sloup1]=FinalVynucenyTahAnotherColor[1]
                        # print('SolutionForVynucenyTahAnother',SolutionForVynucenyTah)
                    # if len(rmoveForLoop) == 0:
                ForLoopTrueFinalAnalysisAllBoard.clear()
                ForLoopTrueFinalAnalysisAllBoard.extend(ForLoopTrueFinalAnalysisAllBoardNaPotom)
                ForLoopTrueFinalAnalysisAllBoardNaPotom.clear()
                # print('ForLoopTrueFinalAnalysisAllBoardPoslePotom',ForLoopTrueFinalAnalysisAllBoard)
                    #     break
                # rmove.extend(rmoveForLoop)
                # print('rmove1',rmove)
                # rmoveForLoop.clear()
                if len(ForLoopTrueFinalAnalysisAllBoard)==0:
                    rmove.extend(ForLoopControle[:])
                    return rmove
                    # print('rmove2',rmove)
                # if len(rmoveForLoop) != 0:
                #     rmove.extend(rmoveForLoop)
            if Stop==1:
                ForLoopControle.clear()
            if len(ForLoopControle)!=0:
                rmove.extend(ForLoopControle[:])
                # print('rmoveF',rmove)
        self.board = copy.deepcopy(c)
        return rmove


if __name__ == "__main__":
    # call you functions from this block:

    boardRows = 10
    boardCols = boardRows
    board = [["none"] * boardCols for _ in range(boardRows)]

    board[boardRows // 2][boardCols // 2] = ["lldd", "dlld", "ddll", "lddl", "dldl", "ldld"][random.randint(0, 5)]

    d = Drawer.Drawer()

    p1 = Player(board, "player1", 'l');
    p2 = Player(board, "player2", 'd');

    # test game. We assume that both player play correctly. In Brute/Tournament case, more things will be checked
    # like types of variables, validity of moves, etc...

    idx = 0
    while True:

        # call player for his move
        rmove = p1.move()

        # rmove is: [ [r1,c1,tile1], ... [rn,cn,tile] ]
        # write to board of both players
        for move in rmove:
            row, col, tile = move
            p1.board[row][col] = tile
            p2.board[row][col] = tile

        # make png with resulting board
        d.draw(p1.board, "move-{:04d}.png".format(idx))
        idx += 1

        if len(rmove) == 0:
            print("End of game")
            break
        p1, p2 = p2, p1  # switch players