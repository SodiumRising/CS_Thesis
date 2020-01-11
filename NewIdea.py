# import statements here
from PIL import Image

def main():

    hiraganaVowels = buildCharacters()
    quizStudent(hiraganaVowels)

def buildCharacters():

    # lists
    hiraganaVowels = []
    hiraganaStrokes = getCharacterStrokes()

    # create stroke orders
    astrokes = combineStrokes('a', hiraganaStrokes)
    istrokes = combineStrokes('i', hiraganaStrokes)
    ustrokes = combineStrokes('u', hiraganaStrokes)
    estrokes = combineStrokes('e', hiraganaStrokes)
    ostrokes = combineStrokes('o', hiraganaStrokes)

    # Put characters together
    a = createCharacters('a', 3, astrokes)
    i = createCharacters('i', 2, istrokes)
    u = createCharacters('u', 2, ustrokes)
    e = createCharacters('e', 2, estrokes)
    o = createCharacters('o', 3, ostrokes)

    # add completed characters to finished list
    hiraganaVowels.append(a)
    hiraganaVowels.append(i)
    hiraganaVowels.append(u)
    hiraganaVowels.append(e)
    hiraganaVowels.append(o)

    return hiraganaVowels


def createCharacters(character, strokeCount, strokeOrder):

    currentCharacter = makeHiragana(character, strokeCount, strokeOrder)

    return currentCharacter


class Hiragana(object):

    character = ""
    strokeCount = 0
    strokeOrder = []

    def __init__(self, character, strokeCount, strokeOrder):
        self.character = character
        self.strokeCount = strokeCount
        self.strokeOrder = strokeOrder


def makeHiragana(character, strokeOrder, strokeCount):

    hiragana = Hiragana(character, strokeOrder, strokeCount)
    return hiragana


def getCharacterStrokes():

    hiraganaStrokes = []

    try:
        horizontalTop = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes\horizontaltop.png')
        abody = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/abody.png')
        abottom = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/abottom.png')
        ileft = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ileft.png')
        iright = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/iright.png')
        smallhorizontaltop = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/smallhorizontaltop.png')
        ubody = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ubody.png')
        ebody = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ebody.png')
        obody = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/obody.png')
        odash = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/odash.png')

    except IOError:
        pass

    hiraganaStrokes.append(horizontalTop)
    hiraganaStrokes.append(abody)
    hiraganaStrokes.append(abottom)
    hiraganaStrokes.append(ileft)
    hiraganaStrokes.append(iright)
    hiraganaStrokes.append(smallhorizontaltop)
    hiraganaStrokes.append(ubody)
    hiraganaStrokes.append(ebody)
    hiraganaStrokes.append(obody)
    hiraganaStrokes.append(odash)

    return hiraganaStrokes


def combineStrokes(character, hiraganaStrokes):

    strokeOrder = []

    if character == 'a':

        strokeOrder.append(hiraganaStrokes[0])
        strokeOrder.append(hiraganaStrokes[1])
        strokeOrder.append(hiraganaStrokes[2])

        return strokeOrder

    if character == 'i':

        strokeOrder.append(hiraganaStrokes[3])
        strokeOrder.append(hiraganaStrokes[4])

        return strokeOrder

    if character == 'u':

        strokeOrder.append(hiraganaStrokes[5])
        strokeOrder.append(hiraganaStrokes[6])

        return strokeOrder


    if character == 'e':

        strokeOrder.append(hiraganaStrokes[5])
        strokeOrder.append(hiraganaStrokes[7])

        return strokeOrder

    if character == 'o':

        strokeOrder.append(hiraganaStrokes[0])
        strokeOrder.append(hiraganaStrokes[8])
        strokeOrder.append(hiraganaStrokes[9])

        return strokeOrder

    strokeOrder.clear()


def quizStudent(hiraganaVowels):

    usrCount = int(input("How many strokes are in the character ", hiraganaVowels[0].character, "? \n"))

    if hiraganaVowels[0].strokeCount == usrCount:

        print("Correct!")

    else:

        print("You suck.")


main()
