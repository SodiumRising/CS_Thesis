# import statements here
import numpy as np
import cv2

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
        horizontalTop = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes\horizontaltop.png')
        abody = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/abody.png')
        abottom = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/abottom.png')
        ileft = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ileft.png')
        iright = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/iright.png')
        smallhorizontaltop = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/smallhorizontaltop.png')
        ubody = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ubody.png')
        ebody = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ebody.png')
        obody = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/obody.png')
        odash = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/odash.png')

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

    # correct/incorrect flag
    correct = True

    print("How many strokes are in the character", hiraganaVowels[0].character, "? \n")
    usrCount = int(input("Strokes: "))

    # test data

    # correct stroke order
    correctOrder = hiraganaVowels[0].strokeOrder

    # all wrong stroke order
    incorrectOrder = []

    ileft = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/ileft.png')
    iright = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/iright.png')
    smallhorizontaltop = cv2.imread(r'C:\Users\loofa\Desktop\TestHiragana\HiraganaStrokes/smallhorizontaltop.png')
    incorrectOrder.append(ileft)
    incorrectOrder.append(iright)
    incorrectOrder.append(smallhorizontaltop)


    if hiraganaVowels[0].strokeCount == usrCount:

        print("Correct!")
        result = compareImages(incorrectOrder, hiraganaVowels[0].strokeOrder)

        if result > 0:

            correct = False
            print("Incorrect stroke order")

        else:

            correct = True
            print("Correct stroke order")

    else:

        print("You suck.")


def compareImages(usrOrder, correctOrder):

    for i in range(len(usrOrder)):

        imageA = usrOrder[i]
        imageB = correctOrder[i]

        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are

        print(err)
        return err


main()
