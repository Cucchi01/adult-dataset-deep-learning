# ----------------------------------------------------------------------
# creates the hot encoded version of the data set adult
# The list of feature is known and fixed. Some of them are mapped
# directly with an integer, but some of them are hot encoded because
#  there is not a order between the possible values of that feuture.
# The class HotEconding takes the list of possible values and creates a
# mapping to a list of string of "1" and "0". Infact at this stage the
# tensor still contains strings.
#
# In "hotEncoding" function are fist created all the various encoding
# for the attributes and then are used to create a new tensor
#
# Andrea Cucchietti, 2022
# ----------------------------------------------------------------------


import numpy as np

LIST_NAMES_FEATURE = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                      "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]


class HotEncoding(object):
    def __init__(self, listPossibleValue) -> None:
        self._map = {}
        self._listPossibleValues = listPossibleValue
        self._numPossibleValues = len(listPossibleValue)
        self.__generateMapping()

    def __generateMapping(self):
        for (i, word) in enumerate(self._listPossibleValues):
            wordHotEncoded = np.zeros(self._numPossibleValues)
            wordHotEncoded[i] = 1
            self._map[word] = wordHotEncoded.astype(dtype="S1")

    def getNumPossibleValues(self):
        return self._numPossibleValues

    def getEncodedFromValue(self, value):
        return self._map[value]

    def HotEncodeOldRow(self, oldRow, newRow, startNewPosFeature, oldPosFeature):
        ''' changes the row accordingly to the hot encoding and then returns the last position of the feature encoded'''
        endPosFeature = startNewPosFeature + self._numPossibleValues
        newRow[startNewPosFeature:endPosFeature] = self.getEncodedFromValue(
            oldRow[oldPosFeature])
        return endPosFeature


def generateHotEncoding(adultData):
    possibleValuesForFeature = {
        "sex": [b"Female", b"Male", b"?"],
        "workclass": [b"Private", b"Self-emp-not-inc", b"Self-emp-inc",
                      b"Federal-gov", b"Local-gov", b"State-gov", b"Without-pay", b"Never-worked", b"?"],
        "marital-status": [b"Married-civ-spouse", b"Divorced", b"Never-married",
                           b"Separated", b"Widowed", b"Married-spouse-absent", b"Married-AF-spouse", b"?"],
        "occupation": [b"Tech-support", b"Craft-repair", b"Other-service", b"Sales", b"Exec-managerial", b"Prof-specialty", b"Handlers-cleaners",
                       b"Machine-op-inspct", b"Adm-clerical", b"Farming-fishing", b"Transport-moving", b"Priv-house-serv", b"Protective-serv", b"Armed-Forces", b"?"],
        "relationship": [b"Wife", b"Own-child", b"Husband",
                         b"Not-in-family", b"Other-relative", b"Unmarried", b"?"],
        "race": [b"White", b"Asian-Pac-Islander",
                 b"Amer-Indian-Eskimo", b"Other", b"Black", b"?"],
        "native-country": [b"United-States", b"Cambodia", b"England", b"Puerto-Rico", b"Canada", b"Germany", b"Outlying-US(Guam-USVI-etc)", b"India", b"Japan", b"Greece", b"South", b"China", b"Cuba", b"Iran", b"Honduras", b"Philippines", b"Italy", b"Poland", b"Jamaica", b"Vietnam", b"Mexico",
                           b"Portugal", b"Ireland", b"France", b"Dominican-Republic", b"Laos", b"Ecuador", b"Taiwan", b"Haiti", b"Columbia", b"Hungary", b"Guatemala", b"Nicaragua", b"Scotland", b"Thailand", b"Yugoslavia", b"El-Salvador", b"Trinadad&Tobago", b"Peru", b"Hong", b"Holand-Netherlands", b"?"]
    }

    dictHotEncoding = generateEncodings(possibleValuesForFeature)
    newNumberOfColumns = getNewNumCol(dictHotEncoding, adultData)

    newAdultData = generateNewTensor(
        adultData, newNumberOfColumns, possibleValuesForFeature, dictHotEncoding)
    del adultData
    adultData = newAdultData

    return adultData


def generateEncodings(possibleValuesForFeature):
    dictHotEncoding = {}
    for nameFeature in possibleValuesForFeature:
        Encoding = HotEncoding(possibleValuesForFeature[nameFeature])
        dictHotEncoding[nameFeature] = Encoding
    return dictHotEncoding


def getNewNumCol(dictHotEncoding, adultData):
    newNumberOfColumns = adultData.shape[1]
    for nameFeature in dictHotEncoding:
        # -1 because 1 element for this feature was already existing
        newNumberOfColumns += dictHotEncoding[nameFeature].getNumPossibleValues() - 1
    return newNumberOfColumns


def generateNewTensor(adultData, newNumberOfColumns, possibleValuesForFeature, dictHotEncoding):
    # TODO: change the dimention accordingly to the max lenght of the strings
    newAdultData = np.empty(
        dtype="S30", shape=[adultData.shape[0], newNumberOfColumns])
    listNamesHotEncodedFeatures = possibleValuesForFeature.keys()
    fillNewTensor(newAdultData, adultData,
                  listNamesHotEncodedFeatures, dictHotEncoding)
    return newAdultData


def fillNewTensor(newAdultData, adultData, listNamesHotEncodedFeatures, dictHotEncoding):
    for newRow, oldRow in zip(newAdultData, adultData):
        fillRowNewTensor(
            oldRow, newRow, listNamesHotEncodedFeatures, dictHotEncoding)


def fillRowNewTensor(oldRow, newRow, listNamesHotEncodedFeatures, dictHotEncoding):
    posInRow = 0
    oldPosInRow = 0
    for nameFeature in LIST_NAMES_FEATURE:
        if nameFeature in listNamesHotEncodedFeatures:
            Encoding = dictHotEncoding[nameFeature]
            posInRow = Encoding.HotEncodeOldRow(
                oldRow, newRow, posInRow, oldPosInRow)
        else:
            newRow[posInRow] = oldRow[oldPosInRow]
            posInRow += 1
        oldPosInRow += 1
