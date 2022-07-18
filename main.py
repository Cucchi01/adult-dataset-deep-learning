import numpy as np
import torch


def main():
    adultData = loadData()


def loadData():
    adultData = np.genfromtxt("data/adult.data", delimiter=", ", dtype="S")

    mapWords(adultData)

    hotEncoding(adultData)

    return adultData


def mapWords(adultData):
    education2Int = {
        b"?": b"0",
        b"Preschool": b"1",
        b"1st-4th": b"2",
        b"5th-6th": b"3",
        b"7th-8th": b"4",
        b"9th": b"5",
        b"10th": b"6",
        b"11th": b"7",
        b"12th": b"8",
        b"HS-grad": b"9",
        b"Prof-school": b"10",
        b"Assoc-acdm": b"11",
        b"Assoc-voc": b"12",
        b"Some-college": b"12",
        b"Bachelors": b"14",
        b"Masters": b"15",
        b"Doctorate": b"16",
    }

    mapIncome2Int = {
        b"<=50K": b"0",
        b">50K": b"1"
    }

    for row in adultData:
        row[3] = education2Int[row[3]]
        row[14] = mapIncome2Int[row[14]]



def hotEncoding(adultData):
    sexHotEncoding = {
        b"Female": [b"1", b"0", b"0"],
        b"Male": [b"0", b"1", b"0"],
        b"?": [b"0", b"0", b"1"]
    }
    # -1 because 1 elemnt for this feature was already existing
    numSex = len(sexHotEncoding[b"Female"])
    newNumberOfColumns = adultData.shape[1]+numSex-1

    workClassHotEncoding = {}
    workClassList = [b"Private", b"Self-emp-not-inc", b"Self-emp-inc",
                     b"Federal-gov", b"Local-gov", b"State-gov", b"Without-pay", b"Never-worked", b"?"]
    numWorkClass = len(workClassList)
    newNumberOfColumns += numWorkClass-1    
    for (i, word) in enumerate(workClassList):
        wordHotEncoded = np.zeros(len(workClassList))
        wordHotEncoded[i] = 1
        workClassHotEncoding[word] = wordHotEncoded.astype(dtype="S1")


    maritalHotEncoding = {}
    maritalList = [b"Married-civ-spouse", b"Divorced", b"Never-married",
                   b"Separated", b"Widowed", b"Married-spouse-absent", b"Married-AF-spouse", b"?"]
    numMarital = len(maritalList)
    newNumberOfColumns += numMarital-1
    for (i, word) in enumerate(maritalList):
        wordHotEncoded = np.zeros(len(maritalList))
        wordHotEncoded[i] = 1
        maritalHotEncoding[word] = wordHotEncoded.astype(dtype="S1")

    occupationHotEncoding = {}
    occupationList = [b"Tech-support", b"Craft-repair", b"Other-service", b"Sales", b"Exec-managerial", b"Prof-specialty", b"Handlers-cleaners",
                      b"Machine-op-inspct", b"Adm-clerical", b"Farming-fishing", b"Transport-moving", b"Priv-house-serv", b"Protective-serv", b"Armed-Forces", b"?"]
    numOccupation = len(occupationList)
    newNumberOfColumns += numOccupation-1
    for (i, word) in enumerate(occupationList):
        wordHotEncoded = np.zeros(len(occupationList))
        wordHotEncoded[i] = 1
        occupationHotEncoding[word] = wordHotEncoded.astype(dtype="S1")

    relationshipHotEncoding = {}
    relationshipList = [b"Wife", b"Own-child", b"Husband",
                        b"Not-in-family", b"Other-relative", b"Unmarried", b"?"]
    numRelationship = len(relationshipList)
    newNumberOfColumns += numRelationship-1
    for (i, word) in enumerate(relationshipList):
        wordHotEncoded = np.zeros(len(relationshipList))
        wordHotEncoded[i] = 1
        relationshipHotEncoding[word] = wordHotEncoded.astype(dtype="S1")

    raceHotEncoding = {}
    raceList = [b"White", b"Asian-Pac-Islander",
                b"Amer-Indian-Eskimo", b"Other", b"Black", b"?"]
    numRace = len(raceList)
    newNumberOfColumns += numRace-1
    for (i, word) in enumerate(raceList):
        wordHotEncoded = np.zeros(len(raceList))
        wordHotEncoded[i] = 1
        raceHotEncoding[word] = wordHotEncoded.astype(dtype="S1")

    nativeCountryHotEncoding = {}
    nativeCountryList = [b"United-States", b"Cambodia", b"England", b"Puerto-Rico", b"Canada", b"Germany", b"Outlying-US(Guam-USVI-etc)", b"India", b"Japan", b"Greece", b"South", b"China", b"Cuba", b"Iran", b"Honduras", b"Philippines", b"Italy", b"Poland", b"Jamaica", b"Vietnam", b"Mexico",
                         b"Portugal", b"Ireland", b"France", b"Dominican-Republic", b"Laos", b"Ecuador", b"Taiwan", b"Haiti", b"Columbia", b"Hungary", b"Guatemala", b"Nicaragua", b"Scotland", b"Thailand", b"Yugoslavia", b"El-Salvador", b"Trinadad&Tobago", b"Peru", b"Hong", b"Holand-Netherlands", b"?"]
    numNativeCountry = len(nativeCountryList)
    newNumberOfColumns += numNativeCountry-1
    for (i, word) in enumerate(nativeCountryList):
        wordHotEncoded = np.zeros(len(nativeCountryList))
        wordHotEncoded[i] = 1
        nativeCountryHotEncoding[word] = wordHotEncoded.astype(dtype="S1")

    # TODO: change the dimention acconrdingly to the max lenght
    newAdultData = np.empty(
        dtype="S30", shape=[adultData.shape[0], newNumberOfColumns])
    for newRow, oldRow in zip(newAdultData, adultData):
        newRow[0] = oldRow[0]
        newRow[1:1+numWorkClass] = workClassHotEncoding[oldRow[1]]
        n = 1+numWorkClass
        newRow[n:n+3] = oldRow[2:5]
        n+=3
        newRow[n: n+numMarital] = maritalHotEncoding[oldRow[5]]
        n+=numMarital
        newRow[n:n+numOccupation] = occupationHotEncoding[oldRow[6]]
        n+= numOccupation
        newRow[n: n+numRelationship] = relationshipHotEncoding[oldRow[7]]
        n+=numRelationship
        newRow[n: n+numRace] = raceHotEncoding[oldRow[8]]
        n+=numRace
        newRow[n: n+numSex] = sexHotEncoding[oldRow[9]]
        n+=numSex
        newRow[n: n+3] = oldRow[10:13]
        n+=3
        newRow[n: n+numNativeCountry] = nativeCountryHotEncoding[oldRow[13]]
        n+=numNativeCountry
        newRow[n] = oldRow[14]


    del adultData
    adultData = newAdultData
    print(newAdultData[0])

    adultData = adultData.astype(dtype=np.float32)
    
main()
