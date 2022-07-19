import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import hotEncoding as hE


def main():
    if os.path.exists("data/adultEncoded.npy"):
        adultData = np.load("data/adultEncoded.npy")
    else:
        adultData = loadData()
        np.save("data/adultEncoded.npy", adultData)

    adultTensor = torch.from_numpy(adultData)
    normalizeTensor(adultTensor)
    print(adultTensor, adultTensor.shape)


def loadData():
    adultData = np.genfromtxt("data/adult.data", delimiter=", ", dtype="S")
    adultData = mapWords(adultData)
    adultData = hE.generateHotEncoding(adultData)

    adultData = adultData.astype(dtype=np.float32)

    return adultData


def normalizeTensor(adultData):
    n_channels = adultData.shape[1]
    for i in range(0, n_channels):
        mean = torch.mean(adultData[i, :])
        std = torch.std(adultData[i, :])
        adultData[i, :] = (adultData[i, :] - mean) / std


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

    return adultData


def trainingLoop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs+1):
        t_p


main()
