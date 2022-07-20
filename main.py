import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os.path
import hotEncoding as hE
import pandas as pd
import matplotlib.pyplot as plt

N_EPOCHS = 20000
LR = 1e-3


class Model(nn.Module):
    def __init__(self, numInFeatures):
        super(Model, self).__init__()
        self.hidden_layer = nn.Linear(numInFeatures, 1)
        #self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(1, 1)
        torch.nn.init.normal_(self.output_linear.weight, mean=0, std=1e-9)

    def forward(self, input):
        hidden_t = self.hidden_layer(input)
        #activated_t = self.hidden_activation(hidden_t)
        #output_t = self.output_linear(activated_t)
        output_t = self.output_linear(hidden_t)
        return output_t


def main():
    trainData = loadData("adultEncoded.npy", "adult.data")
    testData = loadData("testEncoded.npy", "adult.test")

    trainTensor = torch.from_numpy(trainData)
    testTensor = torch.from_numpy(testData)
    print(trainTensor[0])
    normalizeTensor(trainTensor)
    normalizeTensor(testTensor)

    t_inp_train = trainTensor[:, 0: trainTensor.shape[1]-1]
    t_inp_test = testTensor[:, 0: trainTensor.shape[1]-1]

    t_res_train = trainTensor[:, trainTensor.shape[1]-1]
    t_res_test = testTensor[:, trainTensor.shape[1]-1]
    t_res_train = t_res_train.unsqueeze(1)
    t_res_test = t_res_test.unsqueeze(1)

    model = Model(t_inp_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainingLoop(n_epochs=N_EPOCHS, optimizer=optimizer, model=model, loss_fn=nn.MSELoss(
    ), t_inp_train=t_inp_train, t_inp_test=t_inp_test,  t_res_train=t_res_train, t_res_test=t_res_test)

    torch.save(model.state_dict(), 'model/model.pth')

    print('ouput', model(t_inp_test))

    outputTensor = model(t_inp_test)

    for i in range(outputTensor.shape[0]):
        if outputTensor[i] <= 0.5:
            outputTensor[i] = 0
        else:
            outputTensor[i] = 1

    difference = outputTensor-t_res_test
    print(difference, difference.shape)
    difference = torch.abs(difference)

    print("ratio errors: " + str(torch.sum(difference)/difference.shape[0]))
    print('answer', t_res_test)
    # print('hidden', model.hidden_layer.weight.grad)


def loadData(nameNpyFile, nameFileData):
    if os.path.exists("data/"+nameNpyFile):
        adultData = np.load("data/"+nameNpyFile)
    else:
        adultData = loadDataFromFile(nameFileData)
        np.save("data/"+nameNpyFile, adultData)
    return adultData


def loadDataFromFile(nameFile):
    adultData = np.genfromtxt("data/"+nameFile, delimiter=", ", dtype="S")
    adultData = mapWords(adultData)
    adultData = hE.generateHotEncoding(adultData)

    adultData = adultData.astype(dtype=np.float32)

    return adultData


def normalizeTensor(adultData):
    n_channels = adultData.shape[1]
    for i in range(0, n_channels-1):
        # -1 because we don't want to normalize the resulta that is already 0 or 1

        max = torch.max(adultData[:, i])
        if max > 1e-8:
            min = torch.min(adultData[:, i])
            adultData[:, i] = (adultData[:, i] - min) / max


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


def trainingLoop(n_epochs, optimizer, model, loss_fn, t_inp_train, t_inp_test, t_res_train, t_res_test):
    results = np.array([["Epoch", "Training Loss", "Test Loss"]])
    for epoch in range(1, n_epochs+1):
        t_p_train = model(t_inp_train)
        loss_train = loss_fn(t_p_train, t_res_train)

        t_p_val = model(t_inp_test)
        loss_val = loss_fn(t_p_val, t_res_test)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        results = np.append(
            results, [[epoch, float(loss_train), float(loss_val)]], axis=0)

        if epoch == 1 or epoch % 100 == 0:
            print("Epoch {}, Training loss {}, Validation loss {},".format(
                epoch, float(loss_train), float(loss_val)))

    printTrainingResults(results, n_epochs)


def printTrainingResults(results, n_epochs):
    # n_epochs+1 because there are the names of the columns in the first row
    results.reshape(n_epochs+1, 3)
    index = results[1:, 0]
    data = results[1:, 1:]
    columns = results[0, 1:]
    resultDF = pd.DataFrame(
        data=data, index=index, columns=columns)
    resultDF['Training Loss'] = resultDF['Training Loss'].astype(float)
    resultDF['Test Loss'] = resultDF['Test Loss'].astype(float)
    resultDF.plot()
    plt.show()


main()
