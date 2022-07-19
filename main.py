import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import hotEncoding as hE

N_EPOCHS = 120
LR = 5e-4


class Model(nn.Module):
    def __init__(self, numInFeatures):
        super(Model, self).__init__()
        self.hidden_layer = nn.Linear(numInFeatures, 17)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(17, 1)
        torch.nn.init.normal_(self.output_linear.weight, mean=0, std=1e-9)

    def forward(self, input):
        hidden_t = self.hidden_layer(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
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
    t_res_val = testTensor[:, trainTensor.shape[1]-1]
    t_res_train = t_res_train.unsqueeze(1)
    t_res_val = t_res_val.unsqueeze(1)

    print(t_inp_train, t_inp_train.shape)
    print(t_inp_test, t_inp_test.shape)
    print(t_res_train, t_res_train.shape)
    print(t_res_val, t_res_val.shape)

    model = Model(t_inp_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainingLoop(n_epochs=N_EPOCHS, optimizer=optimizer, model=model, loss_fn=nn.MSELoss(
    ), t_inp_train=t_inp_train, t_inp_val=t_inp_test, t_res_val=t_res_val, t_res_train=t_res_train)


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


def trainingLoop(n_epochs, optimizer, model, loss_fn, t_inp_train, t_inp_val, t_res_train, t_res_val):
    for epoch in range(1, n_epochs+1):
        t_p_train = model(t_inp_train)
        loss_train = loss_fn(t_p_train, t_res_train)

        t_p_val = model(t_inp_val)
        loss_val = loss_fn(t_p_val, t_res_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print("Epoch {}, Training loss {}, Validation loss {},".format(
              epoch, float(loss_train), float(loss_val)))

    torch.save(model.state_dict(), 'model/model.pth')
    
    #TODO: make the output either 0 or 1 and confront
    print('ouput', model(t_inp_train))
    print('answer', t_res_train)
    # print('hidden', model.hidden_layer.weight.grad)


main()
