"""
# @author: xuwang
# @function:
# @date: 2018/4/2 10:57
"""

import time
import math

class Viterbi:
    def __init__(self, filename):
        self.filename = filename  # 语料文件
        self.beginDict = {}  # 统计以词性x开头的次数
        self.composition = {}  # 词性转移矩阵
        self.compoVocab = {}  # 词性到单词的统计
        self.pi = {}
        self.A = {}
        self.B = {}
        self.trainSize = 0
        self.testList = []
        self.seg = 0

    def viterbi(self, sentenceList):

        delta = []
        phi = []
        N = len(self.composition)
        T = len(sentenceList)
        compoList = list(self.composition.keys())
        # 初始化
        delta.append([])
        phi.append([])
        for i in range(N):
            if sentenceList[0] in self.B[compoList[i]].keys():
                if compoList[i] in self.pi.keys():
                    delta[0].append(math.log(self.pi[compoList[i]]) + math.log(self.B[compoList[i]][sentenceList[0]]))
                    phi[0].append(0)
                else:
                    delta[0].append(math.log(1 / (self.trainSize + 1)) + math.log(self.B[compoList[i]][sentenceList[0]]))
                    phi[0].append(0)

            else:
                if compoList[i] in self.pi.keys():
                    delta[0].append(math.log(self.pi[compoList[i]]) + math.log(1 / 900000))
                    phi[0].append(0)
                else:
                    delta[0].append(math.log(1 / (self.trainSize + 1)) + math.log(1 / 900000))
                    phi[0].append(0)

        # dp计算
        for t in range(1, T):
            delta.append([])
            phi.append([])
            # 对于每一个词性j
            for j in range(N):
                if sentenceList[t] in self.B[compoList[j]].keys():
                    # 寻找max
                    # tempList = []
                    maxDel = -float("inf")
                    indexPh = -1
                    for i in range(N):
                        if compoList[j] in self.A[compoList[i]].keys():
                            temp = delta[t - 1][i] + math.log(self.A[compoList[i]][compoList[j]]) + math.log(self.B[compoList[j]][sentenceList[t]])
                            if temp >= maxDel:
                                maxDel = temp
                                indexPh = i
                            # tempList.append(delta[t - 1][i] + math.log(self.A[compoList[i]][compoList[j]]) + math.log(self.B[compoList[j]][sentenceList[t]]))
                        else:
                            temp = delta[t - 1][i] + math.log(1 / (self.composition[compoList[i]]["count"] + 1)) + math.log(self.B[compoList[j]][sentenceList[t]])
                            if temp >= maxDel:
                                maxDel = temp
                                indexPh = i
                            # tempList.append(delta[t - 1][i] + math.log(1 / (self.composition[compoList[i]]["count"] + 1)) + math.log(self.B[compoList[j]][sentenceList[t]]))
                    delta[t].append(maxDel)
                    phi[t].append(indexPh)
                else:
                    # 寻找max
                    # tempList = []
                    maxDel = -float("inf")
                    indexPh = -1
                    for i in range(N):
                        if compoList[j] in self.A[compoList[i]].keys():
                            temp = delta[t - 1][i] + math.log(self.A[compoList[i]][compoList[j]]) + math.log(1 / 900000)
                            if temp >= maxDel:
                                maxDel = temp
                                indexPh = i
                            # tempList.append(delta[t - 1][i] + math.log(self.A[compoList[i]][compoList[j]]) + math.log(1 / 900000))
                        else:
                            temp = delta[t - 1][i] + math.log(1 / (self.composition[compoList[i]]["count"] + 1)) + math.log(1 / 900000)
                            if temp >= maxDel:
                                maxDel = temp
                                indexPh = i
                            # tempList.append(delta[t - 1][i] + math.log(1 / (self.composition[compoList[i]]["count"] + 1)) + math.log(1 / 900000))
                    delta[t].append(maxDel)
                    phi[t].append(indexPh)


        # 寻找最大概率下的最后一个词的索引，并回溯
        backRoad = [-1] * T
        maxP = max(delta[T-1])
        backRoad[T-1] = delta[T-1].index(maxP)
        # 回溯
        for t in range(T-2, -1, -1):
            backRoad[t] = phi[t+1][backRoad[t+1]]
        preCompoList = []
        for index in backRoad:
            preCompoList.append(compoList[index])
        return preCompoList







    def dataProcess(self, rate, seg):
        dataList = []
        with open(self.filename, encoding="UTF-8") as fileGet:
            for line in fileGet:
                dataList.append(line.strip())
        startLoca = int(len(dataList) * seg)
        testSize = int(len(dataList) * rate)
        self.trainSize = len(dataList) - testSize
        self.testList = dataList[startLoca:(startLoca+testSize)]
        self.seg = seg
        trainList = dataList[:startLoca] + dataList[(startLoca+testSize):]

        for line in trainList:
            line = ' '.join(filter(lambda x: x, line.split(" ")))
            wordList = line.split(" ")
            for i in range(len(wordList)):
                # 拆分每个词和词性
                if "/" not in wordList[i]:
                    continue
                (vocab, compo) = self.splitStr(wordList[i])
                # 添加每一行的起始词性
                if i == 0:
                    if compo in self.beginDict.keys():
                        self.beginDict[compo] += 1
                    else:
                        self.beginDict[compo] = 1
                # 计算composition = {} 词性转移矩阵
                if i != (len(wordList) - 1):
                    (vocabNext, compoNext) = self.splitStr(wordList[i+1])
                    if compo not in self.composition.keys():
                        self.composition[compo] = {}
                        self.composition[compo][compoNext] = 1
                        self.composition[compo]["count"] = 1
                    else:
                        if compoNext not in self.composition[compo].keys():
                            self.composition[compo][compoNext] = 1
                            self.composition[compo]["count"] += 1
                        else:
                            self.composition[compo][compoNext] += 1
                            self.composition[compo]["count"] += 1

                # compoVocab = {} 词性到单词的统计
                if compo not in self.compoVocab.keys():
                    self.compoVocab[compo] = {}
                    self.compoVocab[compo][vocab] = 1
                    self.compoVocab[compo]["count"] = 1
                else:
                    if vocab not in self.compoVocab[compo].keys():
                        self.compoVocab[compo][vocab] = 1
                        self.compoVocab[compo]["count"] += 1
                    else:
                        self.compoVocab[compo][vocab] += 1
                        self.compoVocab[compo]["count"] += 1

    def calculate(self):
        # 计算矩阵A，状态转移概率矩阵,A[][]
        for key in self.composition.keys():
            self.A[key] = {}
            for key2 in self.composition[key]:
                if key2 != "count":
                    self.A[key][key2] = (self.composition[key][key2] + 1) / (self.composition[key]["count"] + 1)
        # 计算矩阵B
        for key in self.compoVocab.keys():
            self.B[key] = {}
            for key2 in self.compoVocab[key]:
                if key2 != "count":
                    self.B[key][key2] = (self.compoVocab[key][key2] + 1) / (self.compoVocab[key]["count"] + 1)
        # 计算矩阵pi
        for key in self.beginDict:
            self.pi[key] = (self.beginDict[key] + 1) / (self.trainSize + 1)


    def splitStr(self, string):
        for strI in range(len(string) - 1, -1, -1):
            if string[strI] == "/":
                break
        vocab = string[:strI]
        compo = string[strI + 1:]
        return (vocab, compo)

    def train(self, rate, seg):
        self.dataProcess(rate,seg)
        self.calculate()

    def predict(self, resultPath):
        resultFile = open(resultPath, "a+")

        dataList = []
        with open(self.filename, encoding="UTF-8") as fileGet:
            for line in fileGet:
                dataList.append(line.strip())
        testList = dataList[self.trainSize:]
        sameNum = 0
        unSameNum = 0
        ccc = 0
        for line in testList:
            if ccc < 10:
                ccc += 1
                line = ' '.join(filter(lambda x: x, line.split(" ")))
                wordList = line.split(" ")
                inputList = []
                outputList = []
                for i in range(len(wordList)):
                    # 拆分每个词和词性
                    if "/" not in wordList[i]:
                        continue
                    (vocab, compo) = self.splitStr(wordList[i])
                    inputList.append(vocab)
                    outputList.append(compo)
                preCompoList = self.viterbi(inputList)
                for j in range(len(preCompoList)):
                    if outputList[j] == preCompoList[j]:
                        sameNum += 1
                    else:
                        unSameNum += 1
        accuracy = sameNum / (sameNum + unSameNum)
        resultStr = "seg:" + str(self.seg) +  "###   " + "same: " + str(sameNum) + "   " + "unsame: " + str(unSameNum) + "   " + "accuracy: " + str(accuracy)
        resultFile.write('%s%s' %(resultStr, "\n"))
        resultFile.close()







if __name__ == '__main__':
    startTime = time.clock()

    filename = "./data/raw_data.txt"
    resultPath = "./data/result"
    rate = 0.2
    segList = [0, 0.2, 0.4, 0.6, 0.8]
    for seg in segList:
        print("start:", seg, "round")
        viterbi = Viterbi(filename)
        viterbi.train(rate=rate, seg=seg)
        viterbi.predict(resultPath)

    endTime = time.clock()
    print("Time used: ", (endTime - startTime), "seconds")
