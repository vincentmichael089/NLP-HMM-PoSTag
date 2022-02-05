import pandas as pd
import swifter
import numpy as np
import sys
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action = "ignore", category = SettingWithCopyWarning)
warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

class HMM:
  def __init__(self):
    self._data = None
    self._emissionProb = None
    self._transitionProb = None
    self._sentencePos = None
    self._sentenceSentencePosPos = None
    self._posList = None
    self._vocabList = None

    self._predictData = None
    self._sentenceSentencePosPosPredict = None
    self._sentencePredict = None
    self._sentenceTag = None
    self._predictionList = []
    self._mg = None

  def initiateMatrix(self):
    print("Initiating transition probability")
    self._transitionProb = pd.DataFrame(
        data = np.zeros((len(self._posList)-1, len(self._posList)-1), dtype = float), 
        index = list(set(self._posList) - set(['</S>'])),
        columns = list(set(self._posList) - set(['<S>']))
      )
    
    for i in range(len(self._sentenceSentencePosPos)):
      for j in range(len(self._sentenceSentencePosPos[i][1])):
        if j != 0: 
          context = self._sentenceSentencePosPos[i][1][j]
          given = self._sentenceSentencePosPos[i][1][j-1]
          self._transitionProb[context].loc[given] += 1

    self._transitionProb = self._transitionProb.div(self._transitionProb.sum(axis=1), axis=0)
    print("Done!")

    print("Initiating emission probability")
    self._emissionProb = pd.DataFrame(
        data = np.zeros((len(self._posList), len(self._vocabList)), dtype = float),
        index = self._posList,
        columns = self._vocabList
    )

    for i in range(len(self._sentencePos)):
      for j in range(len(self._sentencePos[i])):
        vocab = self._sentencePos[i][j][0]
        pos = self._sentencePos[i][j][1] 
        self._emissionProb[vocab.lower()].loc[pos] += 1

    self._emissionProb = self._emissionProb.div(self._emissionProb.sum(axis=1), axis=0)
    print("Done!")

  def smoothMatrix(self):
    self._transitionProb = self._transitionProb.replace(0, 0.0001607717041800643/2)

  def countProb(self, prevTag, currentTag, word):
    return self._emissionProb[word][currentTag] * self._transitionProb[currentTag][prevTag]

  def tokenizeAndTranspose(self, sentence):
    tokenized = [token.split("_") for token in sentence.split()]
    transposed = [list(i) for i in zip(*tokenized)]
    return transposed
  
  def tokenize(self, sentence):
    tokenized = [token.split("_") for token in sentence.split()]
    return tokenized

  def startEndMarker(self, sentence):
    return str("<s>_<S> ") + sentence + str(" </s>_</S>")

  def countPosTagAndVocab(self):
    temp = []
    for i in range(len(self._sentenceSentencePosPos)):
      temp += self._sentenceSentencePosPos[i][1]

    posList = dict((x, temp.count(x)) for x in set(temp))  
    self._posList = list(posList.keys())

    temp = []
    for i in range(len(self._sentenceSentencePosPos)):
      temp += self._sentenceSentencePosPos[i][0]

    vocabList = dict((x.lower(), temp.count(x.lower())) for x in set(temp))  
    self._vocabList = [vocab.lower() for vocab in list(vocabList.keys())]
    
  def fit(self, data):
    self._data = data.copy().swifter.apply(self.startEndMarker)
    self._sentencePos = [self.tokenize(sentence) for sentence in self._data["text"].values]
    self._sentenceSentencePosPos = [self.tokenizeAndTranspose(sentence) for sentence in self._data["text"].values] 

    self.countPosTagAndVocab()
    self.initiateMatrix()
    self.smoothMatrix()

  def accuracy(self):
    correctCount = 0
    totalCount = 0
    for i in range(len(self._predictionList)):
      prediction = self._predictionList[i][1:-1]
      actual = self._sentenceSentencePosPosPredict[i][1][1:-1]
      
      for j in range(len(prediction)):
        totalCount += 1
        if prediction[j] == actual[j]: correctCount += 1

    return correctCount / totalCount

  def predict(self, data, dataFrame = True, printStep = False, getResult = False):
    print("preprocessing data")
    self._predictionList = []

    if dataFrame:
      self._predictData = data.copy().swifter.apply(self.startEndMarker)
      self._sentenceSentencePosPosPredict = [self.tokenizeAndTranspose(sentence) for sentence in self._predictData["text"].values]
    else:
      rebuildSentence = ""
      preprocessedData = "<s> " + data + " </s>" 
      preprocessedData = preprocessedData.split()
      preprocessedData = [pdat + "_MASK" for pdat in preprocessedData]

      for i in preprocessedData:
        rebuildSentence = rebuildSentence + i + " "

      rebuildSentence.strip() 
      self._sentenceSentencePosPosPredict = [self.tokenizeAndTranspose(sentence) for sentence in [rebuildSentence]]

    self._sentencePredict = [sentence[0] for sentence in self._sentenceSentencePosPosPredict]
    self._sentenceTag = [sentence[1] for sentence in self._sentenceSentencePosPosPredict]

    
    
    for i in range(len(self._sentencePredict)):
      for j in range(len(self._sentencePredict[i])):
        if self._sentencePredict[i][j].lower() not in self._emissionProb.columns:
          self._emissionProb[self._sentencePredict[i][j].lower()] = 1
          self._emissionProb[self._sentencePredict[i][j].lower()]['<S>'] = 0
          self._emissionProb[self._sentencePredict[i][j].lower()]['</S>'] = 0
    print("Done!")

    print("Processing HMM")
    for i in range(len(self._sentencePredict)):
      listPath = []
      listProb = []

      for j in range(len(self._sentencePredict[i])):
        tempProb = []
        tempPath = []

        if j == 0:
          listPath.append(['<S>'])
          listProb.append(1)
        else:  
          prevKeyword = self._sentencePredict[i][j-1].lower()
          currentKeyword = self._sentencePredict[i][j].lower()
          prevTag = [prev[-1] for prev in listPath] 
          currentTag = self._emissionProb[currentKeyword][self._emissionProb[currentKeyword] > 0].index.to_list()

          for k in range(len(currentTag)):
            bestProb = 0
            bestTag = None
            
            for l in range(len(prevTag)):
              if self.countProb(prevTag[l], currentTag[k], currentKeyword) * listProb[l] > bestProb:
                bestProb = self.countProb(prevTag[l], currentTag[k], currentKeyword) * listProb[l]
                bestTag = [prevTag[l], currentTag[k]] 
              if l == len(prevTag)-1:
                for path in listPath:
                  if path[-1] == prevTag[l]:
                    tempPath.append(path[:-1] + bestTag)
                    tempProb.append(bestProb)

          listPath = tempPath
          listProb = tempProb
          if printStep is True: print(tempPath, tempProb)

      self._predictionList.append(listPath[0])

    print("Done!")
    if dataFrame: print("Accuracy: ", self.accuracy())
    if getResult is True: return self.getPrediction(dataFrame)

  def getPrediction(self, dataFrame):
    predictionHolder = []

    for i in range(len(self._predictionList)):
      sentenceHolder = ""
      prediction = self._predictionList[i][1:-1]
      word = self._sentenceSentencePosPosPredict[i][0][1:-1]

      for j in range(len(prediction)):
        sentenceHolder = sentenceHolder + str(word[j]) +"_"+ str(prediction[j]) + " "

      predictionHolder.append(sentenceHolder.strip())
    
    if dataFrame: 
      return pd.DataFrame(predictionHolder, columns =['text'])
    else:
      return predictionHolder[0]
