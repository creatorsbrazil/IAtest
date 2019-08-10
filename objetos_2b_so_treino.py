# ETAPA 2b - Somente a etapa do treino sem a geração das amostras

import cv2
import os

os.environ["PATH"] += ";C:\\CreatorsBrazil\\IAtest\\utils"
tempPath = 'temp'
vecPath = tempPath + '/vec'
dataPath = tempPath + '/data'
fotoPath = os.path.abspath('images/torre')
amostraSize = (30, 40)
positivaMaxSize = max(amostraSize) * 5
num = 200

print('Unificando vetores...\n')
cmd = ('py utils/mergevec.py -v ' + vecPath +
       ' -o ' + vecPath + '/final.vec\n')

print(cmd)
os.system(cmd)

print('Treinando... isso ira demorar! va tomar um cafe ou varios...\n')
cmd = ('opencv_traincascade -data '+dataPath+' -vec '+vecPath+'/final.vec' +
       ' -bg bg.txt -numPos '+str(num)+' -numNeg '+str(num)+' -numStages 10 ' +
       ' -stageType BOOST -featureType HAAR' +  # ADABOOST / LBP
       ' -numThreads 8 -precalcValBufSize 2048 -precalcIdxBufSize 2048' +
       ' -w ' + str(amostraSize[0])+' -h ' + str(amostraSize[1]) + '\n')
print(cmd)
os.system(cmd)

print("FIM!")
