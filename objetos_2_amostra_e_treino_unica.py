
# ETAPA 2 - Treino de imagem unica
# http://www.instructables.com/id/Haar-Cascade-Python-OpenCV-Treinando-E-Detectando-/
# http://note.sonots.com/SciSoftware/haartraining.html

# Coloque as imagens na pasta dataset/fotos e execute esse script!
# Apos executar esse programa execute o run.bat!
import cv2
import os

# a pasta 'temp' é um local temporária que será usado para o aprendizado de objetos
os.environ["PATH"] += ";C:\\CreatorsBrazil\\IAtest\\utils"
tempPath = 'temp'

# Diretório de trabalho das amostras positivas em preto e branco
positivaPath = tempPath + '/positiva'
if not os.path.exists(positivaPath):
    os.makedirs(positivaPath)
else:
    # apaga tudo
    for positiva in os.listdir(positivaPath):
        os.remove(positivaPath + '/' + positiva)


# Diretório de trabalho das amostras: adiciona a informação nas negativas
infoPath = tempPath + '/info'
if not os.path.exists(infoPath):
    os.makedirs(infoPath)
else:
  # apaga tudo sempre ao iniciar um novo tratamento de arquivo
    for info in os.listdir(infoPath):
        os.remove(infoPath + '/' + info)


# Diretório para os vetores (conhecimento inicial)
vecPath = tempPath + '/vec'
if not os.path.exists(vecPath):
    os.makedirs(vecPath)
else:
    # apaga tudo
    for vec in os.listdir(vecPath):
        os.remove(vecPath + '/' + vec)


# Diretório para o treinameto
dataPath = tempPath + '/data'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)
else:
    # apaga tudo
    for data in os.listdir(dataPath):
        os.remove(dataPath + '/' + data)


# origem do conhecimento (sempre imagens com fundo branco absoluto 255)
numero_imagem = 0
amostraSize = (20, 30)
positivaMaxSize = max(amostraSize) * 5

# gera as imagens em cinza redimencionando de acordo com a necessidade no tamanho especificado
imageName = 'images/torre/torre-2.jpg'

img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

height, width = img.shape
if height > width and height > positivaMaxSize:
    h = positivaMaxSize
    w = int(h * width / height)
elif width > positivaMaxSize:
    w = positivaMaxSize
    h = int(w * height / width)
else:
    w = width
    h = height

if w != width or h != height:
    print(imageName + ' (' + str(height) + 'x' +
          str(width) + ' => ' + str(w) + 'x' + str(h) + ')\n')
    img = cv2.resize(img, (w, h))

arquivo = positivaPath + '/1.png'
cv2.imwrite(arquivo, img)

print('Criando amostras...\n')
cmd = ('opencv_createsamples -img ' + arquivo +
       ' -bg bg.txt ' +
       ' -info ' + infoPath + '/info.lst ' +
       ' -bgcolor 255 -bgthresh 127 -pngoutput info ' +
       ' -maxxangle 0.3 -maxyangle 0.3 -maxzangle 0.3 -num 2000\n')
print(cmd)
os.system(cmd)

print('Criando vetores...\n')
cmd = ('opencv_createsamples -info ' + infoPath + '/info.lst' +
       ' -num 2000 -w ' + str(amostraSize[0]) + ' -h ' + str(amostraSize[1]) +
       ' -vec ' + vecPath + '/positives1.vec\n')
print(cmd)
os.system(cmd)

print('Treinando... isso ira demorar! va tomar um cafe ou varios...\n')
cmd = ('opencv_traincascade -data '+dataPath +
       ' -vec '+vecPath+'/positives1.vec' +
       ' -bg bg.txt -numPos 1000 -numNeg 1000 -numStages 3 ' +
       ' -stageType BOOST -featureType HAAR' +  # ADABOOST / LBP
       ' -numThreads 8 -precalcValBufSize 2048 -precalcIdxBufSize 2048' +
       ' -w ' + str(amostraSize[0])+' -h ' + str(amostraSize[1]) + '\n')

print(cmd)
os.system(cmd)

print("FIM!")
