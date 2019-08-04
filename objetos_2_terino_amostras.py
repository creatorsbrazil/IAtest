# ETAPA 2 - Geração das informações vetoriais de aprendizado baseada nas imagens positivas e negativas
# http://www.instructables.com/id/Haar-Cascade-Python-OpenCV-Treinando-E-Detectando-/
# http://note.sonots.com/SciSoftware/haartraining.html

# Coloque as imagens na pasta dataset/fotos e execute esse script!
# Apos executar esse programa execute o run.bat!
import cv2
import os
import subprocess

# a pasta 'temp' é um local temporária que será usado para o aprendizado de objetos
basePath = 'temp'


# Diretório de trabalho das amostras positivas em preto e branco
positivaPath = basePath + '/positiva'
if not os.path.exists(positivaPath):
    os.makedirs(positivaPath)
else:
    # apaga tudo
    for positiva in os.listdir(positivaPath):
        os.remove(positivaPath + '/' + positiva)


# Diretório de trabalho das amostras: adiciona a informação nas negativas
infoPath = basePath + '/info'
if not os.path.exists(infoPath):
    os.makedirs(infoPath)
else:
  # apaga tudo sempre ao iniciar um novo tratamento de arquivo
    for info in os.listdir(infoPath):
        os.remove(infoPath + '/' + info)


# Diretório para os vetores (conhecimento inicial)
vecPath = basePath + '/vec'
if not os.path.exists(vecPath):
    os.makedirs(vecPath)
else:
    # apaga tudo
    for vec in os.listdir(vecPath):
        os.remove(vecPath + '/' + vec)


# Diretório para o treinameto
dataPath = basePath + '/data'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)
else:
    # apaga tudo
    for data in os.listdir(dataPath):
        os.remove(dataPath + '/' + data)


# origem do conhecimento (sempre imagens com fundo branco absoluto 255)
fotoPath = 'images/apple'
numero_imagem = 0
positivaSize = 30
amostraSize = 60

fileBat = basePath + "/train.bat"
if os.path.isfile(fileBat):
    os.remove(fileBat)

with open(fileBat, 'a') as bat:

    # gera as imagens em cinza redimencionando de acordo com a necessidade no tamanho especificado
    for imageName in os.listdir(fotoPath):

        numero_imagem += 1
        n = str(numero_imagem)
        img = cv2.imread(fotoPath + '/' + imageName, cv2.IMREAD_GRAYSCALE)
        # 4160x2340 Eventualmente precisa ser redimencionada
        #img2 = img[100:2100,700:2700]

        height, width = img.shape
        if height > width:
            h = positivaSize
            w = int(h * width / height)
        else:
            w = positivaSize
            h = int(w * height / width)

        print(n + ': ' + imageName + ' (' + str(height) + 'x' +
              str(width) + ' => ' + str(w) + 'x' + str(h) + ')\n')

        img = cv2.resize(img, (w, h))
        arquivo = positivaPath + '/' + n + '.jpg'
        cv2.imwrite(arquivo, img)

        #print(n + ': Criando amostras...\n')
        cmd = ('utils\\opencv_createsamples.exe -img ' + arquivo +
               ' -bg ' + basePath + '/bg.txt ' +
               ' -info ' + infoPath + '/info.lst ' +
               ' -bgcolor 255 -bgthresh 3 -pngoutput info ' +
               ' -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 2000\n')
        bat.write(cmd)

        #print(n + ': Criando vetores...\n')
        cmd = ('utils\\opencv_createsamples -info ' + infoPath + '/info.lst' +
               ' -num 2000 -w ' + str(amostraSize) + ' -h ' + str(amostraSize) +
               ' -vec ' + vecPath + '/positives' + n + '.vec\n\n')
        bat.write(cmd)

    # FIM for!

    bat.write('echo Unificando vetores...\n')
    cmd = ('py utils/mergevec.py -v ' + vecPath + ' -o ' + vecPath + '/final.vec\n\n')
    bat.write(cmd)

    bat.write('echo Treinando, isso irá demorar, va tomar um cafe ou varios, e volte depois...\n')
    cmd = ('cd ' + basePath + '\n'
           '..\\utils\\opencv_traincascade -data data -vec vec/final.vec' +
           ' -bg bg.txt -numPos 2000 -numNeg 1000 -numStages 5 ' +
           ' -w ' + str(amostraSize)+' -h ' + str(amostraSize) + '\n\n')
    bat.write(cmd)

input("FIM! Execute o arquivo: " + fileBat)
