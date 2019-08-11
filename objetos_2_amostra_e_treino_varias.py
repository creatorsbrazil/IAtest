# ETAPA 2 - Geração das informações vetoriais de aprendizado baseada nas imagens positivas e negativas
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
fotoPath = os.path.abspath('images/queen')
numero_imagem = 0
amostraSize = (20, 30)
positivaMaxSize = max(amostraSize) * 5
num = 400

# gera as imagens em cinza redimencionando de acordo com a necessidade no tamanho especificado
for imageName in os.listdir(fotoPath):

    if(not imageName.lower().endswith('.jpg') and not imageName.lower().endswith('.png')):
        continue

     # apaga tudo sempre ao iniciar um novo tratamento de arquivo
    for info in os.listdir(infoPath):
        os.remove(infoPath + '/' + info)

    numero_imagem += 1
    n = str(numero_imagem)
    img = cv2.imread(fotoPath + '/' + imageName, cv2.IMREAD_GRAYSCALE)

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
        print(n + ': ' + imageName + ' (' + str(height) + 'x' +
              str(width) + ' => ' + str(w) + 'x' + str(h) + ')\n')
        img = cv2.resize(img, (w, h))

    arquivo = positivaPath + '/' + n + '.jpg'
    cv2.imwrite(arquivo, img)

    print(n + ': Criando amostras...\n')
    cmd = ('opencv_createsamples -img ' + arquivo +
           ' -bg bg.txt ' +
           ' -info ' + infoPath + '/info.lst ' +
           ' -bgcolor 255 -bgthresh 160 -pngoutput info ' +
           ' -maxxangle 0.3 -maxyangle 0.3 -maxzangle 0.3 -num '+str(num)+'\n')
    print(cmd)
    os.system(cmd)

    print(n + ': Criando vetores...\n')
    cmd = ('opencv_createsamples -info ' + infoPath + '/info.lst' +
           ' -num '+str(num)+' -w ' + str(amostraSize[0]) + ' -h ' + str(amostraSize[1]) +
           ' -vec ' + vecPath + '/positives' + n + '.vec\n')
    print(cmd)
    os.system(cmd)

# FIM for!

print('Unificando vetores...\n')
cmd = ('py utils/mergevec.py -v ' + vecPath +
       ' -o ' + vecPath + '/final.vec\n')
print(cmd)
os.system(cmd)

print('Treinando... isso ira demorar! va tomar um cafe ou varios...\n')
cmd = ('opencv_traincascade -data '+dataPath+' -vec '+vecPath+'/final.vec' +
       ' -bg bg.txt -numPos '+str(num)+' -numNeg '+str(num)+' -numStages 20 ' +
       ' -stageType BOOST -featureType LBP'+ # ADABOOST / LBP
       ' -numThreads 8 -precalcValBufSize 2048 -precalcIdxBufSize 2048' +
       ' -w ' + str(amostraSize[0])+' -h ' + str(amostraSize[1]) + '\n')
print(cmd)
os.system(cmd)

print("FIM!")
