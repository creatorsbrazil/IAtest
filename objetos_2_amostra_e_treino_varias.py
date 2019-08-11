# ETAPA 2 - Geração das informações vetoriais de aprendizado baseada nas imagens positivas e negativas
# http://www.instructables.com/id/Haar-Cascade-Python-OpenCV-Treinando-E-Detectando-/
# http://note.sonots.com/SciSoftware/haartraining.html

# Coloque as imagens na pasta dataset/fotos e execute esse script!

import cv2
import os
import argparse

tempPath = 'temp'
os.environ["PATH"] += ";C:\\CreatorsBrazil\\IAtest\\utils"


def amostra(imagePath: str, num: str, amostraMax: int, featureType: str):

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

    amostraSize = (0, 0)
    positivaMaxSize = 0
    finalCascade: str = ''

    # origem do conhecimento (sempre imagens com fundo branco absoluto 255)
    # gera as imagens em cinza redimencionando de acordo com a necessidade no tamanho especificado
    numero_imagem = 0
    for imageName in os.listdir('images/'+imagePath):

        if(not imageName.lower().endswith('.jpg') and not imageName.lower().endswith('.png')):
            continue

        # apaga tudo sempre ao iniciar um novo tratamento de arquivo
        for info in os.listdir(infoPath):
            os.remove(infoPath + '/' + info)

        numero_imagem += 1
        n = str(numero_imagem)
        img = cv2.imread('images/' + imagePath + '/' +
                         imageName, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape

        if positivaMaxSize == 0:
            # A primeira imagem define o aspecto
            if height > width:
                amostraSize = (int(amostraMax * width / height), amostraMax)
            elif width > positivaMaxSize:
                amostraSize = (amostraMax, int(amostraMax * height / width))

            positivaMaxSize = max(amostraSize) * 5

            finalCascade = ('haarcascade\\' +
                            imagePath+'_'+featureType+'_'+num +
                            '_'+str(amostraSize[0])+'x'+str(amostraSize[1])+'.xml')

            if os.path.isfile(finalCascade):
                print('Arquivo final "'+finalCascade+'" já existe')
                return
            else:
                print('Preparando final "'+finalCascade+'"...')

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
                  str(width) + ' => ' + str(w) + 'x' + str(h) + ')')
            img = cv2.resize(img, (w, h))

        arquivo = positivaPath + '/' + n + '.jpg'
        cv2.imwrite(arquivo, img)

        print(n + ': Criando amostras...\n')
        cmd = ('opencv_createsamples -img ' + arquivo +
               ' -bg bg.txt ' +
               ' -info ' + infoPath + '/info.lst ' +
               ' -bgcolor 255 -bgthresh 32 -pngoutput info ' +
               ' -maxxangle 0.3 -maxyangle 0.3 -maxzangle 0.3 -num '+num)
        # print(cmd)
        os.system(cmd)

        print(n + ': Criando vetores...\n')
        cmd = ('opencv_createsamples -info ' + infoPath + '/info.lst' +
               ' -num '+num+' -w ' + str(amostraSize[0]) + ' -h ' + str(amostraSize[1]) +
               ' -vec ' + vecPath + '/positives' + n + '.vec')
        # print(cmd)
        os.system(cmd)

    print('Unificando vetores...\n')
    cmd = ('py utils/mergevec.py -v ' + vecPath +
           ' -o ' + vecPath + '/final.vec')
    # print(cmd)
    os.system(cmd)

    print('Treinando '+imagePath + '... isso ira demorar!\n')
    cmd = ('opencv_traincascade -data '+dataPath+' -vec '+vecPath+'/final.vec' +
           ' -bg bg.txt -numPos '+num+' -numNeg '+num+' -numStages 20 ' +
           ' -stageType BOOST -featureType ' + featureType +
           ' -numThreads 8 -precalcValBufSize 2048 -precalcIdxBufSize 2048' +
           ' -w ' + str(amostraSize[0])+' -h ' + str(amostraSize[1]) + '\n')

    print(cmd)
    os.system(cmd)

    cmd = ('copy '+dataPath + '\\cascade.xml ' + finalCascade)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='imageFolder')
    parser.add_argument('-n', dest='sampleNum')
    parser.add_argument('-s', dest='sampleSize')
    parser.add_argument('-t', dest='featureType')
    args = parser.parse_args()

    if not args.imageFolder:
        sys.exit('especifique a pasta de imagens: -i apple')

    elif not args.sampleNum:
        sys.exit('especifique o numero de amostras -n 200')

    elif not args.sampleSize:
        sys.exit('informe o tamaho maximo da amostras: -s 20 ')

    elif not args.featureType:
        sys.exit('informe o tipo de algoritimo: -t HAAR/LBP ')

    amostra(args.imageFolder, args.sampleNum,
            int(args.sampleSize), args.featureType)
