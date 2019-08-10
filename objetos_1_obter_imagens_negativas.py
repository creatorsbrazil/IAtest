# ETAPA 1 - Geração das Imagens Negativas em Preto e Branco!
# Autoria inicial: http://www.instructables.com/id/Haar-Cascade-Python-OpenCV-Treinando-E-Detectando-/
# Leia as explicações direto com o autor original, aqui é apenas uma livre adaptação para estudo

import urllib.request
import cv2
import os

# a pasta 'temp' é um local temporária que será usado para o aprendizado de objetos
tempPath = 'temp'

if not os.path.exists(tempPath):
    os.makedirs(tempPath)

pathPB = tempPath+'/negativas'
if not os.path.exists(pathPB):
    os.makedirs(pathPB)

originalPath = tempPath+'/negativas/originais'
if not os.path.exists(originalPath):
    os.makedirs(originalPath)

# este é um banco de imagem publico que será usado para criar imagens negativas
url_imagens = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152' # pessoas
# url_imagens = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01503061' # passaros
# url_imagens = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222' # plantas
# url_imagens = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03365991' # construções
# url_imagens = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388' # Animais

imagens_negativas = urllib.request.urlopen(url_imagens).read().decode().splitlines()
# imagens_negativas = [] # para gerar apenas as imagen Preto e Branca

# Logico que por questão de performance e otimização, só serão baixada uma vez as imagens
# Todo treino acontece sempre com as imagem em preto e branco.
# Essas imagens poderão ser usadas para treino de qualquer objeto.
# Caso ocorra qualquer erro ao baixar alguma imagem, mas não tem problema.

numero_imagem = 2000
# image_size = (400, 400)
for imgurl in imagens_negativas:
    n = str(numero_imagem)
    numero_imagem += 1
    original = originalPath + '/' + n + '.jpg'
    if not os.path.isfile(original):
        print(n + ': Baixando: ' + imgurl)
        try:
            a, b = urllib.request.urlretrieve(imgurl, original)

        except Exception as ex:
            print(str(ex))


numero_imagem = 1
for original in os.listdir(originalPath):
    try:
        n = str(numero_imagem)
        numero_imagem += 1
        imagemPB = pathPB + '/' + n + '.jpg'
        if not os.path.isfile(imagemPB):
            img2 = cv2.imread(originalPath + '/' + original, cv2.IMREAD_GRAYSCALE)
            height, width = img2.shape
            if height > 300 and width > 300:
                print('Gerando negativa: ' + imagemPB)
                # img2 = cv2.resize(img, image_size)
                cv2.imwrite(imagemPB, img2)
            else:
                os.remove(original)

    except Exception as ex:
        print(str(ex))

        
print('Imagens negativas baixadas, e criadas em preto e branco')

# lista um TXT com as imagens a ser usada nas etapa 2
bg = 'bg.txt'
if os.path.isfile(bg):
    os.remove(bg)

with open(bg, 'a') as f:
    for img in os.listdir(pathPB):
        line = pathPB + '/'+img+'\n'
        f.write(line)

print('Lista criada negativa em (bg.txt)')

print('FIM da etapa 1!')
