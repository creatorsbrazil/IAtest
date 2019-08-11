# ETAPA 4 - Processa todas as imagens da pasta com o programa da etapa 2
import os

fotoPath = os.path.abspath('images')
for imageName in os.listdir(fotoPath):

    # pula arquivos (pastas geralmente não tem extensão)
    if(imageName.find('.') > 0):
        continue

    print('======================================================')
    print('Processando '+imageName+'...\n')
    
    cmd = ('py objetos_2_amostra_e_treino_varias.py -i '+imageName+ ' -n 400 -s 30 -t LBP')
    os.system(cmd)

    # cmd = ('py objetos_2_amostra_e_treino_varias.py -i '+imageName+ ' -n 400 -s 30 -t HAAR')
    # os.system(cmd)

print("FIM Total!")
