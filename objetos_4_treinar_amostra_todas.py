# ETAPA 4 - Processa todas as imagens da pasta com o programa da etapa 2
import os

fotoPath = os.path.abspath('images')
for num in (20, 200, 400, 800, 1600):
    for imageName in os.listdir(fotoPath):

        # pula arquivos (pastas geralmente não tem extensão)
        if(not os.path.isdir('images\\'+imageName)):
            continue

        print('======================================================')
        print('Processando '+imageName+'...\n')
        
        cmd = ('py objetos_2_amostra_e_treino_varias.py -i '+imageName+ ' -n '+str(num)+' -s 30 -t LBP')
        os.system(cmd)

        cmd = ('py objetos_2_amostra_e_treino_varias.py -i '+imageName+ ' -n '+str(num)+' -s 30 -t HAAR')
        os.system(cmd)

print("FIM Total!")
