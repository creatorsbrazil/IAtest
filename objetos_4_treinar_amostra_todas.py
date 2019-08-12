# ETAPA 4 - Processa todas as imagens da pasta com o programa da etapa 2
import os

# fotoPath = os.path.abspath('images')
for num in (100, 200, 400, 800, 1600):
    for imageName in os.listdir('images'):
    # for imageName in ['rook']:

        # pula arquivos (pastas geralmente não tem extensão)
        if(not os.path.isdir('images\\'+imageName)):
            continue

        print('======================================================')
        print('Processando '+imageName+'...\n')
        
        # Criar apenas o LBP, pois é mais rapido e depois pode-se criar o HAAR que é mais preciso!
        # O arquivo .VEC é salvo separadamente e é a unica coisa necessária para o treino
        cmd = ('py objetos_2_amostra_e_treino_varias.py -i '+imageName+ ' -n '+str(num)+' -s 40 -t HAAR')
        os.system(cmd)

print("FIM Total!")
