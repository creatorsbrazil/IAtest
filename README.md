# IAtest
Testes de IA Phyton

## ContarPessoasFoto.py

Contador de faces em uma foto testando parametros do OpenCV 

![Contador de Pessoas](https://raw.githubusercontent.com/creatorsbrazil/IAtest/master/doc/contar-pressoas.png)

## Aprendizado de Objetos

Os exemplos de aprendizado de objetos está dividido em 3 arquivos:

1) objetos_1_obter_imagens_negativas.py
* Baixa de um site algumas imagens diversas, e converte para preto e branco para otimizar todo o processo
* Também cria o arquivo bg.txt com a lista dessas imagens

2) objetos_2_treino_amostras.py
* Para cada imagem positiva é criada um grupo de amostras baseada nas imagens negativa
* Apos tudo ter sido criado é feito o treino
   
3) objetos_3_testar.py
* O processo se conclui se tudo funcionar, então use a camera para testar

Veja mais

![DOC 2.4 Open CV Traincascade](https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html)
![DOC 3.3 Open CV Traincascade](https://docs.opencv.org/3.3.0/dc/d88/tutorial_traincascade.html)
![Exemplo 1](http://note.sonots.com/SciSoftware/haartraining.html)
![Exemplo 2](https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)
![Exemplo 3](https://codeyarns.com/2014/09/01/how-to-train-opencv-cascade-classifier)
