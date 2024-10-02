import wandb 
import sys
import os
import numpy as np

run = wandb.init(
    project='ECOS da Rota'
    )

def calculate_acurracy(classificacoes, tipo):
    acurracy = 0

    correto = 'Relevante' if tipo == 'Relevantes' else 'Irrelevante'

    for classificacao in classificacoes:
        if classificacao == correto:
            acurracy += 1
    return acurracy/len(classificacoes)

tipo = sys.argv[1]

path_raiz = f'../Experimentos/Logs {tipo}'

for pastas in os.listdir(path_raiz):
    if 'k =' in pastas:
        
        distance_mean = []
        acurracy_mean = []
        
        path_outputs = os.path.join(path_raiz, pastas)
        for arquivos in os.listdir(path_outputs):
            if 'distancias' in arquivos:
                with open(os.path.join(path_outputs, arquivos), "r") as f:
                    distancias = f.readlines()
                    distancias = [float(x.split()[1]) for x in distancias]
                    distance_mean.append(np.mean(distancias))

            if 'classificacoes' in arquivos:
                with open(os.path.join(path_outputs, arquivos), "r") as f:
                    classificacoes = f.readlines()
                    classificacoes = [x.split()[1] for x in classificacoes]
                    acurracy_mean.append(calculate_acurracy(classificacoes, tipo))

            
        if len(distance_mean) == 0:
            distance_mean = [0]
        print(f'K: {pastas}')
        print(f'Distancia média: {distance_mean}')
        print(f'Acurácia média: {acurracy_mean}')

        wandb.log({
                "distance_mean": np.mean(distance_mean),
                "acurracy_mean": np.mean(acurracy_mean)
            })