#!/bin/bash

#Manter o padrão 'Relevantes' e 'Irrelevantes'
tipo='Relevantes'
# tipo='Irrelevantes'

#URL dos datasets de treino e teste
url_train='../datasets/relevantes/EIOS_train.csv'
url_test='../datasets/relevantes/EIOS_test_filtrado.csv'
#url_test='../datasets/irrelevantes/_GPT_test.csv'

#Tamanho máximo do k (número de documentos retornados)
k_max=30

#Tamanho máximo de tokens por prompt
max_prompt_tokens=50000

python3 script.py $url_train $url_test $k_max $tipo $max_prompt_tokens

echo "Execução finalizada!" 