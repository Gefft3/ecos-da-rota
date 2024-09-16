#!/bin/bash

url_train='../datasets/relevantes/EIOS_train.csv'
url_test='../datasets/relevantes/EIOS_test.csv'

#Manter o padrão 'Relevantes' e 'Irrelevantes'
tipo='Relevantes'
# tipo='Irrelevantes'

k_max=100

python3 script.py $url_train $url_test $k_max $tipo

echo "Execução finalizada!"
