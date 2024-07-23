import pandas as pd

# Declarando funções e filtros

def ler_planilha():
    path_planilha = 'SIVEP_2024.xlsx'
    df = pd.read_excel(path_planilha,engine='openpyxl')
    return df

def remover_colunas_vazias(df):
    df = df.dropna(axis=1, how='all')
    return df

def remover_futeis(df):
    mapa_deletar = {
        #Filtrando colunas sobre os dados do paciente
        'SG_UF_INTE', #APENAS MS 
        'CO_MUN_RES', #CÓDIGO IBGE
        'ID_UNIDADE', #CÓDIGO CNES
        'CO_UNI_NOT', #CÓDIGO CNES
        'COD_IDADE',  #CÓDIGO IDADE
        'CS_ESCOL_N', #ESCOLARIDADE
        'ID_MN_RESI', #CÓDIGO MUNICÍPIO
        'CO_MUN_RES', #CÓDIGO MUNICÍPIO
        'NM_BAIRRO',  #NOME DO BAIRRO PADRONIZADO CORREIOS
        'CS_ZONA',    #ZONA (RURAL OU URBANA)  TUDO 1 
        'PAC_DSCBO',  #OCUPAÇÃO PROFISSIONAL
        #Vacina covid
        'LOTE_1_COV', #CÓDIGO LOTE 1 COVID 
        'LOTE_2_COV', #CÓDIGO LOTE 2 COVID
        'LOTE_REF',   #LOTE 1 DE REFORÇO
        'LOTE_REF2',  #LOTE 2 DE REFORÇO
        'LOT_RE_BI',  #LOTE REFORÇO BIVALENTE
        'FNT_IN_COV', #GERADO PELO SISTEMA
        #Filtrando colunas sobre as doenças/exames
        'REINF' #REINFECÇÃO (TUDO NÃO)
    }

    df = df.drop(columns=mapa_deletar)
    return df

def proporcao_pacientes_por_sexo(df):
    df_sexo = df['CS_SEXO'].value_counts(normalize=True)
    return df_sexo 

def sintomas_mais_frequente(df):
    mapa_sintomas = {
        'NOSOCOMIAL',
        'AVE_SUINO',
        'FEBRE',
        'TOSSE',
        'GARGANTA',
        'DISPNEIA',
        'DESC_RESP',
        'SATURACAO',
        'DIARREIA',
        'VOMITO',
        'OUTRO_SIN'
    }

    df_sintomas = df[list(mapa_sintomas)]
    
    contagem = df_sintomas.applymap(lambda x: 1 if (x == 1 or x == 2 or x == 9) else 0).sum()
    
    df_contagem = pd.DataFrame(contagem, columns=['frequencia']).reset_index()
    df_contagem.columns = ['sintoma', 'frequencia']
    
    return df_contagem

def doenca_mais_frequente(df):
    df_doencas = df['CLASSI_FIN'].value_counts(normalize=True)  
    return df_doencas 

def fator_risco(df):
    df_risco = df['FATOR_RISC'].value_counts(normalize=True)
    return df_risco

def fatores_risco(df):
    mapa_fatores = {
        'PUERPERA',
        'CARDIOPATI',
        'HEMATOLOGI',
        'SIND_DOWN',
        'HEPATICA',
        'ASMA',
        'DIABETES',
        'NEUROLOGIC',
        'PNEUMOPATI',
        'IMUNODEPRE',
        'RENAL',
        'OBESIDADE',
    }
    df_fatores = df[list(mapa_fatores)]

    contagem = df_fatores.applymap(lambda x: 1 if x == 1 else 0).sum()

    df_contagem = pd.DataFrame(contagem, columns=['frequencia']).reset_index()
    df_contagem.columns = ['fator', 'frequencia']
    return df_contagem

def tomou_vacina(df):

    tomou = df[df['VACINA'] == 1]
    nao_tomou = df[df['VACINA'] == 2]
    ignorado = df[df['VACINA'] == 9]
    null = df[df['VACINA'].isnull()]
    
    df_vacina = pd.DataFrame({
        'tomou': [tomou.shape[0]],
        'nao_tomou': [nao_tomou.shape[0]],
        'ignorado': [ignorado.shape[0]],
        'null': [null.shape[0]]
    })

    return df_vacina

def media_idade_pacientes(df):
    idades = df[df['TP_IDADE'] == 3]
    media_idades = df['NU_IDADE_N'].mean() 
    dp_idades = df['NU_IDADE_N'].std()
    return idades, media_idades, dp_idades

def frequencia_valores(df):
    result_list = []
    
    for column in df.columns:
        value_counts = df[column].value_counts(dropna=False).items()
        
        value_counts = [(str(value) if pd.notna(value) else 'nulos', count) for value, count in value_counts]
        
        if len(value_counts) <= 4:
            result_list.append((column, value_counts))

    return result_list
