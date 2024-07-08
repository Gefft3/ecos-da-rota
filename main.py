import pandas as pd

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
        'SG_UF_INTE',
        'CO_MUN_RES',
        'ID_UNIDADE',
        'CO_UNI_NOT',
        'COD_IDADE',
        'CS_ESCOL_N',
        'ID_MN_RESI',
        'CO_MUN_RES',
        'NM_BAIRRO',
        'CS_ZONA',
        #Vacina covid
        'LOTE_1_COV',
        'LOTE_2_COV',
        'LOTE_REF',
        'LOT_RE_BI',
        'SURTO_SG',
        #Filtrando colunas sobre as doenças/exames
        'REINF'
    }

    df = df.drop(columns=mapa_deletar)
    return df

def proporcao_pacientes_por_sexo(df):
    pass

def doenca_mais_frequente(df):
    pass

def media_idade_pacientes(df):
    pass

def media_idade_pacientes_por_doenca(df):
    pass

def media_idade_pacientes_por_doenca_e_sexo(df):
    pass

if __name__ == '__main__':

    #pre processamento dos dados
    df_sivep = ler_planilha()
    df_sivep = remover_colunas_vazias(df_sivep)
    df_sivep = remover_futeis(df_sivep)
    
    #calcular proporção de pacientes por sexo

    #calcular doenças mais frequentes 

    #calcular media de idade dos pacientes

    #calcular media de idade dos pacientes por doença

    #calcular media de idade dos pacientes por doença e por sexo