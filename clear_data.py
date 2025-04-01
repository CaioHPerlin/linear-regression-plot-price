import pandas as pd
import re
import json

# Carrega os dados originais
json_df = pd.read_json('./input/data.json')

# Função para extrair área (versão corrigida)
def extrair_area(descricao):
    if not isinstance(descricao, str):
        return None
    
    padroes = [
        r"(?:TAMANHO|DIMENS[ÕO]ES)\s*[:]?\s*(\d+[,.]?\d*)\s*[xX×]\s*(\d+[,.]?\d*)\s*(?:m|M|\u0026sup2;|²)",
        r"Área\s*(?:do\s*Terreno)?\s*[:]?\s*(\d+[,.]?\d*)\s*(?:m|M|\u0026sup2;|²)",
        r"MEDINDO\s*(\d+[,.]?\d*)\s*(?:m|M|\u0026sup2;|²)",
        r"(?:^|\D)(\d+[,.]?\d*)\s*(?:m²|m2|M2)(?:\D|$)"
    ]
    
    for padrao in padroes:
        match = re.search(padrao, descricao, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                return float(match.group(1).replace(',', '.')) * float(match.group(2).replace(',', '.'))
            return float(match.group(1).replace(',', '.'))
    return None

# Calcula a coluna 'Area'
json_df['area'] = json_df['descricaoImovel'].apply(extrair_area)

# Verifica se 'descricaoImovel' contém "esquina" (case insensitive)
json_df['esquina'] = json_df['descricaoImovel'].str.contains(
    r'esquina',  # Padrão regex (sem diferenciar maiúsculas/minúsculas)
    case=False,  # Ignora maiúsculas/minúsculas
    regex=True   # Usa regex para busca
).astype(int)    # Converte True/False para 1/0

# Seleciona as colunas desejadas para exportação
colunas_exportar = [
    'id',                   # Para referência
    'bairro',               # Bairro do terreno
    'endereco',             # Endereço completo
    'area',                 # Área calculada
    'valor',                # Valor do terreno
    'esquina',              # Se é esquina
    'descricaoImovel'       # Descrição original (para validação)
]

# Filtra o DataFrame
dados_exportar = json_df[colunas_exportar]

# Remove linhas onde 'Area' é NaN (opcional)
# dados_exportar = dados_exportar.dropna(subset=['area'])

# Exporta para JSON (formato organizado)
dados_exportar.to_json('./output/clean_data.json', orient='records', indent=2, force_ascii=False)

# Exporta apenas os IDs como array simples (ex: [865, 866])
ids_area_null = json_df[json_df['area'].isnull()]['id'].tolist()

with open('./output/null_area.json', 'w', encoding='utf-8') as f:
    json.dump(ids_area_null, f, indent=2, ensure_ascii=False)


print("✅ Dados exportados para 'clean_data.json'!")