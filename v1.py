import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import os
import subprocess

# Verificando se o arquivo clean_data.json existe
if not os.path.exists('./output/clean_data.json'):
    print("Arquivo clean_data.json não encontrado. Executando o script clear_data.py...")
    subprocess.run(['python3', 'clear_data.py'], check=True)

# 1. Carregar dados limpos
df = pd.read_json('./output/clean_data.json')

# 2. Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('bairro', OneHotEncoder(), ['bairro']),  # OneHot para bairros
        ('num', 'passthrough', ['area', 'esquina'])  # Mantém numéricas
    ])

# 3. Definir variáveis
X = df[['bairro', 'area', 'esquina']]
y = df['valor']

# 4. Criar e treinar modelo
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X, y)

# 5. Métricas de desempenho
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nMétricas do Modelo:")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.2f}")

# 6. Teste F (significância estatística)
X_processed = model.named_steps['preprocessor'].transform(X)
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

f_stats, p_values = f_regression(X_processed, y)
resultados_f = pd.DataFrame({
    'Variável': feature_names,
    'F': f_stats,
    'p-valor': p_values
}).sort_values('F', ascending=False)

print("\nTeste F - Significância Estatística:")
print(resultados_f)

# 7. Coeficientes do modelo
regressor = model.named_steps['regressor']
coef_df = pd.DataFrame({
    'Variável': feature_names,
    'Coeficiente': regressor.coef_
}).sort_values('Coeficiente', ascending=False)

print("\nCoeficientes do Modelo:")
print(coef_df)
print(f"\nIntercepto: {regressor.intercept_:.2f}")

# 8. Exemplo de previsão
novo_terreno = pd.DataFrame({
    'bairro': ['AUSTRILIO CAPILE DE CASTRO'],
    'area': [400],
    'esquina': [0]
})
valor_previsto = model.predict(novo_terreno)
print(f"\nPrevisão para terreno : R${valor_previsto[0]:,.2f}")