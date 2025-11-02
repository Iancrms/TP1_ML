import pandas as pd

arquivo = "WebsitePhishing1k.csv"

df = pd.read_csv(arquivo)
df_filtrado = df[df['Result'] != 0]

df_filtrado.to_csv("WebsitePhishing1k_filtrado.csv", index=False)
