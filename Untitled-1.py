# %%
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
import missingno as msno
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% [markdown]
# ## Lendo e analisando df Geral

# %%
dados = pd.read_excel('WDI_EXCEL/WDIEXCEL.xlsx')

# %%
dados.head()

# %%
dados.columns = [col.strip().replace("_", " ") for col in dados.columns]

# %%
print(dados.columns)


# %%
dados_long = dados.melt(
    id_vars=["Country Name", "Country Code",
             "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="Value"
)

# %%
dados_long["Year"] = dados_long["Year"].astype(int)

# %%
# Lista completa de indicadores categorizados
indicadores = {
    # Demográficos
    "demograficos": [
        "Fertility rate, total (births per woman)",
        "Mortality rate, infant (per 1.000 live births)",
        "Life expectancy at birth, total (years)",
        "Net migration",
        "Birth rate, crude (per 1.000 people)",
        "Population growth (annual %)",
        "Population ages 0-14 (% of total population)",
        "Population ages 15-64 (% of total population)",
        "Population ages 65 and above (% of total population)",
        "Population, total",
    ],

    # Econômicos
    "economicos": [
        "GDP per capita (current US$)",
        "GDP growth (annual %)",
        "Unemployment, total (% of total labor force)",
        "Domestic credit to private sector (% of GDP)",
        "GDP (current US$)",
        "Gross domestic savings (% of GDP)",
        "Labor force, total",
        "Labor force participation rate, total (% of total population ages 15+)",
    ],

    # Urbanização
    "urbanizacao": [
        "Urban population (% of total population)",
        "Urban population growth (annual %)",
        "Population in largest city",
        "Urban land area (% of total land area)",
        "Population density (people per sq. km of land area)",
        "Rural population (% of total population)",
        "Rural population growth (annual %)",
        "Population living in slums (% of urban population)",
        "Population in largest city (% of urban population)",
        "Population in cities with 1 million inhabitants (% of total population)",
    ],

    # Educação
    "educacao": [
        "School enrollment, primary (% net)",
        "Literacy rate, adult total (% of people ages 15 and above)",
        "School enrollment, secondary (% net)",
        "School enrollment, tertiary (% gross)",
        "Pupil-teacher ratio, primary",
        "Pupil-teacher ratio, secondary",
        "Pupil-teacher ratio, tertiary",
        "Education expenditure, total (% of GDP)",
        "Education expenditure, total (% of government expenditure)",
    ],

    # Saúde
    "saude": [
        "Health expenditure, total (% of GDP)",
        "Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)",
        "Mortality rate, under-5 (per 1,000 live births)",
        "Prevalence of HIV, total (% of population ages 15-49)",
        "People using at least basic sanitation services (% of population)",
        "Access to electricity (% of population)",
    ],

    # Pobreza e desigualdade
    "pobreza": [
        "Gini index",
        "Income share held by highest 10%",
        "Income share held by lowest 10%",
    ]
}

# Criar dicionário de DataFrames filtrados
dados_categorizados = {
    categoria: dados_long[dados_long["Indicator Name"].isin(indicadores)]
    for categoria, indicadores in indicadores.items()
}

# DataFrame com todos os indicadores de interesse (equivalente ao dados_filtrados original)
dados_filtrados = dados_long[dados_long["Indicator Name"].isin(
    [item for sublist in indicadores.values() for item in sublist]
)]


# %%
# Processamento por país (mantendo sua estrutura original)
paises_unicos = dados_filtrados["Country Name"].unique()

sub_dfs_por_pais = {
    pais: grupo.drop(
        columns=["Country Code", "Country Name"]).reset_index(drop=True)
    for pais, grupo in dados_filtrados.groupby("Country Name")
}

# Exemplo para o Brasil
brasil = sub_dfs_por_pais["Brazil"]
print("\nDados do Brasil:")
print(brasil.head())

# %% [markdown]
# ## Analise do Brasil

# %% [markdown]
# ### Explorando o DF

# %%
brasil = sub_dfs_por_pais["Brazil"]
brasil.head()

# %%


# %%
# 1. Filtrar apenas as colunas importantes para a pivotagem
df_brasil = brasil[["Year", "Indicator Name", "Value"]].copy()

# 2. Pivotar para ficar no formato "wide" (uma coluna por indicador)
df_brasil_wide = df_brasil.pivot(
    index="Year",
    columns="Indicator Name",
    values="Value"
)

# 3. Ordenar pelas datas (anos) e exibir as primeiras linhas
df_brasil_wide.sort_index(inplace=True)
df_brasil_wide.head()

# %%
df_brasil_wide.describe()


# %%
df_brasil_wide.isna().sum()


# %%
# Preencher valores restantes com a média do período
dados_brasil_filled = df_brasil_wide.fillna(df_brasil_wide.mean())

# %%
msno.matrix(df_brasil_wide)
plt.title('Mapa de Valores Ausentes - Brasil')
plt.show()

# %%
# Análise de correlação entre fatores demográficos e econômicos
indicadores_correlacao = [
    'Population, total',
    'Population growth (annual %)',
    'GDP per capita (current US$)',
    'Urban population (% of total population)',
    'Life expectancy at birth, total (years)',
    'Fertility rate, total (births per woman)',
    "Gini index",
    "Labor force, total"
]

# Filtrar apenas os indicadores selecionados
dados_corr = brasil[brasil['Indicator Name'].isin(indicadores_correlacao)]

# Pivotar para formato amplo
dados_corr_wide = dados_corr.pivot(
    index='Year', columns='Indicator Name', values='Value')

# Calcular correlação
matriz_corr = dados_corr_wide.corr()

# Plotar mapa de calor de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlação entre Indicadores Demográficos e Econômicos')
plt.tight_layout()
plt.show()

# %%

# Preparar os dados
df_plot = corr_target.reset_index()
df_plot.columns = ['Variável', 'Correlação']
df_plot['Correlação'] = df_plot['Correlação'].round(3)
df_plot['Cor_abs'] = df_plot['Correlação'].abs()
df_plot['Tipo'] = df_plot['Correlação'].apply(
    lambda x: 'Positiva' if x > 0 else 'Negativa')

# Criar gráfico
fig = px.bar(df_plot,
             x='Correlação',
             y='Variável',
             color='Correlação',
             color_continuous_scale='RdBu',
             range_color=[-1, 1],
             title='<b>Correlação com Crescimento Populacional (Brasil)</b>',
             text='Correlação',
             hover_data={'Cor_abs': False, 'Tipo': True},
             height=600)

# Ajustes estéticos
fig.update_traces(texttemplate='%{text:.2f}',
                  textposition='outside',
                  marker_line_color='black',
                  marker_line_width=0.5)
fig.update_layout(plot_bgcolor='white',
                  xaxis_range=[-1.1, 1.1],
                  xaxis_title='Coeficiente de Correlação',
                  yaxis_title='Variáveis',
                  coloraxis_colorbar_title='Correlação')
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")

fig.show()

# %%


# Filtrar os dados
crescimento_pop = brasil[brasil['Indicator Name']
                         == 'Population growth (annual %)']
pop_brasil = brasil[brasil['Indicator Name'] == 'Population, total']

# Criar figura com eixo secundário
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Adiciona o trace de crescimento populacional (eixo Y primário)
fig.add_trace(
    go.Scatter(
        x=crescimento_pop['Year'],
        y=crescimento_pop['Value'],
        mode='lines+markers',
        name='Crescimento Anual (%)',
        line=dict(color='blue')
    ),
    secondary_y=False,
)

# Adiciona o trace de população total (eixo Y secundário)
fig.add_trace(
    go.Scatter(
        x=pop_brasil['Year'],
        y=pop_brasil['Value'],
        mode='lines+markers',
        name='População Total',
        line=dict(color='orange')
    ),
    secondary_y=True,
)

# Atualiza o layout e os títulos dos eixos
fig.update_layout(
    title_text="População Total e Crescimento Populacional do Brasil",
    xaxis_title="Ano",
    template='plotly_white',
    hovermode='x unified'
)
fig.update_yaxes(title_text="Crescimento Anual (%)", secondary_y=False)
fig.update_yaxes(title_text="População Total", secondary_y=True)

fig.show()


# %%

# Exemplo: Fertility rate vs. Population growth
plt.figure(figsize=(12, 6))
df_brasil_wide[[
    "Fertility rate, total (births per woman)", "Population growth (annual %)"]].plot()
plt.title("Taxa de Fertilidade vs. Crescimento Populacional (Brasil)")
plt.show()

# %%
# Análise de fatores demográficos que afetam a população
fatores_demograficos = [
    'Fertility rate, total (births per woman)',
    'Mortality rate, infant (per 1.000 live births)',
    'Life expectancy at birth, total (years)',
    'Birth rate, crude (per 1.000 people)'
]

# Normalizando os dados para visualização conjunta
dados_demograficos = brasil[brasil['Indicator Name'].isin(
    fatores_demograficos)].copy()

# Pivot para ter cada indicador como coluna
dados_pivot = dados_demograficos.pivot(
    index='Year', columns='Indicator Name', values='Value')

# Normalização min-max para cada coluna
dados_norm = (dados_pivot - dados_pivot.min()) / \
    (dados_pivot.max() - dados_pivot.min())


# %%
# Plotando dados normalizados
plt.figure(figsize=(14, 7))
sns.lineplot(data=dados_norm)
plt.title('Fatores Demográficos Normalizados (Brasil)')
plt.ylabel('Valor Normalizado')
plt.xlabel('Ano')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(title='Indicador', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Tendência de urbanização vs. crescimento populacional
plt.figure(figsize=(12, 6))
dados_brasil_filled[["Urban population (% of total population)",
                     "Population growth (annual %)"]].rolling(window=5).mean().plot()
plt.title("Urbanização vs. Crescimento Populacional (Média Móvel 5 anos)")
plt.show()

# %%
# Análise da estrutura etária ao longo do tempo
faixas_etarias = [
    'Population ages 0-14 (% of total population)',
    'Population ages 15-64 (% of total population)',
    'Population ages 65 and above (% of total population)'
]

estrutura_etaria = brasil[brasil['Indicator Name'].isin(faixas_etarias)]

plt.figure(figsize=(14, 7))
sns.lineplot(data=estrutura_etaria, x='Year', y='Value', hue='Indicator Name')
plt.title('Estrutura Etária da População Brasileira')
plt.ylabel('Percentual da População Total (%)')
plt.xlabel('Ano')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend(title='Faixa Etária', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
fertilidade = brasil[brasil['Indicator Name'] == 'Fertility rate, total (births per woman)'][[
    'Year', 'Value']].rename(columns={'Value': 'Fertility Rate'})
populacao_total = brasil[brasil['Indicator Name'] == 'Population, total'][[
    'Year', 'Value']].rename(columns={'Value': 'Population Total'})
urbanizacao = brasil[brasil['Indicator Name'] == 'Urban population (% of total population)'][[
    'Year', 'Value']].rename(columns={'Value': 'Urbanization (%)'})


df_plot = populacao_total.merge(
    fertilidade, on='Year').merge(urbanizacao, on='Year')


fig = px.scatter(
    df_plot,
    x='Population Total',
    y='Fertility Rate',
    # Tamanho dos pontos baseado no percentual de urbanização
    size='Urbanization (%)',
    # Cor dos pontos baseado no percentual de urbanização
    color='Urbanization (%)',
    hover_data=['Year'],  # Mostrar o ano ao passar o mouse
    title='Relação entre Fertilidade, População Total e Urbanização (Brasil)',
    labels={'Fertility Rate': 'Taxa de Fertilidade',
            'Population Total': 'População Total', 'Urbanization (%)': 'Urbanização (%)'}
)

fig.show()

# %%
# Para dados anuais, podemos analisar a tendência de crescimento populacional
# Primeiro, vamos garantir que os dados estão na estrutura correta

# Suponha que temos uma série temporal de população por país
# Selecionar dados de população para um país específico
pais = "Brazil"  # ou qualquer outro país de interesse
dados_pop = dados_filtrados[(dados_filtrados["Country Name"] == pais) &
                            (dados_filtrados["Indicator Name"] == "Population, total")]

# Criar uma série temporal adequada
pop_ts = dados_pop.set_index("Year")["Value"]

# Verificar se temos dados suficientes
print(f"Dados disponíveis para {pais}: {len(pop_ts)} anos")

# Visualizar a série temporal básica
plt.figure(figsize=(12, 6))
pop_ts.plot()
plt.title(f"População de {pais} (1960-2023)")
plt.ylabel("População")
plt.grid(True)
plt.show()

# Analisar a tendência usando o filtro Hodrick-Prescott
if len(pop_ts) > 5:  # Verificar se há dados suficientes
    # lambda=100 para dados anuais
    ciclo, tendencia = hpfilter(pop_ts, lamb=100)

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(pop_ts, label="População original")
    plt.plot(tendencia, 'r--', label="Tendência")
    plt.legend()
    plt.title(f"População e Tendência - {pais}")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(ciclo, label="Ciclo (flutuações)")
    plt.title(f"Componente cíclico - {pais}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calcular taxa de crescimento anual
    taxa_crescimento = pop_ts.pct_change() * 100

    plt.figure(figsize=(12, 6))
    taxa_crescimento.plot(kind='bar')
    plt.title(f"Taxa de Crescimento Populacional Anual (%) - {pais}")
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# %%
# Modelagem com múltiplas variáveis socioeconômicas
# Primeiro, reorganizar os dados para ter indicadores como colunas
dados_wide = dados_filtrados.pivot_table(
    index=["Country Name", "Year"],
    columns="Indicator Name",
    values="Value"
).reset_index()

# Filtrar para um país específico
dados_pais = dados_wide[dados_wide["Country Name"] == pais].sort_values("Year")

# Selecionar variáveis para o modelo
variaveis_modelo = [
    "Fertility rate, total (births per woman)",
    "Life expectancy at birth, total (years)",
    "GDP per capita (current US$)",
    "Urban population (% of total population)",
    "Population, total"  # nossa variável alvo
]

# Verificar disponibilidade das variáveis
colunas_disponiveis = [
    col for col in variaveis_modelo if col in dados_pais.columns]
print(f"Variáveis disponíveis: {colunas_disponiveis}")

# Preparar o DataFrame para modelagem
df_modelo = dados_pais[["Year"] + colunas_disponiveis].dropna()

# Se tivermos dados suficientes, criar modelo de regressão
if len(df_modelo) > 10 and "Population, total" in colunas_disponiveis:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # Separar variáveis independentes e dependente
    X = df_modelo.drop(columns=["Year", "Population, total"])
    y = df_modelo["Population, total"]

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar modelo
    modelo = LinearRegression()
    modelo.fit(X_scaled, y)

    # Analisar coeficientes
    coefs = pd.DataFrame({
        'Variável': X.columns,
        'Coeficiente': modelo.coef_
    }).sort_values('Coeficiente', ascending=False)

    print("Importância das variáveis para previsão populacional:")
    print(coefs)

    # Visualizar importância das variáveis
    plt.figure(figsize=(10, 6))
    plt.barh(coefs['Variável'], coefs['Coeficiente'])
    plt.title(f"Impacto das variáveis socioeconômicas na população - {pais}")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### Urbanizão x economia

# %%

# Análise de correlação entre fatores de urbanização e econômicos
indicadores_correlacao = [
    # Econômicos:
    "GDP per capita (current US$)",
    "Unemployment, total (% of total labor force)",
    "GDP (current US$)",
    "Labor force, total",

    # Urbanização:
    "Urban population (% of total population)",
    "Urban population growth (annual %)",
    "Population in largest city",
    "Urban land area (% of total land area)",
    "Population density (people per sq. km of land area)",
    "Rural population (% of total population)"
]

# Filtrar apenas os indicadores selecionados
dados_corr = brasil[brasil['Indicator Name'].isin(indicadores_correlacao)]

# Pivotar para formato amplo
dados_corr_wide = dados_corr.pivot(
    index='Year', columns='Indicator Name', values='Value')

# Calcular correlação
matriz_corr = dados_corr_wide.corr()

# Plotar mapa de calor de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlação entre Indicadores de Urbanização e Econômicos')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparação com outros países

# %%
# Lista de países para comparação
paises_comparacao = ["Brazil", "Russian Federation", "India", "China", "South Africa",  # BRICS
                     "Argentina", "Mexico", "Colombia", "United States", "Canada"  # America
                     "Germany", "France", "Italy", "Spain", "United Kingdom"]  # Europa]

# Filtrar dados apenas para esses países
dados_comparacao = dados_long[dados_long["Country Name"].isin(
    paises_comparacao)]
dados_comparacao = dados_comparacao[dados_comparacao["Indicator Name"].isin(
    indicadores_interesse)]

# Criar subconjuntos para cada país
sub_dfs_paises_comparacao = {pais: grupo for pais,
                             grupo in dados_comparacao.groupby("Country Name")}

# Criar dataframes pivotados para cada país
dfs_wide_por_pais = {}
for pais in sub_dfs_paises_comparacao:
    df_temp = sub_dfs_paises_comparacao[pais][[
        "Year", "Indicator Name", "Value"]].copy()
    dfs_wide_por_pais[pais] = df_temp.pivot(
        index="Year",
        columns="Indicator Name",
        values="Value"
    )
    dfs_wide_por_pais[pais].sort_index(inplace=True)

# %%

# Filtrar os dados de crescimento populacional
dados_crescimento = dados_comparacao[dados_comparacao['Indicator Name']
                                     == 'Population growth (annual %)']
dados_crescimento = dados_crescimento.sort_values(by='Year')

# Definir os ticks do eixo X com espaçamento adequado
anos_min = int(dados_crescimento["Year"].min())
anos_max = int(dados_crescimento["Year"].max())
tick_spacing = 5 if (anos_max - anos_min) > 10 else 1
tick_vals = list(range(anos_min, anos_max + 1, tick_spacing))

# Criar gráfico de linhas com marcadores e cores distintas usando a paleta Set1
fig = px.line(
    dados_crescimento,
    x='Year',
    y='Value',
    color='Country Name',
    markers=True,
    title='Comparação da Taxa de Crescimento Populacional entre Países',
    # Paleta com cores bem diferentes
    color_discrete_sequence=px.colors.qualitative.Set1
)

# Atualizar os ticks do eixo X com rotação de 45° e valores definidos
fig.update_xaxes(
    tickangle=45,
    tickmode='array',
    tickvals=tick_vals,
    title="Ano"
)

# Atualizar o rótulo do eixo Y
fig.update_yaxes(title="Crescimento Anual (%)")

# Atualizar o layout do gráfico para um visual limpo e dimensionamento adequado
fig.update_layout(
    template='plotly_white',
    xaxis_title='Ano',
    yaxis_title='Crescimento Anual (%)',
    legend_title_text='País',
    width=1400,
    height=800,
    margin=dict(l=50, r=150, t=80, b=50)
)

fig.show()


# %%

# Definição dos grupos de países
grupos_paises = {
    "BRICS": ["Brazil", "Russian Federation", "India", "China", "South Africa"],
    "América": ["Brazil", "Argentina", "Mexico", "Colombia", "United States", "Canada"],
    "Europa": ["Germany", "France", "Italy", "Spain", "United Kingdom"]
}

# Filtrar os dados para o indicador "Population growth (annual %)"
dados_crescimento = dados_comparacao[dados_comparacao["Indicator Name"]
                                     == "Population growth (annual %)"]
dados_crescimento = dados_crescimento.sort_values(by="Year")

# Gerar um gráfico para cada grupo de países
for grupo, paises in grupos_paises.items():
    # Filtrar os dados para o grupo atual
    dados_grupo = dados_crescimento[dados_crescimento["Country Name"].isin(
        paises)]

    # Definir os ticks do eixo X com espaçamento adequado
    anos_min = int(dados_grupo["Year"].min())
    anos_max = int(dados_grupo["Year"].max())
    tick_spacing = 5 if (anos_max - anos_min) > 10 else 1
    tick_vals = list(range(anos_min, anos_max + 1, tick_spacing))

    # Criar o gráfico de linhas com marcadores e cores distintas
    fig = px.line(
        dados_grupo,
        x="Year",
        y="Value",
        color="Country Name",
        markers=True,
        title=f"Taxa de Crescimento Populacional: {grupo}",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Atualizar os ticks do eixo X com rotação de 45°
    fig.update_xaxes(
        tickangle=45,
        tickmode="array",
        tickvals=tick_vals,
        title="Ano"
    )

    # Atualizar o rótulo do eixo Y
    fig.update_yaxes(title="Crescimento Anual (%)")

    # Atualizar o layout para um visual limpo e dimensionamento adequado
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Ano",
        yaxis_title="Crescimento Anual (%)",
        legend_title_text="País",
        width=1400,
        height=800,
        margin=dict(l=50, r=150, t=80, b=50)
    )

    fig.show()


# %%
plt.figure(figsize=(12, 6))
sns.kdeplot(data=dados_comparacao[dados_comparacao['Indicator Name'] == 'Population ages 15-64 (% of total population)'],
            x='Value', hue='Country Name', fill=True)
plt.title('Distribuição da População Adulta (15-64 anos) por País')

# %%
pivot_table = dados_comparacao.pivot_table(
    index='Country Name', columns='Indicator Name', values='Value')
sns.heatmap(pivot_table[[
            'Population ages 15-64 (% of total population)']], annot=True, cmap='YlOrRd')
plt.title('Percentual de População Adulta por País')
