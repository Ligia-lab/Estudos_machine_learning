# %%

import pandas as pd

df = pd.read_csv('Dados Comunidade (respostas) - dados.csv')
df.head()


# %%

df = df.replace({"Sim":1, "Não":0})
df.head()

# %%

num_vars = ['Curte games?',
            'Curte futebol?',
            'Curte livros?',
            'Curte jogos de tabuleiro?',
            'Curte jogos de fórmula 1?',
            'Curte jogos de MMA?',
            'Idade'
]

dummy_vars = [
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz?'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise


# %%

features = df_analise.columns[:-1].tolist()
X = df_analise[features]
y = df_analise['pessoa feliz?']

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
arvore.fit(X, y)

# %%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['pessoa feliz?']]
df_predict['predict_arvore'] = arvore_predict
df_predict

# %%

#Acurácia
(df_predict['pessoa feliz?'] == df_predict['predict_arvore']).mean()


# %%

pd.crosstab(df_predict['pessoa feliz?'], df_predict['predict_arvore'])

# %%
