
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import metrics


#leitura do arquivo csv extraído do kaggle
df = pd.read_csv('Dados Comunidade (respostas) - dados.csv')
df.head()


#definição dos valores únicos da coluna
df['Posição da cadeira (senioridade)'].sort_values().unique()


#substituição de Sim e Não por 0 e 1
df = df.replace({"Sim":1, "Não":0})
df.head()

#definição da variáveis numéricas
num_vars = ['Curte games?',
            'Curte futebol?',
            'Curte livros?',
            'Curte jogos de tabuleiro?',
            'Curte jogos de fórmula 1?',
            'Curte jogos de MMA?',
            'Idade'
]

#definição das variáveis dummies
dummy_vars = [
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]


#criação das variáveis dummies
df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz?'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise


#criação de features e target
features = df_analise.columns[:-1].tolist()
X = df_analise[features]
y = df_analise['pessoa feliz?']

#ajuste de modelo de árvore
arvore = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
arvore.fit(X, y)

#ajuste de modelo naive
naive = naive_bayes.GaussianNB()
naive.fit(X, y)

#ajuste de modelo de regressão logística
reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(X, y)

#criação da predição e probabilidade para árvore, naive e regressão
df_predict = df_analise[['pessoa feliz?']].copy()

df_predict['predict_arvore'] = arvore.predict(X)
df_predict['proba_arvore'] = arvore.predict_proba(X)[:,1]

df_predict['predict_naive'] = naive.predict(X)
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]

df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]

#salvamento da predição em csv
df_predict.to_csv('predict.csv', sep=';', index=False)

#métricas de acurácia, precisão, recall e curva roc para a árvore
acc_arvore = metrics.accuracy_score(df_predict['pessoa feliz?'], df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(df_predict['pessoa feliz?'], df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(df_predict['pessoa feliz?'], df_predict['predict_arvore'])
roc_arvore = metrics.roc_curve(df_predict['pessoa feliz?'],df_predict['proba_arvore'])
auc_arvore = metrics.roc_auc_score(df_predict['pessoa feliz?'],df_predict['proba_arvore'])
auc_arvore

#métricas de acurácia, precisão, recall e curva roc para naive bayes
acc_naive= metrics.accuracy_score(df_predict['pessoa feliz?'], df_predict['predict_naive'])
precisao_naive = metrics.precision_score(df_predict['pessoa feliz?'], df_predict['predict_naive'])
recall_naive = metrics.recall_score(df_predict['pessoa feliz?'], df_predict['predict_naive'])
roc_naive = metrics.roc_curve(df_predict['pessoa feliz?'],df_predict['proba_naive'])
auc_naive = metrics.roc_auc_score(df_predict['pessoa feliz?'],df_predict['proba_naive'])
auc_naive

#métricas de acurácia, precisão, recall e curva roc para regressão logística
acc_reg = metrics.accuracy_score(df_predict['pessoa feliz?'], df_predict['predict_reg'])
precisao_reg = metrics.precision_score(df_predict['pessoa feliz?'], df_predict['predict_reg'])
recall_reg = metrics.recall_score(df_predict['pessoa feliz?'], df_predict['predict_reg'])
roc_reg = metrics.roc_curve(df_predict['pessoa feliz?'],df_predict['proba_reg'])
auc_reg = metrics.roc_auc_score(df_predict['pessoa feliz?'],df_predict['proba_reg'])
auc_reg

#plot do gráfico com os três modelos
plt.figure(dpi=400)

plt.plot(roc_arvore[0], roc_arvore[1], 'o-')
plt.plot(roc_naive[0], roc_naive[1], 'o-')
plt.plot(roc_reg[0], roc_reg[1], 'o-')
plt.grid(True)
plt.title('ROC curve')
plt.xlabel('1 - Especificidade')
plt.ylabel('Recall')

plt.legend([f'Árvore: {auc_arvore:.2f}', f'Naive: {auc_naive:.2f}', f'Reg: {auc_reg:.2f}'])

#salvamento do modelo de regressão em .pkl
pd.Series({'model': reg, 'features': features}).to_pickle('model_feliz.pkl')
