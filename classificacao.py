

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

#leitura do arquivo excel
df = pd.read_excel('dados_cerveja_nota.xlsx')


#notas maiores que 5 terão valor 1, as menores terão valor 0
df['aprovado'] = (df['nota'] > 5).astype(int)


#gráfico que mostra quantidade de cervejas vs aprovação
# plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
# plt.grid(True)
# plt.title('Cerveja vs Aprovação')
# plt.xlabel('Cervejas')
# plt.ylabel('Aprovados')


#criação, predição e proba do modelo de regressão logística, excluindo duplicatas
reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#criação, predição e proba do modelo de árvore de decisão, excluindo duplicatas
arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#criação, predição e proba do modelo de árvore de decisão, com profundidade 2, excluindo duplicatas
arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#criação, predição e proba do modelo de naive bayes, excluindo duplicatas
nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]


#plot do gráfico considerando todos os modelos
plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovados')
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='red')
plt.plot(df['cerveja'].drop_duplicates(), reg_proba, color='green')

#plt.plot(df['cerveja'].drop_duplicates(), arvore_full_predict, color='magenta')
#plt.plot(df['cerveja'].drop_duplicates(), arvore_full_proba, color='blue')

#plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict, color='yellow')
#plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_proba, color='black')

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='magenta')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color='blue')

plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')

plt.legend(['Observaação', 
            'Reg Predict', 
            'Reg Proba',
            #'Árvore Predict',
            #'Árvore Proba',
            #'Árvore D2 Predict',
            #'Árvore D2 Proba',
            'NB Predict',
            'NB Proba'])

