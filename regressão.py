#%%

import pandas as pd
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt


df = pd.read_excel("dados_cerveja_nota.xlsx")
df.head()

# %%

X = df[['cerveja']]   # X é uma matriz (DF)
y = df['nota']        # y é um vetor (series)

#isso é o ML
reg = linear_model.LinearRegression()
reg.fit(X, y)    # ajuste do modelo

#esses são os coeficientes
a, b = reg.intercept_, reg.coef_[0]
print(a, b)

#exclusão de duplicatas
predict_reg = reg.predict(X.drop_duplicates())


arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y) 
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y) 
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%

#plot do gráfico
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação Cerveja vs Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full, color='green')
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2, color='magenta')

plt.legend(['Observado', 
            f'y = {a:.3f} + {b:.3f} x',
            'Árvore Full',
            'Árvore Depth = 2'
            ])

# %%

#plot da árvore
plt.figure(dpi=400)

tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)

# %%
