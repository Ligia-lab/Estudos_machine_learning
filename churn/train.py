#%%
#SEMMA

#read do arquivo csv
import pandas as pd

df = pd.read_csv('../data/abt_churn.csv')
df.head()

# %%

#exploração da col dtRef
df['dtRef'].sort_values().unique()
df['dtRef'].value_counts().sort_index()

# %%

#criação da var Out Of Time (que separa uma amostra dos dados mais recentes para validação)
oot = df[df['dtRef']==df['dtRef'].max()].copy()

# %%

#separação dos dados excluindo os dados OOT
df_train = df[df['dtRef']<df['dtRef'].max()].copy()

# %%

#separação de features(variáveis) e target
df_train.head()

features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

#%%

#SAMPLE

#separação das bases de treino e teste (X e y devem ter a mesma quantidade de linhas)
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    random_state=42, 
                                                                    test_size=0.2,
                                                                    stratify=y)

print('Taxa variável resposta Treino:', y_train.mean())
print('Taxa variável resposta Teste:', y_test.mean())

# %%

#EXPLORE 

X_train.isna().sum().sort_values(ascending=False)

# %%

df_analise = X_train.copy()
df_analise[target] = y_train
summario = df_analise.groupby(by=target).agg(['mean', 'median']).T
summario
summario['diff_abs'] = summario[0] - summario[1]
summario['diff_rel'] = summario[0] / summario[1]
summario.sort_values(by=['diff_rel'], ascending=False)

# %%

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# %%

feature_importances = (pd.Series(arvore.feature_importances_, index=X_train.columns)
                                .sort_values(ascending=False)
                                .reset_index())
feature_importances['acum'] = feature_importances[0].cumsum()
feature_importances[feature_importances[0] > 0.01]

 # %%


