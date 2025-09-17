#%%
#SEMMA

#import da base
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

#separação dos dados de treino excluindo os dados OOT
df_train = df[df['dtRef']<df['dtRef'].max()].copy()

# %%

#definição de features(variáveis) e target
df_train.head()

features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

#%%

#SAMPLE

#separação das bases de treino e teste (X e y devem ter a mesma quantidade de linhas) e taxa da variável resposta
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    random_state=42, 
                                                                    test_size=0.2,
                                                                    stratify=y)

print('Taxa variável resposta Treino:', y_train.mean())
print('Taxa variável resposta Teste:', y_test.mean())

# %%

#EXPLORE 

#verificação de missing
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

#criação da árvore de decisão para descobrir as principais features
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# %%

#definição das features para serem utilizadas
feature_importances = (pd.Series(arvore.feature_importances_, index=X_train.columns)
                                .sort_values(ascending=False)
                                .reset_index())
feature_importances['acum'] = feature_importances[0].cumsum()
feature_importances[feature_importances['acum'] > 0.96]

 # %%

#criação da variável só com as melhores features
best_features = (feature_importances[feature_importances['acum'] > 0.96]['index'].to_list())

best_features

# %%

#MODIFY

from feature_engine import discretisation

tree_discretization = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3)

tree_discretization.fit(X_train, y_train)

X_train_transform = tree_discretization.transform(X_train)
X_train_transform

# %%

#MODEL

from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
reg.fit(X_train_transform, y_train)

# %%

#ASSESS

from sklearn import metrics


#TREINO
y_train_predict = reg.predict(X_train_transform)
y_train_proba = reg.predict_proba(X_train_transform)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print('Acurácia treino: ', acc_train)
print('AUC treino: ', auc_train)


#TESTE
X_test_transform = tree_discretization.transform(X_test)

y_test_predict = reg.predict(X_test_transform)
y_test_proba = reg.predict_proba(X_test_transform)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print('Acurácia test: ', acc_test)
print('AUC test: ', auc_test)


#OOT
oot_transform = tree_discretization.transform(oot[features])

y_oot_predict = reg.predict(oot_transform)
y_oot_proba = reg.predict_proba(oot_transform)[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

print('Acurácia oot: ', acc_oot)
print('AUC oot: ', auc_oot)


# %%
