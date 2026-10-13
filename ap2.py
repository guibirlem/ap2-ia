import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=== ETAPA 1: Carregando e preparando o dataset ===")

path = kagglehub.dataset_download("adaoduque/campeonato-brasileiro-de-futebol")
csv_path = os.path.join(path, "campeonato-brasileiro-full.csv")
df = pd.read_csv(csv_path)

df = df[['mandante', 'visitante', 'mandante_Placar', 'visitante_Placar']].dropna()

def definir_resultado(linha):
    if linha['mandante_Placar'] > linha['visitante_Placar']:
        return 'Vitoria_Mandante'
    elif linha['mandante_Placar'] < linha['visitante_Placar']:
        return 'Vitoria_Visitante'
    else:
        return 'Empate'
df['resultado'] = df.apply(definir_resultado, axis=1)
df['resultado'] = df.apply(definir_resultado, axis=1)

le = LabelEncoder()
todos_os_times = pd.concat([df['mandante'], df['visitante']]).unique()
le.fit(todos_os_times)

df['mandante_id'] = le.transform(df['mandante'])
df['visitante_id'] = le.transform(df['visitante'])

X = df[['mandante_id', 'visitante_id']]
y = df['resultado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Total de amostras: {len(df)}")
print(f"Amostras de Treino: {len(X_train)} | Amostras de Teste: {len(X_test)}\n")

print("=== ETAPA 2: Treinando Modelo 1 (Árvore de Decisão) ===")
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"Acurácia da Árvore de Decisão: {acc_tree * 100:.2f}%\n")

print("=== ETAPA 3: Treinando Modelo 2 (K-Nearest Neighbors - KNN) ===")
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Acurácia do KNN: {acc_knn * 100:.2f}%\n")

print("--------------------------------------------------")
print("RELATÓRIO DE CLASSIFICAÇÃO: Árvore de Decisão")
print("--------------------------------------------------")
print(classification_report(y_test, y_pred_tree))

print("--------------------------------------------------")
print("RELATÓRIO DE CLASSIFICAÇÃO: KNN")
print("--------------------------------------------------")
print(classification_report(y_test, y_pred_knn))

cv_scores_tree = cross_val_score(tree_model, X, y, cv=5)
cv_scores_knn = cross_val_score(knn_model, scaler.fit_transform(X), y, cv=5)

print("--------------------------------------------------")
print("VALIDAÇÃO CRUZADA (5 Folds)")
print("--------------------------------------------------")
print(f"Árvore de Decisão - Média: {cv_scores_tree.mean() * 100:.2f}% (Desvio: ±{cv_scores_tree.std() * 100:.2f}%)")
print(f"KNN               - Média: {cv_scores_knn.mean() * 100:.2f}% (Desvio: ±{cv_scores_knn.std() * 100:.2f}%)\n")

print("Gerando gráficos comparativos das Matrizes de Confusão...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

classes = tree_model.classes_

cm_tree = confusion_matrix(y_test, y_pred_tree, labels=classes)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=classes)
disp_tree.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title("Matriz de Confusão - Árvore de Decisão")
axes[0].tick_params(axis='x', rotation=45)

cm_knn = confusion_matrix(y_test, y_pred_knn, labels=classes)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=classes)
disp_knn.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title("Matriz de Confusão - KNN")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()