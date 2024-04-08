from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from matplotlib import pyplot as plt
from graphviz import Source
import pandas as pd
import numpy as np

# Відкриття файлу та зчитування даних
data = pd.read_csv('dataset_2.txt', sep=',', header=None)
data.columns = ['ID', 'Date', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Target']
print("Файл було успішно зчитано програмою\n")

# Визначення та виведення кількості записів та полів у кожному записі
rows,columns = data.shape
print(f"Кількість записів у наборі даних: {rows}")
print(f"Кількість полів у кожному записі: {columns}")

# Виведення перших 10 записів
print("\nПерші 10 записів набору даних:")
print(data.head(10))

# Розділення набору даних на навчальну та тестову вибірки (66% - навчальна, 33% - тестова)
train_data, test_data = train_test_split(data, test_size=0.33, random_state=17)
print("\nРозмір навчальної вибірки:", len(train_data))
print("Розмір тестової вибірки:", len(test_data))

# Розділення вхідних ознак та цільової змінної
x_train = train_data.iloc[:, 2:-1]
y_train = train_data.iloc[:, -1]

x_test = test_data.iloc[:, 2:-1]
y_test = test_data.iloc[:, -1]

# Побудова моделі дерева прийняття рішень з глибиною 5
entropy_tree = DecisionTreeClassifier(criterion = "entropy",max_depth=5)
entropy_tree.fit(x_train, y_train)

gini_tree = DecisionTreeClassifier(criterion = "gini",max_depth=5)
gini_tree.fit(x_train, y_train)
print("\nЗбудовано класифікаційну модель дерева прийняття рішень ")

# Графічне відображення дерева
def save_decision_tree(model, filename):
    dot_data = export_graphviz(model, out_file=None, 
                               feature_names=x_train.columns,
                               class_names=["0", "1"],
                               filled=True, rounded=True, special_characters=True)
    graph = Source(dot_data)
    graph.render(filename, format="png", cleanup=True)
    print(f"Граф було створено окремо у файлі: {filename}.png")

save_decision_tree(entropy_tree, "Entropy_Decision_tree")
save_decision_tree(gini_tree, "Gini_Decision_tree")

# Обчислення класифікаційних метрик для тренувальної та тестової вибірки
def calculate_metrics(model, x_cord, y_cord):
    pred = model.predict(x_cord)
    accuracy = metrics.accuracy_score(y_cord, pred)
    precision = metrics.precision_score(y_cord, pred)
    recall = metrics.recall_score(y_cord, pred)
    f1_score = metrics.f1_score(y_cord, pred)
    mcc = metrics.matthews_corrcoef(y_cord, pred)
    ba = metrics.balanced_accuracy_score(y_cord, pred)
    return [accuracy, precision, recall, f1_score, mcc, ba]

metrics_test_gini = calculate_metrics(gini_tree, x_test, y_test)
metrics_train_gini = calculate_metrics(gini_tree, x_train, y_train)

metrics_test_entropy = calculate_metrics(entropy_tree, x_test, y_test)
metrics_train_entropy = calculate_metrics(entropy_tree, x_train, y_train)

# Побудова графіку тренувальної вибірки на основі ентропії чи неоднорідності Джині 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
mlabels = ['accuracy', 'precision', 'recall', 'f_scores', 'MCC', 'BA']
bar_width = 0.35

# Виведення стовпчикових діаграм для тренувальної вибірки
axes[0].bar(np.arange(len(mlabels)), metrics_train_entropy, bar_width, label="Тренувальна вибірка entropy", color='blue')
axes[0].bar(np.arange(len(mlabels)) + bar_width, metrics_train_gini, bar_width, label="Тренувальна вибірка gini", color='orange')
axes[0].set_ylim((0.95, 1))
axes[0].set_ylabel('Значення')
axes[0].set_xticks(np.arange(len(mlabels)) + bar_width / 2)
axes[0].set_xticklabels(mlabels, rotation=45)
# axes[0].grid(True)
axes[0].legend()

# Виведення стовпчикових діаграм для тестової вибірки
axes[1].bar(np.arange(len(mlabels)), metrics_test_entropy, bar_width, label="Тестова вибірка entropy", color='green')
axes[1].bar(np.arange(len(mlabels)) + bar_width, metrics_test_gini, bar_width, label="Тестова вибірка gini", color='red')
axes[1].set_ylim((0.95, 1))
axes[1].set_ylabel('Значення')
axes[1].set_xticks(np.arange(len(mlabels)) + bar_width / 2)
axes[1].set_xticklabels(mlabels, rotation=45)
# axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('Combined_metrics.png')
plt.show()

# Вплив максимальної кількості листів та мінімальної кількості елементів в листі дерева
max_leaf_nodes_range = range(2, 20, 1) 
min_samples_leaf_range = range(1, 50, 2)

accuracy_results_max_leaf = []
precision_results_max_leaf = []
recall_results_max_leaf = []
f1_score_results_max_leaf = []
mcc_results_max_leaf = []
ba_results_max_leaf = []

accuracy_results_min_samples = []
precision_results_min_samples = []
recall_results_min_samples = []
f1_score_results_min_samples = []
mcc_results_min_samples = []
ba_results_min_samples = []

for max_leaf_nodes in max_leaf_nodes_range:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=5, max_leaf_nodes=max_leaf_nodes, random_state=17)
    tree.fit(x_train, y_train)
    test_metrics = calculate_metrics(tree, x_test, y_test)
    
    accuracy_results_max_leaf.append(test_metrics[0])
    precision_results_max_leaf.append(test_metrics[1])
    recall_results_max_leaf.append(test_metrics[2])
    f1_score_results_max_leaf.append(test_metrics[3])
    mcc_results_max_leaf.append(test_metrics[4])
    ba_results_max_leaf.append(test_metrics[5])

for min_samples_leaf in min_samples_leaf_range:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=min_samples_leaf, random_state=17)
    tree.fit(x_train, y_train)
    test_metrics = calculate_metrics(tree, x_test, y_test)
    
    accuracy_results_min_samples.append(test_metrics[0])
    precision_results_min_samples.append(test_metrics[1])
    recall_results_min_samples.append(test_metrics[2])
    f1_score_results_min_samples.append(test_metrics[3])
    mcc_results_min_samples.append(test_metrics[4])
    ba_results_min_samples.append(test_metrics[5])

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(max_leaf_nodes_range, accuracy_results_max_leaf, label="Accuracy")
plt.plot(max_leaf_nodes_range, precision_results_max_leaf, label="Precision")
plt.plot(max_leaf_nodes_range, recall_results_max_leaf, label="Recall")
plt.plot(max_leaf_nodes_range, f1_score_results_max_leaf, label="F1-score")
plt.plot(max_leaf_nodes_range, mcc_results_max_leaf, label="MCC")
plt.plot(max_leaf_nodes_range, ba_results_max_leaf, label="Balanced Accuracy")
plt.xlabel("Максимальна кількість листів")
plt.ylabel("Значення метрики")
plt.title("Вплив максимальної кількості листів")
plt.legend()
# plt.grid(True)
plt.xticks(max_leaf_nodes_range)

plt.subplot(1, 2, 2)
plt.plot(min_samples_leaf_range, accuracy_results_min_samples, label="Accuracy")
plt.plot(min_samples_leaf_range, precision_results_min_samples, label="Precision")
plt.plot(min_samples_leaf_range, recall_results_min_samples, label="Recall")
plt.plot(min_samples_leaf_range, f1_score_results_min_samples, label="F1 Score")
plt.plot(min_samples_leaf_range, mcc_results_min_samples, label="MCC")
plt.plot(min_samples_leaf_range, ba_results_min_samples, label="Balanced Accuracy")
plt.xlabel("Мінімальна кількість елементів в листі")
plt.ylabel("Значення метрики")
plt.title("Вплив мінімальної кількості елементів в листі")
plt.legend()
# plt.grid(True)
plt.xticks(min_samples_leaf_range)

plt.tight_layout()
plt.show()

# Важливість атрбутів для класифікації
feature_importance1 = entropy_tree.feature_importances_
feature_importance2 = gini_tree.feature_importances_
sorted_idx1 = np.argsort(feature_importance1)
sorted_idx2 = np.argsort(feature_importance2)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(range(len(feature_importance1)), feature_importance1[sorted_idx1], align="center")
axes[0].set_ylabel("Важливість атрибутів")
axes[0].set_xlabel("Атрибути")
axes[0].set_title("Важливість атрибутів (Entropy Tree)")
# axes[0].grid(True)
axes[0].set_ylim(0.0, 1.0)

axes[1].bar(range(len(feature_importance2)), feature_importance2[sorted_idx2], align="center")
axes[1].set_ylabel("Важливість атрибутів")
axes[1].set_xlabel("Атрибути")
axes[1].set_title("Важливість атрибутів (Gini Tree)")
# axes[0].grid(True)
axes[1].set_ylim(0.0, 1.0)

fig.tight_layout()
plt.savefig("Feature_importances.png")
plt.show()
