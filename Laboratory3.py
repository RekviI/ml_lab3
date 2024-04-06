from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from matplotlib import pyplot as plt
from graphviz import Source
import pandas as pd


data = pd.read_csv('dataset_2.txt', sep=',', header=None)
data.columns = ['ID', 'Date', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Target']
print("Файл було успішно зчитано програмою\n")

rows,columns = data.shape
print(f"Кількість записів у наборі даних: {rows}")
print(f"Кількість полів у кожному записі: {columns}\n")

print("Перші 10 записів набору даних:")
print(data.head(10))


train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
print(f"\nРозмір навчальної вибірки: {len(train_data)}")
print(f"Розмір тестової вибірки: {len(test_data)}\n")


x_train = train_data.iloc[:, 2:-1]
y_train = train_data.iloc[:, -1]

model = DecisionTreeClassifier(max_depth=5)

model.fit(x_train, y_train)

x_test = test_data.iloc[:, 2:-1]
y_test = test_data.iloc[:, -1]
y_pred = model.predict(x_test)
print("Збудовано класифікаційну модель дерева прийняття рішень \n")

dot_data = export_graphviz(model, out_file=None, 
                           feature_names = ["F_1", "F_2", "F_3", "F_4", "F_5"],
                           class_names=["Class_0", "Class_1"],
                           filled=True, rounded=True, special_characters=True)

graph = Source(dot_data)

graph.render("Decision_tree", format="png", cleanup=True)
print("Граф було створено окремо у файл: Decision_tree.png\n")


def calculate_metrics(model, x_cord, y_cord):
    all_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f_scores': 0, 'MCC': 0, 'BA': 0, 'Y_J_statistics': 0}
    model_predictions = model.predict(x_cord)
    all_metrics['accuracy'] = accuracy_score(y_cord, model_predictions)
    all_metrics['precision'] = precision_score(y_cord, model_predictions)
    all_metrics['recall'] = recall_score(y_cord, model_predictions)
    all_metrics['f_scores'] = f1_score(y_cord, model_predictions)
    all_metrics['MCC'] = matthews_corrcoef(y_cord, model_predictions)
    all_metrics['BA'] = balanced_accuracy_score(y_cord, model_predictions)
    all_metrics['Y_J_statistics'] = recall_score(y_cord, model_predictions) + recall_score(y_cord, model_predictions, pos_label=0) - 1
    return all_metrics

train_model_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy')
train_model_entropy.fit(x_train, y_train)

train_model_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
train_model_gini.fit(x_train, y_train)

metrics_test_gini = calculate_metrics(train_model_gini, x_test, y_test)
metrics_train_gini = calculate_metrics(train_model_gini, x_train, y_train)

metrics_test_entropy = calculate_metrics(train_model_entropy, x_test, y_test)
metrics_train_entropy = calculate_metrics(train_model_entropy, x_train, y_train)


data_train_values = pd.DataFrame({'Тренувальна вибірка (gini)': metrics_train_gini, 'Тренувальна вибірка (entropy)': metrics_train_entropy})
data_train_values.plot(kind='bar', figsize=(10, 6))
plt.title('Порівняння метрик для тренувальних вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0.95, 1.0)
# plt.show()
plt.savefig('train_metrics.png')

print(data_train_values, "\n")


data_test_values = pd.DataFrame({'Тестова вибірка (gini)': metrics_test_gini, 'Тестова вибірка (entropy)': metrics_test_entropy})
data_test_values.plot(kind='bar', figsize=(10, 6))
plt.title('Порівняння метрик для тестових вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0.95, 1.0)
# plt.show()
plt.savefig('test_metrics.png')

print(data_test_values)