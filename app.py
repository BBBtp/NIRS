# файл: app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Заголовок
st.title("Демонстрация модели GradientBoostingClassifier для задачи дефолта по кредиту")

# Загружаем данные
@st.cache_data
def load_data():
    df = pd.read_excel('data/default_of_credit_card_clients.xls', header=1)
    df.head()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    return df

df = load_data()

st.write("Первые 5 строк датасета:")
st.dataframe(df.head())

# Разделяем X и y
X = df.drop(columns=['ID', 'default'])
y = df['default']

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Интерфейс для гиперпараметров
st.sidebar.header("Настройка гиперпараметров GradientBoostingClassifier")

n_estimators = st.sidebar.slider('Количество деревьев (n_estimators)', min_value=50, max_value=500, value=100, step=10)
learning_rate = st.sidebar.select_slider('Скорость обучения (learning_rate)', options=[0.01, 0.05, 0.1, 0.2, 0.3])
max_depth = st.sidebar.slider('Максимальная глубина дерева (max_depth)', min_value=1, max_value=10, value=3)

# Обучение модели
model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Метрики
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# Вывод метрик
st.subheader("Результаты модели:")
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**F1-score:** {f1:.4f}")
st.write(f"**ROC-AUC:** {roc_auc:.4f}")

# Confusion Matrix
st.subheader("Матрица ошибок (Confusion Matrix):")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

# ROC Curve
st.subheader("ROC-кривая:")
fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
st.pyplot(fig_roc)
