import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction, prediction_logreg
from plotly.io import templates
templates.default = "ggplot2"

st.set_page_config(
    page_title="Прогноз оттока работника",
    page_icon="♾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18'], inplace=True)
    for feature in [
        "JobInvolvement", "JobSatisfaction", 'EnvironmentSatisfaction',
        'RelationshipSatisfaction', "PerformanceRating", 'WorkLifeBalance'
    ]:
        df[feature].replace(
            {
                1: "Low",
                2: "Medium",
                3: "High",
                4: "Very High"
            },
            inplace=True
        )

    df['Education'].replace(
        {
            1: 'High School',
            2: 'Undergrad',
            3: 'Graduate',
            4: 'Post Graduate',
            5: 'Doctorate'
        },
        inplace=True
    )
    df['MaritalStatus'].replace(
        {
            0: 'Single',
            1: 'Married',
            2: 'Divorced'
        },
        inplace=True
    )
    df["StockOptionLevel"].replace(
        {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Very High"
        },
        inplace=True
    )

    for feature in df.select_dtypes(include='object').columns.tolist():
        df[feature] = df[feature].astype('category')
    return df


class Menu:
    apps = [
        {
            "func"  : home.app,
            "title" : "Главная",
            "icon"  : "house-fill"
        },
        {
            "func": prediction_logreg.app,
            "title": "Прогнозирование — Логистическая регрессия",
            "icon": "person-check-fill"
        },
        {
            "func"  : prediction.app,
            "title" : "Прогнозирование — Деревья решений",
            "icon"  : "pie-chart-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons  = [app["icon"]  for app in self.apps]

            selected = option_menu(
                "Menu",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )
            st.info(
                """
                ## Анализ датасета HR Analytics и предсказание ухода сотрудников
                
                Это веб-приложение предназначено для анализа данных о сотрудниках компании с целью предсказания их увольнения. 
                Оно включает в себя различные аспекты анализа данных, включая визуализацию, статистический анализ и машинное обучение.
                
                Проект выполняется в рамках задачи HR Analytics, которая помогает компаниям понять, что заставляет сотрудников уходить и какие факторы влияют на их решение остаться или уйти.
                """
            )
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    df = preprocess_data(current_dir / 'Employee-Attrition.csv')
    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    menu = Menu()
    selected = menu.run()
    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break