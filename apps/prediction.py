import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from streamlit_option_menu import option_menu


@st.cache_data
def create_plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC кривая (AUC = %0.2f)' % roc_auc_score(y_true, y_prob)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Диагональ'
    ))

    fig.update_layout(
        xaxis_title='Доля ложно-положительных результатов',
        yaxis_title='Доля истинно-положительных результатов',
        title='Кривая ROC',
        width=900
    )
    return fig


@st.cache_data
def create_plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)

    fig = px.imshow(
        cm,
        labels=dict(x="Предсказанный класс", y="Истинный класс"),
        x=['Нет', 'Да'], y=['Нет', 'Да'],
        title='Нормализованная матрица ошибок' if normalize else 'Матрица ошибок',
        color_continuous_scale='Blues'
    )

    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    fig.update_yaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    # Добавление надписей
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=i, y=j,
                text=str(cm[j, i]),
                showarrow=False,
                font=dict(color="white" if cm[j, i] > thresh else "black"),
                align="center"
            )
    return fig


def score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        prob = clf.predict_proba(X_train)[:, 1]
        clf_report = classification_report(y_train, pred, output_dict=True)
        st.subheader("Результат обучения:")
        st.write(f"Точность модели: {accuracy_score(y_train, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_train, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_train, pred, normalize=True), use_container_width=True)
    else:
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]
        clf_report = classification_report(y_test, pred, output_dict=True)
        st.subheader("Результат тестирования:")
        st.write(f"Точность модели: {accuracy_score(y_test, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_test, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_test, pred, normalize=True), use_container_width=True)


@st.cache_data
def get_tree_clf(X_train, y_train):
    # Определение параметров модели
    params = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 20))),
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": list(range(1, 20)),
    }

    # Обучение модели с использованием GridSearchCV
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_cv = GridSearchCV(
        tree_clf,
        params,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        cv=5
    )
    tree_cv.fit(X_train, y_train)
    best_params = tree_cv.best_params_

    tree_clf = DecisionTreeClassifier(**best_params)
    tree_clf.fit(X_train, y_train)

    st.subheader("Лучшие параметры и параметры по умолчанию")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Лучшие параметры:", best_params)
    with c2:
        st.write("Параметры для GridSearchCV:")
        st.json(params, expanded=False)
    return tree_clf


def plot_feature_importance(rf_clf, X_train):
    # Определение цветовой палитры
    col = sns.color_palette("mako", 20).as_hex()[::-1]

    # Вычисление важности признаков
    feat = pd.DataFrame(
        {"Importance": rf_clf.feature_importances_},
        index=X_train.columns
    ).sort_values(by="Importance")

    # Выбор топ 15 признаков с наибольшей важностью
    plot_df = feat.nlargest(15, columns='Importance').sort_values(by="Importance", ascending=False)

    # Построение графика
    fig = px.bar(
        plot_df,
        x=plot_df.index, y='Importance', text='Importance',
        color=plot_df.index, opacity=0.7, color_discrete_sequence=col
    )
    fig.update_traces(
        texttemplate='%{text:.2f}', textposition='outside',
        hovertemplate='%{x}<br>Importance = %{y:.2f}'
    )
    fig.update_layout(
        title_text='Топ 15 наиболее важных предикторов увольнения сотрудников',
        coloraxis_showscale=False,
        showlegend=False,
        height=500,
        yaxis_title="Importance",
        xaxis=dict(title="", showgrid=True, gridcolor="#f5f5f5")
    )
    st.plotly_chart(fig, use_container_width=True)


def print_model_adequacy_section(current_dir: Path):
    st.markdown(
        """
        ## Оценка адекватности модели
        При оценки адекватности модели важно использовать несколько метрик, которые помогают оценить различные аспекты производительности модели.
 
        ### Матрица ошибок (Confusion Matrix)
 
        Матрица ошибок позволяет визуально оценить, как модель справляется с каждым из классов задачи. Она показывает, сколько примеров, предсказанных в каждом классе, действительно принадлежат этому классу.
        """
    )
    st.image(str(current_dir / 'images' / 'matrix.jpg'))
    st.markdown(
        """
        ### Отчет о классификации (Precision, Recall, F1-Score)
        * Precision (Точность) описывает, какая доля положительных идентификаций была верной (TP / (TP + FP)).
        * Recall (Полнота) показывает, какая доля фактических положительных классов была идентифицирована (TP / (TP + FN)).
        * F1-Score является гармоническим средним Precision и Recall и помогает учесть обе эти метрики в одной.

        ### Кривая ROC и площадь под кривой AUC
        * ROC кривая (Receiver Operating Characteristic curve) помогает визуально оценить качество классификатора. Ось X показывает долю ложноположительных результатов (False Positive Rate), а ось Y — долю истинноположительных результатов (True Positive Rate).
        * AUC (Area Under Curve) — площадь под ROC кривой, которая дает количественную оценку производительности модели.
        """
    )


def app(df, current_dir: Path):
    st.title("Прогнозирование ухода сотрудников")

    st.image(str(current_dir / "images" / "main.bmp"), width=150, use_column_width='auto')

    df_encoded = df.copy(deep=True)
    st.markdown(
        """
        # Подготовка набора данных
        Перед подачей наших данных в модель машинного обучения нам сначала нужно подготовить данные. Это включает в себя кодирование всех категориальных признаков (либо LabelEncoding, либо OneHotEncoding), поскольку модель ожидает, что признаки будут представлены в числовой форме. Также для лучшей производительности мы выполним масштабирование признаков, то есть приведение всех признаков к одному масштабу с помощью StandardScaler, предоставленного в библиотеке scikit-learn.
        """
    )
    categorical_features = df.select_dtypes(include='category').columns.tolist()
    label_encoders = {}
    for feature in categorical_features:
        label_encoder = LabelEncoder()
        df_encoded[feature] = label_encoder.fit_transform(df[feature])
        label_encoders[feature] = label_encoder

    st.markdown(
        """
        ## Кодирование признаков (Feature Encoding)
        """
    )
    feature_encoding_tab1, feature_encoding_tab2 = st.tabs([
        "Данные до Feature Encoding",
        "Данные после Feature Encoding"
    ])
    with feature_encoding_tab1:
        st.dataframe(df.head())
    with feature_encoding_tab2:
        st.dataframe(df_encoded.head())

    X = df_encoded.drop('Attrition', axis=1).copy(deep=True)
    Y = df_encoded['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.write("Train Shape:", X_train.shape, y_train.shape)
    st.write("Test Shape:", X_test.shape, y_test.shape)

    st.markdown(
        """
        # Моделирование
        ## Работа с несбалансированными данными
        Обратите внимание, что у нас есть несбалансированный набор данных, в котором большинство наблюдений относятся к одному типу ('NO'). В нашем случае, например, примерно 84% наблюдений имеют метку 'No', а только 16% - 'Yes', что делает этот набор данных несбалансированным.
        
        Для работы с такими данными необходимо принять определенные меры, иначе производительность нашей модели может существенно пострадать. В этом разделе я рассмотрю два подхода к решению этой проблемы.
        
        ### Увеличение числа примеров меньшинства или уменьшение числа примеров большинства
        В несбалансированных наборах данных основная проблема заключается в том, что данные сильно искажены, т.е. количество наблюдений одного класса значительно превышает количество наблюдений другого. Поэтому в этом подходе мы либо увеличиваем количество наблюдений для класса-меньшинства (oversampling), либо уменьшаем количество наблюдений для класса-большинства (undersampling).
        
        Стоит отметить, что в нашем случае количество наблюдений и так довольно мало, поэтому более подходящим будет метод увеличения числа примеров.
        
        Ниже я использовал технику увеличения числа примеров, известную как SMOTE (Synthetic Minority Oversampling Technique), которая случайным образом создает некоторые "синтетические" инстансы для класса-меньшинства, чтобы данные по обоим классам стали более сбалансированными.
        
        Важно использовать SMOTE до шага кросс-валидации, чтобы избежать переобучения модели, как это бывает при выборе признаков.
        
        ###  Выбор правильной метрики оценки
        Еще один важный аспект при работе с несбалансированными классами - это выбор правильных оценочных метрик.
        
        Следует помнить, что точность (accuracy) не является хорошим выбором. Это связано с тем, что из-за искажения данных даже алгоритм, всегда предсказывающий класс-большинство, может показать высокую точность. Например, если у нас есть 20 наблюдений одного типа и 980 другого, классификатор, предсказывающий класс-большинство, также достигнет точности 98%, но это не будет полезной информацией.
        
        В таких случаях мы можем использовать другие метрики, такие как:
        
        - **Точность (Precision)** — (истинно положительные)/(истинно положительные + ложно положительные)
        - **Полнота (Recall)** — (истинно положительные)/(истинно положительные + ложно отрицательные)
        - **F1-Score** — гармоническое среднее точности и полноты
        - **AUC ROC** — ROC-кривая, график между чувствительностью (Recall) и (1-specificity) (Специфичность=Точность)
        - **Матрица ошибок** — отображение полной матрицы ошибок
        """
    )

    st.write(
        r"""
        # Классификатор на основе дерева решений
        ## Введение

        **Дерево решений** — это наглядный и легко интерпретируемый метод машинного обучения, используемый в задачах классификации и регрессии. Классификатор на основе дерева решений строит модель принятия решений в форме древовидной структуры, разделяя набор данных на всё меньшие подмножества, хотя сам алгоритм также способен работать с непрерывными зависимыми переменными.

        ### Как работает дерево решений

        Дерево решений моделирует прогнозы, используя дерево решений, которое представляет ряд последовательных решений, которые можно интерпретировать как "вопросы", на которые ответы разделяют данные на два подмножества. Процесс начинается с корневого узла (вершины дерева), который делит данные на две группы на основе наиболее значимого атрибута (признака), и продолжается до тех пор, пока не будут достигнуты критерии останова, такие как максимальная глубина дерева или минимальное количество объектов в листе.

        ### Формулы и критерии разделения:
        - Энтропия (мера неопределенности или чистоты):
        $$
        H(S) = -\sum_{x \in X} p(x) \log_2 p(x)
        $$
        - Прирост информации (Information Gain):**
        $$
        IG(S, A) = H(S) - \sum_{t \in T} p(t) H(t)
        $$
        где $p(t)$ это вероятность того, что элемент из $S$ приходится на подмножество $t$, которое создается при разделении по атрибуту $A$.

        Классификаторы на основе дерева решений часто используются в различных областях благодаря своей
        """
    )
    st.write(
        """
        ## Параметры Дерева Решений

        - **criterion (Критерий)**: Функция для оценки качества разделения. Поддерживаемые критерии: "gini" для неопределенности Джини и "entropy" (энтропия) для прироста информации.
        - **splitter (Разделитель)**: Стратегия, используемая для выбора разделения в каждом узле. Поддерживаемые стратегии: "best" для выбора лучшего разделения и "random" для случайного выбора разделения.
        - **max_depth (Максимальная глубина)**: Максимальная глубина дерева. Если не указано, то узлы расширяются до тех пор, пока все листья не станут чистыми или пока в каждом листе не будет менее min_samples_split образцов.
        - **min_samples_split (Минимальное количество образцов для разделения)**: Минимальное количество образцов, необходимых для разделения внутреннего узла.
        - **min_samples_leaf (Минимальное количество образцов в листе)**: Минимальное количество образцов, которые должны быть в листовом узле. Точка разделения на любой глубине будет рассматриваться только в том случае, если она оставляет по крайней мере min_samples_leaf обучающих образцов в каждой из левой и правой ветвей. Это может сгладить модель, особенно в задачах регрессии.
        - **min_weight_fraction_leaf (Минимальная долевая масса в листе)**: Минимальная взвешенная доля от общей суммы весов (всех входных образцов), требуемая для нахождения в листовом узле. Образцы имеют равный вес, если sample_weight не предоставлен.
        - **max_features (Максимальное количество признаков)**: Количество признаков, которое следует учитывать при поиске лучшего разделения.
        - **max_leaf_nodes (Максимальное количество листьев)**: Построение дерева с максимальным количеством листьев в порядке лучших первых. Лучшие узлы определяются как относительное снижение неопределенности. Если не указано, число листьев не ограничено.
        - **min_impurity_decrease (Минимальное уменьшение неопределенности)**: Узел будет разделен, если это разделение вызывает снижение неопределенности, равное или большее, чем данное значение.
        - **min_impurity_split (Порог для ранней остановки роста дерева)**: Узел будет разделен, если его неопределенность выше порога, в противном случае он будет листом.

        ## Настройка гиперпараметров для Классификатора Дерева Решений

        Настройка гиперпараметров - это важный шаг в процессе моделирования, позволяющий оптимизировать производительность классификатора дерева решений посредством экспериментов с различными значениями параметров, чтобы найти наилучшую комбинацию для конкретного набора данных.
        """
    )

    try:
        def test2():
            return joblib.load(str(current_dir / "models" / 'decision_tree_model.joblib'))

        tree_clf = test2()
    except Exception as e:
        print('!!' * 50)
        print(e)
        tree_clf = get_tree_clf(X_train, y_train)
        joblib.dump(tree_clf, str(current_dir / "models" / 'decision_tree_model.joblib'))

    st.markdown(
        """
        ### Топ важных характеристик
        """
    )
    plot_feature_importance(tree_clf, X_train)

    st.write(
        """
        ## Визуализация дерева решений
        """
    )

    print_model_adequacy_section(current_dir)

    tab1, tab2 = st.tabs(["Результаты модели на зависимых данных", "Результаты модели на независимых данных", ])
    with tab1:
        score(tree_clf, X_train, y_train, X_test, y_test, train=True)
    with tab2:
        score(tree_clf, X_train, y_train, X_test, y_test, train=False)
    st.markdown(
        """
        ### Выводы

        Как и логистическая регрессия, классификатор дерева решений также лучше предсказывает класс 0. Однако, если сравнить его с логистической регрессией, дерево решений показывает немного более низкую точность, precision и recall для обоих классов.
        Касательно AUC, модель на основе дерева решений показала AUC 0.81 для обучающей выборки и AUC 0.66 для тестовой выборки, что указывает на ухудшение способности модели правильно ранжировать положительные случаи по сравнению с логистической регрессией.
        Когда сравниваются ROC-кривые и нормализованные матрицы ошибок, логистическая регрессия в целом показывает лучшие результаты по AUC и более сбалансированные показатели точности и полноты. Дерево решений немного отстает по этим метрикам, особенно в области полноты (recall) для класса 1.
        """
    )

    st.markdown(
        """
        # Проверка моделей на практике
        """
    )
    ##__________________________ Prediction Form ______________________
    with st.form('Emploee Features'):
        ##______________________ Section 1 ____________________________
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            # MaritalStatus
            st.markdown('Семейное положение')
            marital_status = option_menu(
                None,
                options=['Single', 'Married', 'Divorced'],
                menu_icon="::",
                icons=["::", "::", "::"],
                default_index=0,
                key=1

            )

            # OverTime
            st.markdown('Сверхурочная работа')
            overtime = option_menu(
                None,
                options=['Yes', 'No'],
                menu_icon="::",
                icons=["::", "::"],
                default_index=1,
                key=2
            )

            # OverTime
            st.markdown('Гендер')
            gender = option_menu(
                None,
                options=['Male', 'Female'],
                menu_icon="::",
                icons=["::", "::"],
                default_index=1,
                key=3
            )

            # StockOptionLevel
            st.markdown('Уровень акций')
            stock_level = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                default_index=2,
                key=4
            )

        with cc2:
            age = st.slider(
                "Возраст",
                min_value=df["Age"].min(),
                max_value=df["Age"].max(),
                value=int(df["Age"].median()),
                key=5
            )
            monthly_rate = st.slider(
                'Реальный месячный доход сотрудника',
                min_value=df["MonthlyRate"].min(),
                max_value=df["MonthlyRate"].max(),
                value=int(df["MonthlyRate"].mean()),
                key=6
            )

            monthly_income = st.slider(
                'Базовая месячная ставка сотрудника',
                min_value=df["MonthlyIncome"].min(),
                max_value=df["MonthlyIncome"].max(),
                value=int(df["MonthlyIncome"].mean()),
                key=7
            )

            daily_rate = st.slider(
                "Дневная ставка",
                min_value=df["DailyRate"].min(),
                max_value=df["DailyRate"].max(),
                value=int(df["DailyRate"].mean()),
                key=8
            )

            hourly_rate = st.slider(
                "Часовая ставка",
                min_value=df["HourlyRate"].min(),
                max_value=df["HourlyRate"].max(),
                value=int(df["HourlyRate"].mean()),
                key=9
            )

            distance_from_home = st.slider(
                "Расстояние от дома",
                min_value=df["DistanceFromHome"].min(),
                max_value=df["DistanceFromHome"].max(),
                value=int(df["DistanceFromHome"].mean()),
                key=10
            )

            trainings_last_year = st.slider(
                'Количество обучений в прошлом году',
                min_value=df["TrainingTimesLastYear"].min(),
                max_value=df["TrainingTimesLastYear"].max(),
                value=int(df["TrainingTimesLastYear"].mean()),
                key=11
            )

            total_working_years = st.slider(
                'Общий трудовой стаж',
                min_value=df["TotalWorkingYears"].min(),
                max_value=df["TotalWorkingYears"].max(),
                value=int(df["TotalWorkingYears"].mean()),
                key=12
            )
            percent_salary_hike = st.slider(
                'Повышение зарплаты в процентах',
                min_value=df["PercentSalaryHike"].min(),
                max_value=df["PercentSalaryHike"].max(),
                value=int(df["PercentSalaryHike"].mean()),
                format="%d%%",
                key=13
            )

        ##______________________ Section 2 ____________________________
        ccc1, ccc2 = st.columns(2)
        with ccc1:
            job_role = st.selectbox(
                'Должность',
                df["JobRole"].unique(),
                index=0,
                key=14
            )
            business_travel = st.selectbox(
                'Командировки',
                df["BusinessTravel"].unique(),
                key=15
            )

            department = st.selectbox(
                "Отдел",
                df["Department"].unique(),
                key=16
            )
            years_in_current_role = st.number_input(
                'Количество лет на текущей должности',
                min_value=df["YearsInCurrentRole"].min(),
                max_value=df["YearsInCurrentRole"].max() * 2,
                value=int(df["YearsInCurrentRole"].mean()),
                key=17
            )

            years_with_curr_manager = st.number_input(
                'Количество лет с текущим руководителем',
                min_value=df["YearsWithCurrManager"].min(),
                max_value=df["YearsWithCurrManager"].max() * 2,
                value=int(df["YearsWithCurrManager"].mean()),
                key=18
            )

            years_at_company = st.number_input(
                'Количество лет в компании',
                min_value=df["YearsAtCompany"].min(),
                max_value=int(df["YearsAtCompany"].max() * 1.5),
                value=int(df["YearsAtCompany"].mean()),
                key=19
            )

        with ccc2:
            education = st.selectbox(
                "Уровень образования",
                df["Education"].unique(),
                index=0,
                key=20
            )

            education_field = st.selectbox(
                "Область образования",
                df["EducationField"].unique(),
                index=0,
                key=21
            )

            job_level = st.number_input(
                'Разряд работы',
                min_value=df["JobLevel"].min(),
                max_value=df["JobLevel"].max(),
                value=int(df["JobLevel"].mean()),
                key=22
            )
            num_companies_worked = st.number_input(
                'Количество компаний, в которых работал сотрудник',
                min_value=df["NumCompaniesWorked"].min(),
                max_value=df["NumCompaniesWorked"].max() * 2,
                value=int(df["NumCompaniesWorked"].mean()),
                key=23
            )
            years_since_last_promotion = st.number_input(
                'Количество лет с последнего повышения',
                min_value=df["YearsSinceLastPromotion"].min(),
                max_value=df["YearsSinceLastPromotion"].max() * 2,
                value=int(df["YearsSinceLastPromotion"].mean()),
                key=24
            )

        ##______________________ Section 2 ____________________________
        st.subheader('Показатели работы')
        cссс1, cссс2 = st.columns(2)
        with cссс1:
            st.markdown('Вовлеченность в работу')
            job_involvement = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                key=25,
                default_index=2
            )
        with cссс2:
            st.markdown('Производительность работника')
            performance_rating = option_menu(
                None,
                options=['High', 'Very High'],
                menu_icon="::",
                icons=["emoji-smile-fill", "emoji-grin-fill"],
                key=26,
                default_index=0
            )

        ##______________________ Section 3 ____________________________
        st.subheader('Показатели удовлетворенности')
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('Удовлетворенность окружением')
            environment_satisfaction = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                key=27,
                default_index=2
            )

        with c2:
            st.markdown('Баланс работы и жизни')
            work_life_balance = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                key=28,
                default_index=3
            )

        with c3:
            st.markdown('Удовлетворенность работой')
            job_satisfaction = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                key=29,
                default_index=2
            )

        with c4:
            st.markdown('Удовлетворенность руководством')

            relationship_satisfaction = option_menu(
                None,
                options=['Low', 'Medium', 'High', 'Very High'],
                menu_icon="::",
                icons=["emoji-frown-fill", "emoji-neutral-fill", "emoji-smile-fill", "emoji-grin-fill"],
                key=30,
                default_index=1
            )

        s1, s2 = st.columns(2)
        submitted = s1.form_submit_button("Отправить", use_container_width=True, type='primary')

        if submitted:
            ##______________________ Prediction and Subimition ____________________________
            data = pd.DataFrame(columns=X.columns)
            # Categorical features
            data.BusinessTravel           = label_encoders['BusinessTravel'].transform([business_travel])
            data.Department               = label_encoders['Department'].transform([department])
            data.Education                = label_encoders['Education'].transform([education])
            data.EducationField           = label_encoders['EducationField'].transform([education_field])
            data.EnvironmentSatisfaction  = label_encoders['EnvironmentSatisfaction'].transform([environment_satisfaction])
            data.Gender                   = label_encoders['Gender'].transform([gender])
            data.JobInvolvement           = label_encoders['JobInvolvement'].transform([job_involvement])
            data.JobRole                  = label_encoders['JobRole'].transform([job_role])
            data.JobSatisfaction          = label_encoders['JobSatisfaction'].transform([job_satisfaction])
            data.MaritalStatus            = label_encoders['MaritalStatus'].transform([marital_status])
            data.OverTime                 = label_encoders['OverTime'].transform([overtime])
            data.PerformanceRating        = label_encoders['PerformanceRating'].transform([performance_rating])
            data.RelationshipSatisfaction = label_encoders['RelationshipSatisfaction'].transform([relationship_satisfaction])
            data.StockOptionLevel         = label_encoders['StockOptionLevel'].transform([stock_level])
            data.WorkLifeBalance          = label_encoders['WorkLifeBalance'].transform([work_life_balance])

            # Numerical features
            data.Age                     = age
            data.DailyRate               = daily_rate
            data.DistanceFromHome        = distance_from_home
            data.HourlyRate              = hourly_rate
            data.JobLevel                = job_level
            data.MonthlyIncome           = monthly_income
            data.MonthlyRate             = monthly_rate
            data.NumCompaniesWorked      = num_companies_worked
            data.PercentSalaryHike       = percent_salary_hike
            data.TotalWorkingYears       = total_working_years
            data.TrainingTimesLastYear   = trainings_last_year
            data.YearsAtCompany          = years_at_company
            data.YearsInCurrentRole      = years_in_current_role
            data.YearsSinceLastPromotion = years_since_last_promotion
            data.YearsWithCurrManager    = years_with_curr_manager


            prediction = tree_clf.predict_proba(data)[:, 1][0]

            if prediction > 0.4:
                st.error('Ваш сотрудник собирается уволиться!')
                s2.metric("Вероятность", round(prediction, 4), "Риск увольнения: сильный", delta_color='inverse')
            elif prediction > 0.3:
                st.warning('Позаботьтесь о своем сотруднике')
                s2.metric("Вероятность", round(prediction, 4), "Риск увольнения: средний", delta_color='off')
            else:
                st.balloons()
                s2.metric("Вероятность", round(prediction, 4), "Риск увольнения: слабый", delta_color='normal')
