import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px

@st.cache_data
def get_melt_categorical(df, categorical_features):
    """Функция для отображения категориальных признаков"""
    # melt (расплавление)
    cat_df = pd.DataFrame(
        df[categorical_features].melt(
            var_name='column',
            value_name='value'
        ).value_counts()
    ).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])
    return cat_df


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    pr = ProfileReport(df)
    return pr


@st.cache_data
def create_histogram(df, column_name, title):
    fig = px.histogram(
        df,
        x=column_name,
        color='Attrition',
        marginal="box",
        title=title,
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, numerical_features):
    corr = df[numerical_features].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.values
    )
    fig.update_layout(height=800)
    return fig


def display_simple_histograms(df, categorical_features, numerical_features):
    selected_category = st.selectbox(
        'Категория для анализа',
        categorical_features,
        key='histograms_category_selectbox'
    )
    fig = px.histogram(
        df,
        x=selected_category,
        title=f'Распределение по {selected_category}'
    )
    st.plotly_chart(fig, use_container_width=True)
    selected_numerical = st.selectbox(
        'Числовой признак для анализа',
        numerical_features,
        key='histograms_numerical_selectbox'
    )
    fig = px.box(df, y=selected_numerical, title=f'Распределение {selected_numerical}')
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def create_box_plots(df):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Возраст', 'Ежедневная ставка', 'Количество лет с последнего повышения',
                        'Общее количество рабочих лет', 'Количество лет в компании', 'Расстояние до работы')
    )
    colors = {'Yes': '#FF5733', 'No': '#1F77B4'}
    features = [
        'Age', 'DailyRate', 'YearsSinceLastPromotion',
        'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome'
    ]

    for i, feature in enumerate(features):
        for attrition_type in df['Attrition'].unique():
            filtered_df = df[df['Attrition'] == attrition_type]
            fig.add_trace(
                go.Box(
                    y=filtered_df[feature],
                    name=f'{feature} ({attrition_type})',
                    marker_color=colors[attrition_type],
                    showlegend=False
                ),
                row=(i // 2) + 1, col=(i % 2) + 1
            )

    fig.update_layout(height=1200, title_text="Распределение ключевых метрик по статусу увольнения")
    return fig

@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


@st.cache_data
def create_plot_attrition_percentage(df):
    attrition_percentage = (
        df.groupby(
            ['Gender', 'Department']
        )['Attrition']
        .value_counts(normalize=True).mul(100)
        .rename('Процент увольнений').reset_index()
    )
    fig = px.bar(
        attrition_percentage,
        x="Department",
        y="Процент увольнений",
        color="Attrition",
        barmode="group",
        facet_col="Gender",
        category_orders={'Attrition': ['Yes', 'No']},
        color_discrete_map={'Yes': '#FF5733', 'No': '#1F77B4'},
        labels={
            'Department': 'Отдел',
            'Attrition': 'Увольнение',
            'Gender': 'Пол'
        })
    fig.update_traces(
        texttemplate='%{y:.3s}%',
        textposition='outside',
    )
    fig.update_layout(title_text='Процент увольнений по отделам и полу', yaxis_ticksuffix='%')
    return fig


@st.cache_data
def create_plot_work_life_balance_attrition(df):
    work_life_balance_attrition = (
        df.groupby(['WorkLifeBalance', 'Gender'])['Attrition']
        .value_counts(normalize=True).mul(100)
        .rename('Процент увольнений').reset_index()
    )

    fig = px.bar(
        work_life_balance_attrition,
        x='WorkLifeBalance',
        y='Процент увольнений',
        color='Attrition',
        facet_row='Gender',
        barmode='group',
        category_orders={'Attrition': ['Yes', 'No']},
        color_discrete_map={'Yes': '#FF5733', 'No': '#1F77B4'},
        labels={
            'WorkLifeBalance': 'Баланс работы',
            'Attrition': 'Увольнение',
            'Gender': 'Пол'
        }
    )

    fig.update_traces(
        texttemplate='%{y:.3s}%',
        textposition='outside',
    )

    fig.update_layout(
        title_text='Процент увольнений по балансу работы и полу',
        height=750,
        xaxis_title='Баланс работы',
        yaxis_ticksuffix='%',
    )

    return fig


@st.cache_data
def create_plot_job_satisfaction_attrition(df):
    job_satisfaction_attrition = (
        df.groupby(['Attrition'])['JobSatisfaction']
        .value_counts(normalize=True)
        .mul(100)
        .rename('Процент увольнений')
        .reset_index()
    )

    fig = px.bar(
        job_satisfaction_attrition,
        x='JobSatisfaction',
        y='Процент увольнений',
        color='Attrition',
        title='Уровни удовлетворенности работы и текучести кадров',
        labels={
            'Attrition': 'Увольнение',
        }
    )
    fig.update_xaxes(title='Уровень удовлетворенности работой')
    fig.update_yaxes(ticksuffix='%')
    fig.update_traces(texttemplate='%{y:.2s}%', textposition='inside')
    return fig


@st.cache_data
def create_plot_salary_department_attrition_gender(df):
    import plotly.express as px

    salary_department_attrition_gender = (
        df.groupby(['Department', 'Attrition', 'Gender'])['MonthlyIncome']
        .median().mul(12)  # Перевод в годовую зарплату
        .rename('Salary').reset_index()
        .sort_values(by=['Gender', 'Salary'], ascending=[True, False])  # Сортировка по полу и зарплате
    )

    fig = px.bar(
        salary_department_attrition_gender,
        x='Department',
        y='Salary',
        color='Gender',
        barmode='group',
        facet_col='Attrition',
        category_orders={'Attrition': ['Yes', 'No']},
        labels={
            'Department': 'Отдел',
            'Salary': 'Зарплата',
            'Attrition': 'Увольнение',
            'Gender': 'Пол'
        }
    )

    fig.update_traces(
        texttemplate='$%{y:.3s}',
        textposition='outside',
    )

    fig.update_layout(
        title_text='Медианная зарплата по отделам и статусу увольнения',
        yaxis=dict(title='Зарплата', tickprefix='$', range=(0, 79900)),
        height=500
    )
    return fig


@st.cache_data
def display_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Количество сотрудников", len(df))
    col2.metric("Мужчины", len(df[df['Gender'] == 'Male']))
    col3.metric("Женщины", len(df[df['Gender'] == 'Female']))
    col4.metric("Уровень ухода сотрудников", "16.2%")


def display_bar_chart(df):
    c1, c2 = st.columns(2)
    selected = c1.selectbox(
        'Категория для анализа',
        [
            'JobRole', 'JobLevel', 'Department', 'EducationField', 'MaritalStatus', 'Education'
        ],
        key='bar_selected'
    )
    colored = c2.selectbox(
        'Фильтр',
        [
            'Attrition', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction'
        ],
        key='bar_colored'
    )

    sub_data = df.groupby([selected, colored])["Age"].count().reset_index(name='Counts')

    fig = px.bar(
        sub_data,
        x=selected, y="Counts",
        color=colored,
        title=f"{selected} с {colored}",
    )
    st.plotly_chart(fig, use_container_width=True, key='bar_chart')


def display_box_plot(df, numerical_features, categorical_features):
    c1, c2, c3 = st.columns(3)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='box_feature1')
    feature2 = c2.selectbox('Второй признак', categorical_features, key='box_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, 'Attrition', 'OverTime'], key='box_filter_by')

    if feature2 == filter_by:
        filter_by = None

    fig = px.box(
        df,
        x=feature1, y=feature2,
        color=filter_by,
        title=f"Распределение {feature1} по разным {feature2}"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_scatter_plot(df, numerical_features):
    from scipy.stats import stats
    c1, c2, c3, c4 = st.columns(4)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='scatter_feature1')
    feature2 = c2.selectbox('Второй признак', ['MonthlyIncome', 'Age', 'YearsAtCompany'], key='scatter_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, 'Attrition', 'OverTime', 'Department'], key='scatter_filter_by')

    correlation = round(stats.pearsonr(df[feature1], df[feature2])[0], 4)
    c4.metric("Корреляция", correlation)

    fig = px.scatter(
        df,
        x=feature1, y=feature2,
        color=filter_by, trendline='ols',
        opacity=0.5,
        height=400,
        title=f'Корреляция между {feature1} и {feature2}'
    )
    st.plotly_chart(fig, use_container_width=True)


def app(df, current_dir: Path):
    st.title("Прогнозирование ухода сотрудников")
    st.image(str(current_dir / "images" / "main2.png"), use_column_width='auto')

    st.markdown(
        """
        ## Область применения
        Система анализа данных будет использоваться для предсказания ухода сотрудников из компании.
        Это позволит HR-отделам оптимизировать стратегии удержания персонала и улучшить общую рабочую атмосферу.

        ## Ключевые параметры и характеристики данных
        """
    )

    tab1, tab2 = st.tabs(["Показать описание данных", "Показать пример данных"])
    with tab1:
        st.markdown(
            """
            ## Описание данных
            | Параметр                                                 | Описание                                                                 |
            |----------------------------------------------------------|--------------------------------------------------------------------------|
            | Возраст (Age)                                            | Возраст сотрудника, целочисленное значение                               |
            | Уход (Attrition)                                         | Уход сотрудника, бинарный признак (Yes/No)                               |
            | Командировки (BusinessTravel)                            | Частота командировок, категориальный признак                             |
            | Дневная ставка (DailyRate)                               | Дневная ставка сотрудника, целочисленное значение                        |
            | Отдел (Department)                                       | Отдел, в котором работает сотрудник, категориальный признак              |
            | Расстояние от дома (DistanceFromHome)                    | Расстояние от дома до работы, целочисленное значение                     |
            | Образование (Education)                                  | Уровень образования сотрудника, категориальный признак                   |
            | Область образования (EducationField)                     | Область образования, категориальный признак                              |
            | Удовлетворенность окружением (EnvironmentSatisfaction)   | Удовлетворенность рабочим окружением, категориальный признак             |
            | Пол (Gender)                                             | Пол сотрудника, категориальный признак                                   |
            | Уровень работы (JobLevel)                                | Уровень должности сотрудника, категориальный признак                     |
            | Должность (JobRole)                                      | Название должности, категориальный признак                               |
            | Удовлетворенность работой (JobSatisfaction)              | Удовлетворенность работой, категориальный признак                        |
            | Семейное положение (MaritalStatus)                       | Семейное положение, категориальный признак                               |
            | Месячный доход (MonthlyIncome)                           | Месячный доход сотрудника, целочисленное значение                        |
            | Сверхурочные (OverTime)                                  | Сверхурочная работа, бинарный признак (Yes/No)                           |
            | Удовлетворенность отношениями (RelationshipSatisfaction) | Удовлетворенность отношениями на работе, категориальный признак          |
            | Уровень акций (StockOptionLevel)                         | Уровень акционных опционов у сотрудника, категориальный признак          |
            | Общий трудовой стаж (TotalWorkingYears)                  | Общий трудовой стаж в годах, целочисленное значение                      |
            | Обучение в прошлом году (TrainingTimesLastYear)          | Количество обучений в прошлом году, целочисленное значение               |
            | Баланс работы и жизни (WorkLifeBalance)                  | Баланс между работой и личной жизнью, категориальный признак             |
            |Количество лет в компании (YearsAtCompany)                         | Количество лет, проведенных в компании, целочисленное значение  |
            | Количество лет на текущей должности (YearsInCurrentRole)           | Количество лет на текущей должности, целочисленное значение    |
            | Количество лет с последнего повышения (YearsSinceLastPromotion)    | Количество лет с последнего повышения, целочисленное значение  |
            | Количество лет с текущим руководителем (YearsWithCurrManager)      | Годы работы с текущим руководителем, целочисленное значение    |
            | Количество компаний (NumCompaniesWorkedGroup)            | Количество компаний, в которых работал сотрудник, категориальный признак |
            """
        )
    with tab1:
        st.header("Пример данных")
        st.dataframe(df.head())

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.header("Предварительный анализ данных")
    st.dataframe(get_data_info(df), use_container_width=True)

    st.markdown(
        """
        Предварительный анализ данных показал следующее:
        - Данные содержат 1470 строк и 35 столбцов.
        - В датасете нет пропущенных значений, что указывает на хорошее качество данных.
        - Некоторые параметры имеют высокое количество уникальных значений, что может указывать на разнообразие данных и потенциал для более детального анализа.
        - Большинство категориальных данных имеют несколько категорий (2-6), что удобно для анализа и визуализации.
        """
    )
    st.header("Основные статистики для признаков")

    display_metrics(df)

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe())
        st.markdown(
            """
            Эти основные статистики помогают понять распределение и диапазон значений каждого признака, что важно для последующего анализа и моделирования данных.
            - Возраст (Age): Средний возраст сотрудников составляет примерно 37 лет, с минимальным возрастом 18 лет и максимальным 60 лет. Распределение возраста показывает стандартное отклонение в 9 лет.
            - Дневная ставка (DailyRate): Средняя дневная ставка равна 802 с большим разбросом значений (минимум 102, максимум 1499).
            - Месячный доход (MonthlyIncome): Средний месячный доход составляет около 6503 с вариативностью от 1009 до 19999. Это указывает на значительное различие в уровнях оплаты труда внутри организации.
            - Общий трудовой стаж (TotalWorkingYears): Варьируется от 0 до 40 лет с медианой в 10 лет, что говорит о широком диапазоне опыта среди сотрудников.

            Числовые данные показывают большое разнообразие среди сотрудников по возрасту, оплате труда и общему трудовому стажу, что может влиять на HR-стратегии по удержанию сотрудников и определению потребностей в обучении.
            """
        )
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='category'))
        st.markdown(
            """
            - Уход (Attrition): Два уникальных значения (Yes и No), с преобладанием "No" (1233 случая).
            - Командировки (BusinessTravel): Три категории с большинством путешествующих редко (1043 случая).
            - Отдел (Department): Три отдела, с наибольшим числом сотрудников в исследовании и разработке (961).
            - Образование (Education) и Пол (Gender): Несколько уровней образования и два пола, с мужским полом чуть больше представлен (882 против женского).
            - Удовлетворенность работой (JobSatisfaction) и Баланс работы и жизни (WorkLifeBalance): По четыре уровня каждый, где чаще всего встречаются высокие значения удовлетворенности и баланса.
            Категориальные данные показывают, что большинство сотрудников не склонны к уходу, чаще всего редко командируются, работают в отделе исследований и разработок, и имеют высокую степень удовлетворенности работой и баланса работы и жизни. Эти факторы могут быть важны при разработке стратегий для улучшения удержания и удовлетворенности сотрудников."""
        )

    st.header("Визуализация данных")



    tab1, tab2 = st.tabs(["Простые графики", "Показать отчет о данных"])
    with tab1:
        st.subheader("Распределение сотрудников")
        display_bar_chart(df)
        st.subheader("Столбчатые диаграммы для категориальных признаков")
        display_simple_histograms(df, categorical_features, numerical_features)

    with tab2:
        if st.button("Сформировать отчёт", use_container_width=True, type='primary'):
            st_profile_report(get_profile_report(df))

    st.header("Распределение числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox"
    )
    hist_fig = create_histogram(
        df,
        selected_feature,
        f"Распределение {selected_feature}"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.header("Корреляционный анализ")

    st.markdown(
        """
        Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае:
        * Сильные положительные корреляции между такими параметрами как: TotalWorkingYears и MonthlyIncome (0.77), YearsAtCompany и YearsWithCurrManager (0.77) указывают на взаимосвязь между опытом работы и заработной платой, а также стабильностью в компании и длительностью работы с одним руководителем.
        * Отрицательная корреляция между Age и AttrionLikelihood (-0.31) показывает, что молодые сотрудники склонны уходить из компании чаще, чем более взрослые.
        """
    )
    corr_fig = create_correlation_matrix(df, numerical_features)
    st.plotly_chart(corr_fig, use_container_width=True)

    st.subheader("Корреляция между числовыми признаками")
    display_scatter_plot(df, numerical_features)

    st.markdown(
        """
        ## Ящики с усами для числовых признаков
        Эти графики позволяют наглядно оценить распределение основных числовых параметров сотрудников в зависимости от их статуса увольнения.
       """
    )
    st.plotly_chart(create_box_plots(df), use_container_width=True)

    st.markdown(
        """
        ### Процент увольнений по отделам и полу
        Наибольшая текучесть кадров наблюдается среди женщин, работающих в отделе кадров: почти каждая третья женщина, работающая в отделе кадров, покидает компанию. Среди мужчин самая высокая текучесть кадров наблюдается в отделе продаж.
        """
    )
    st.plotly_chart(create_plot_attrition_percentage(df), use_container_width=True)

    st.markdown(
        """
        ### Процент увольнений по балансу работы и полу
        Среди женщин с самым высоким уровнем баланса между работой и личной жизнью, каждая четвертая покинула компанию, что является самой высокой долей среди всех оценок для женщин. Для мужчин самая высокая доля увольнений наблюдалась среди тех, у кого был самый низкий баланс между работой и личной жизнью.
        """
    )
    st.plotly_chart(create_plot_work_life_balance_attrition(df), use_container_width=True)

    st.markdown(
        """
        ### Уровни удовлетворенности работы и текучести кадров
        Среди уволившихся сотрудников большинство были удовлетворены своей работой: 53 % оценили удовлетворенность работой как хорошую или отличную, а 28 % были наименее удовлетворены своей работой.
        """
    )
    st.plotly_chart(create_plot_job_satisfaction_attrition(df), use_container_width=True)

    st.markdown(
        """
        ### Медианная зарплата по отделам и статусу увольнения
        По сравнению с нынешними сотрудниками, бывшие сотрудники имели более низкую медианную зарплату во всех трех отделах. В отделе кадров женщины, как правило, имеют более высокую медианную зарплату, чем мужчины.
        """
    )
    st.plotly_chart(create_plot_salary_department_attrition_gender(df), use_container_width=True)

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany'],
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=1,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
