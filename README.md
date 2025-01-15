# Анализ факторов, влияющих на продолжительность жизни в разных странах, и построение предсказательной модели. 

Цель данного анализа — обработка данных, исследование взаимосвязей признаков, построение и оценка моделей машинного обучения для предсказания продолжительности жизни (Life_expectancy). Дополнительно создана база данных SQLite для сохранения обработанного набора данных.  

Ссылка на презентацию: https://drive.google.com/drive/folders/1fkvx1TBtcIVKUPAhFNdYXc9mWl_ngD75?usp=drive_link

## Этап 1. Модель исследования данных 
На этапе исследования данных мы проводим анализ доступной информации для выявления закономерностей и подготовки данных для построения прогностической модели.
Основные этапы: сбор данных, очистка данных, анализ данных, формирование гипотез.

### 1.	Сбор и загрузка данных
Загрузили данные из CSV-файла, включающего такие показатели, как уровень образования, доход на душу населения, уровень смертности и прочие факторы, которые могут влиять на продолжительность жизни.
Данные загружаются из файла Life-Expectancy-Data-Averaged.csv с использованием функции `load_data`. Пустые значения в данных заменяются на NaN с помощью параметра `na_values`.

### 2.	Предварительная обработка данных
   - Анализ данных.  
Отобразили форму данных (размерность) и типы столбцов. Просматриваются первые строки таблицы для общего понимания структуры данных.
Вычислили метрики качества данных: количество строк, столбцов, дубликатов, процент пропущенных значений.
   - Очистка данных.  
Удалили дубликаты и пропуски, обработали выбросы, чтобы избежать искажения итоговых результатов. Выбросы (аномалии) в числовых столбцах обработали с помощью метода интерквартильного размаха (IQR).
   - Обработка категориальных данных.  
Кодировали категориальные признаки, чтобы их можно было использовать в числовом анализе. Столбцы типа object кодируются числовыми значениями с помощью `astype("category").cat.codes`.


### 3.	Корреляционный анализ данных
Построили корреляционную матрицу, чтобы выявить признаки, которые имеют значительное влияние на продолжительность жизни.
После очистки данных был проведен корреляционный анализ с целью определения связей между различными признаками и целевой переменной — ожидаемой продолжительностью жизни (Life_expectancy).
-	Построена корреляционная матрица с использованием метода `df.corr()`.
-	Визуализирована корреляционная матрица с помощью библиотеки `seaborn` для удобного анализа взаимосвязей.
Визуализация матрицы с помощью тепловой карты (heatmap) позволяет выявить взаимосвязи между признаками и целевой переменной.

### 4.	Визуализация влияния ключевых признаков
Выделены признаки с наибольшей корреляцией с продолжительностью жизни:  
- Schooling (образование),
- GDP_per_capita (уровень дохода на душу населения),
- Adult_mortality (смертность среди взрослых),
- BMI (индекс массы тела).
  
Для каждого из них построены графики рассеяния (scatterplot), демонстрирующие взаимосвязь признака с целевой переменной.

### 5.	Масштабирование данных
Признаки масштабируются с помощью `StandardScaler` для стандартизации (среднее значение = 0, стандартное отклонение = 1), что повышает эффективность моделей. Масштабирование позволяет привести значения признаков к единому диапазону.

### 6.	Формирование гипотез
Мы предположили, что такие факторы, как уровень образования, уровень дохода на душу населения, смертность среди взрослых и индекс массы тела, оказывают наибольшее влияние на продолжительность жизни. Эти гипотезы проверяются с помощью построения прогностической модели.

## Этап 2. Обучение модели 
После того как данные исследованы и подготовлены, мы переходим к этапу построения и обучения модели. Это позволяет сделать автоматизированные прогнозы продолжительности жизни на основе признаков.  
Для прогнозирования продолжительности жизни были выбраны две модели машинного обучения: *Линейная регрессия (Linear Regression)* и *Случайный лес (Random Forest Regressor)*.

### 1.	Выбор целевой переменной и признаков
Целевая переменная - **Life_expectancy (продолжительность жизни)**.  
Признаки - экономические, социальные, медицинские данные, такие как уровень образования, ВВП на душу населения, показатели смертности и т. д.

### 2.	Разделение данных и обучение моделей
***Линейная регрессия*** - простая модель, проверяющая, насколько целевая переменная линейно зависит от признаков.  
- Разделили данные на обучающую и тестовую выборки (80% данных использовали для обучения модели, а 20% - для проверки качества).  
- Использовали функцию `LinearRegression()` из библиотеки `sklearn` для создания модели.  
- Обучили модель на обучающей выборке и сделали предсказания на тестовой выборке.

***Метод случайного леса*** - более сложная модель, которая учитывает нелинейные зависимости и взаимодействия между признаками. 
Создали модель случайного леса с параметрами:  
- 100 деревьев в лесу.
- random_state = 42 для воспроизводимости результатов.
  
Обучили модель на обучающей выборке и сделали предсказания на тестовой выборке.

### 3.	Оценка моделей
Мы использовали метрики **MSE (среднеквадратичная ошибка)** и **R² (коэффициент детерминации)**, чтобы понять, насколько точно модели предсказывают продолжительность жизни на тестовой выборке.  
- ***Среднеквадратичная ошибка (MSE)*** - показатель среднего отклонения прогнозов от реальных значений. Чем меньше ошибка, тем лучше модель.
- ***Коэффициент детерминации (R²)*** - отражает долю дисперсии целевой переменной, объясненную моделью. Значение R² близкое к 1 говорит о хорошем качестве модели.


***Линейная регрессия***  
- Среднеквадратичная ошибка (MSE): 0.05  
- Коэффициент детерминации (R²): 0.95
  
***Случайный лес***  
- Среднеквадратичная ошибка (MSE): 0,07  
- Коэффициент детерминации (R²): 0.94

Результаты показали, линейная регрессия объясняет 98.4% вариации данных и имеет меньшую ошибку, что говорит о более точных прогнозах.
Модель случайного леса также показывает высокое качество, объясняя 97.2% вариации, но имеет чуть более высокую ошибку по сравнению с линейной регрессией.

### 4.	Оценка значимости признаков
***Линейная регрессия***  
График показывает коэффициенты признаков, где видно, какие факторы наиболее сильно влияют на прогнозы.  
Наибольшее положительное влияние оказывает признак *Infant_deaths (смертность младенцев)*, *Adult_mortality (смертность взрослых)*, *Under_five_deaths (смертность детей до 5 лет)*.  
*Alcohol_consumption (потребление алкоголя)* и *Incidents_HIV (распространенность ВИЧ)* также оказывают отрицательное влияние на продолжительность жизни.  
Такие признаки, как *Schooling (уровень образования)* и *GDP_per_capita (ВВП на душу населения)*, имеют положительное влияние на продолжительность жизни, но их влияние гораздо слабее по сравнению с факторами смертности.  

***Случайный лес***  
Наибольшее влияние оказывают признаки: *Adult_mortality (смертность взрослых)*, *Infant_deaths (смертность младенцев)*.   
*Under_five_deaths (смертность детей до 5 лет)* — этот показатель также имеет высокую значимость. Он свидетельствует о том, что состояние здравоохранения для детей является ключевым фактором для увеличения продолжительности жизни.

### Выводы:  
В данной работе мы провели полный цикл анализа данных о продолжительности жизни с применением методов предобработки данных, визуализации, построения моделей машинного обучения и оценки их качества. Основной целью было выявить ключевые факторы, влияющие на продолжительность жизни, и построить модели для прогноза этого показателя.  
- Данные успешно очищены и подготовлены для анализа.
- Построены и оценены две модели (линейная регрессия и случайный лес). Модель случайного леса показала более высокое качество за счёт учёта нелинейных связей.
- Выделены ключевые признаки, оказывающие наибольшее влияние на продолжительность жизни, что может быть полезно для принятия решений в области здравоохранения и социальной политики.
- Создан дашборд с предсказанием продолжительности жизни на основе ключевых параметров.
