import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Ваш код для создания прогнозов
df = pd.read_excel('edited-dataset.xlsx')

# Выберем нужные столбцы
selected_columns = ['Филиал', 'Регион', 'Подрядчик', 'Интеграция в сеть П']

# Уберем строки с пропущенными значениями в столбцах с датами
forecast_data = df[selected_columns].dropna(subset=['Интеграция в сеть П'])

forecast_data['Месяц'] = forecast_data['Интеграция в сеть П'].dt.to_pydatetime()

# Группируем данные по филиалу, региону, подрядчику и месяцу
grouped_data = forecast_data.groupby(['Филиал', 'Регион', 'Подрядчик', 'Месяц']).count().reset_index()

# Создаем сводную таблицу для каждого региона
pivot_data = pd.pivot_table(grouped_data, values='Интеграция в сеть П', index='Месяц', columns=['Филиал', 'Регион', 'Подрядчик'], fill_value=0)

# Прогнозирование для каждого региона
forecasts = {}
for col in pivot_data.columns:
    region_data = pd.DataFrame({'ds': pivot_data.index, 'y': pivot_data[col]})
    region_data.reset_index(inplace=True)
    region_data.rename(columns={'index': 'ds'}, inplace=True)
    
    # Инициализация и обучение модели Prophet
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(region_data)
    
    # Создание будущего DataFrame для прогноза
    future = model.make_future_dataframe(periods=12, freq='M')  # Прогноз на 12 месяцев вперед
    
    # Прогнозирование
    forecast = model.predict(future)
    
    # Сохранение прогноза для данного региона
    forecasts[col] = forecast
# Выводим графики с помощью Streamlit
st.title('Прогноз выполнения плана интеграции БС и сайтов')
st.markdown('Ключи представляют собой кортежи с тремя элементами: :violet[**филиал**] (B1, B2, B3 и т. д.), :violet[**регион**] (R1, R2, R3, R4, R5) и :violet[**подрядчик**] (ПО1, ПО2, ПО3, и т. д.).')
st.markdown(":violet[**Пример**]: ('B1', 'R2', 'ПО1')")
st.markdown('Эти ключи представляют уникальные комбинации филиала, региона и подрядчика, для которых вы выполняете прогноз. Каждый ключ возвращает данные для конкретной комбинации.')



# Выбор ключа с помощью виджета
selected_key = st.selectbox('Выберите ключ:', list(forecasts.keys()))

# Вывод графика для выбранного ключа
forecast = forecasts[selected_key]

# График прогноза
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(forecast['ds'], forecast['yhat'], label='Прогноз', color='orange')

# График фактических значений
ax.scatter(region_data['ds'], region_data['y'], label='Фактические значения', color='mediumpurple')

ax.set_title(f'Прогноз выполнения плана интеграции БС и сайтов для {selected_key}')
ax.set_xlabel('Месяц')
ax.set_ylabel('Интеграции в сеть')
ax.legend()

# Отображение графика в Streamlit
st.pyplot(fig)