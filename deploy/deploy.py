import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Đường dẫn đến các model
rf_model_path = 'D:\\Đồ án\\DACN1\\deploy\\model\\rf_model.pkl'
dt_model_path = 'D:\\Đồ án\\DACN1\\deploy\\model\\dt_model.pkl'
knn_model_path = 'D:\\Đồ án\\DACN1\\deploy\\model\\knn_model.pkl'

# Hàm load model từ file pickle
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(data):
    data_copy = data.copy()  # Tạo một bản sao của dữ liệu đầu vào

    # Mã hóa các cột phân loại
    categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data_copy[col] = label_encoder.fit_transform(data_copy[col])

    st.write("test1 Data:", data_copy)

    # Trích xuất năm, tháng và ngày từ 'Date'
    data_copy['Year'] = pd.to_datetime(data_copy['Date']).dt.year
    data_copy['Month'] = pd.to_datetime(data_copy['Date']).dt.month
    data_copy['Day'] = pd.to_datetime(data_copy['Date']).dt.day

    st.write("test2 Data:", data_copy)

    # Tính chênh lệch nhiệt độ
    data_copy['TempDiff'] = data_copy['MaxTemp'] - data_copy['MinTemp']

    st.write("test3 Data:", data_copy)

    # Xử lý giá trị thiếu trong các cột số học
    numeric_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                    'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                    'Temp9am', 'Temp3pm', 'RISK_MM']
    imputer = SimpleImputer(strategy='mean')
    data_copy[numeric_cols] = imputer.fit_transform(data_copy[numeric_cols])

    st.write("test4 Data:", data_copy)

    # # Chuẩn hóa các đặc trưng số học
    # data_copy[numeric_cols] = (data_copy[numeric_cols] - data_copy[numeric_cols].mean()) / data_copy[numeric_cols].std()

    # Áp dụng biến đổi log cho các đặc trưng lệch
    skewed_features = ['Rainfall', 'Evaporation']
    data_copy[skewed_features] = np.log1p(data_copy[skewed_features])

    # st.write("test5 Data:", data_copy)

    # Loại bỏ cột 'Date' sau khi trích xuất thông tin
    data_copy.drop(columns=['Date'], inplace=True)

    return data_copy

def main():

    # Tiêu đề của ứng dụng
    st.title('Rain Prediction Website')
    
    # Sidebar cho nhập thông tin từ người dùng
    st.sidebar.header('Input Parameters')
    
    # Nhập các thông số từ người dùng
    date = st.sidebar.date_input('Date', value=pd.Timestamp('today'))
    location = st.sidebar.selectbox('Location', ["Bac Lieu", "Ben Tre", "Bien Hoa", "Buon Me Thuot", "Ca Mau", "Cam Pha", "Cam Ranh", "Can Tho", "Chau Doc", "Da Lat", "Ha Noi", "Hai Duong", "Hai Phong", "Hanoi", "Ho Chi Minh City", "Hoa Binh", "Hong Gai", "Hue", "Long Xuyen", "My Tho", "Nam Dinh", "Nha Trang", "Play Cu", "Phan Rang", "Phan Thiet", "Qui Nhon", "Rach Gia", "Soc Trang", "Tam Ky", "Tan An", "Tuy Hoa", "Thai Nguyen", "Thanh Hoa", "Tra Vinh", "Uong Bi", "Viet Tri", "Vinh", "Vinh Long", "Vung Tau", "Yen Bai"])
    min_temp = st.sidebar.number_input('Min Temperature (°C)', value=23.0)
    max_temp = st.sidebar.number_input('Max Temperature (°C)', value=30.0)
    rainfall = st.sidebar.number_input('Rainfall (mm)', value=6.0)
    evaporation = st.sidebar.number_input('Evaporation (mm)', value=5.0)
    sunshine = st.sidebar.number_input('Sunshine (hours)', value=5.0)
    wind_gust_dir = st.sidebar.selectbox('Wind Gust Direction', ["NNE", "ENE", "E", "WSW", "SSE", "NE", "SE", "ESE", "SW", "NNW", "SSW", "S", "W", "WNW", "NW", "N"])
    wind_dir_9am = st.sidebar.selectbox('Wind Direction 9am', ["NNE", "ENE", "E", "WSW", "SSE", "NE", "SE", "ESE", "SW", "NNW", "SSW", "S", "W", "WNW", "NW", "N"])
    wind_dir_3pm = st.sidebar.selectbox('Wind Direction 3pm', ["NNE", "ENE", "E", "WSW", "SSE", "NE", "SE", "ESE", "SW", "NNW", "SSW", "S", "W", "WNW", "NW", "N"])
    wind_gust_speed = st.sidebar.number_input('Wind Gust Speed (km/h)', value=11)
    wind_speed_9am = st.sidebar.number_input('Wind Speed 9am (km/h)', value=11)
    wind_speed_3pm = st.sidebar.number_input('Wind Speed 3pm (km/h)', value=11)
    humidity_9am = st.sidebar.slider('Humidity 9am (%)', min_value=0, max_value=100, value=50)
    humidity_3pm = st.sidebar.slider('Humidity 3pm (%)', min_value=0, max_value=100, value=50)
    pressure_9am = st.sidebar.number_input('Pressure 9am (hPa)', value=1010.0)
    pressure_3pm = st.sidebar.number_input('Pressure 3pm (hPa)', value=1010.0)
    cloud_9am = st.sidebar.number_input('Cloud 9am', value=41.0)
    cloud_3pm = st.sidebar.number_input('Cloud 3pm', value=41.0)
    rain_today = st.sidebar.selectbox('Rain Today', ['No', 'Yes'])

    # Tạo DataFrame từ thông tin nhập từ người dùng
    input_data = pd.DataFrame({
        'Date': [date],
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir_9am],
        'WindDir3pm': [wind_dir_3pm],
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_3pm],
        'Cloud9am': [cloud_9am],
        'Cloud3pm': [cloud_3pm],
        'Temp9am': [min_temp], 
        'Temp3pm': [max_temp],  
        'RainToday': [rain_today],
        'RISK_MM': [rainfall]
    })

    st.write("Input Data:", input_data)

    # Tiền xử lý dữ liệu đầu vào
    input_data = preprocess_input(input_data)

    # Chọn mô hình huấn luyện
    model_type = st.selectbox('Select Model', ['Random Forest', 'Decision Tree', 'K-Nearest Neighbors'])

    # Load model
    if model_type == 'Random Forest':
        model = load_model(rf_model_path)
    elif model_type == 'Decision Tree':
        model = load_model(dt_model_path)
    elif model_type == 'K-Nearest Neighbors':
        model = load_model(knn_model_path)

    if st.button('Predict'):

        st.write("Final Input Data for Prediction:", input_data)

        prediction = model.predict(input_data)
        st.write(f'Prediction: {prediction[0]}')

# Chạy ứng dụng
if __name__ == '__main__':
    main()
