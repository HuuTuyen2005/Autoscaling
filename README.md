# Freifin - Autoscaling 
Thành viên:
1. Nguyễn Hữu Tuyên (Trưởng nhóm)
2. Trần Hà Linh
3. Đặng Huy Phúc
4. Trịnh Lê Bách

## 1. Prerequisites
Trước khi chạy dự án, cần đảm bảo các yêu cầu sau:
- **Python**: >= 3.10
- **RAM tối thiểu**: 8GB (khuyến nghị 16GB)
- **Hệ điều hành**: Windows / Linux / macOS

## 2. Installation
Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
pip install ipykernel  # Bắt buộc để chạy Notebook trong môi trường ảo
```

## 3. How to Run
- Bước 0: Trước khi chạy cần giải nén file data.zip
- Bước 1: Tiền xử lý dữ liệu
Chuyển dữ liệu log thô thành dữ liệu metrics để huấn luyện và mô phỏng autoscaling.
```bash
python src/ingest.py
```
Sau đó chạy notebook: `01_eda.ipynb`

- Bước 2: Train model dự đoán
(Có thể chạy 1 hoặc nhiều mô hình để so sánh)
Chạy các notebook sau:
+ Prophet: `02_prophet_forecasting.ipynb`
+ Xgboost: `03_xgb_forecasting.ipynb`
+ LSTM: `04_lstm_forecasting.ipynb`

- Bước 3: Mô phỏng Autoscaling
Chạy notebook: `05_autoscaling_simulation.ipynb`

- Bước 4: Demo
```bash
streamlit run dashboard/app.py
```

## 4. Project Structure
```text
AUTOSCALING-ANALYSIS/
├── dashboard/
│   └── app.py              # File chính chạy giao diện Dashboard
├── data/
│   ├── processed/          # Dữ liệu đã làm sạch và xử lý đặc trưng
│   └── raw/                # Dữ liệu gốc chưa qua xử lý
├── notebooks/              # Các bước thực hiện chi tiết trên Jupyter Notebook
│   ├── 01_eda.ipynb        # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_prophet_forecasting.ipynb  # Huấn luyện & dự báo với Facebook Prophet
│   ├── 03_xgb_forecasting.ipynb      # Huấn luyện & dự báo với XGBoost
│   ├── 04_lstm_forecasting.ipynb     # Huấn luyện & dự báo với mạng LSTM
│   └── 05_autoscaling_simulation.ipynb # Mô phỏng cơ chế Autoscaling thực tế
├── src/                    # Mã nguồn triển khai hệ thống
│   ├── models/             # Định nghĩa kiến trúc các mô hình
│   │   ├── lstm_model.py
│   │   ├── prophet_model.py
│   │   └── xgb_model.py
│   ├── autoscaler.py       # Logic điều hướng và tính toán Scale In/Out
│   ├── data_loader.py      # Module nạp dữ liệu từ metrics_full.csv và xgb_forecast_5m.csv
│   ├── features.py         # Xử lý biến đổi và trích xuất đặc trưng
│   ├── ingest.py           # Tiếp nhận và chuẩn hóa dữ liệu đầu vào
│   ├── metrics.py          # Tính toán các chỉ số đo lường (RMSE, MSE, MAE, MAPE)
│   └── split.py            # Chia tập dữ liệu Train/Test/Validation
├── venv/                   # Môi trường ảo của dự án
├── README.md               # Tài liệu hướng dẫn sử dụng
└── requirements.txt        # Danh sách các thư viện Python cần thiết