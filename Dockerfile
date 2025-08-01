FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "iris_week_6_app:app", "--host", "0.0.0.0", "--port", "8000"]

