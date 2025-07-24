FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN dvc pull && \
    rm -rf .dvc .dvcignore .git && \
    dvc config core.analytics false

EXPOSE 8000

CMD ["uvicorn", "iris_week_6_app:app", "--host", "0.0.0.0", "--port", "8000"]

