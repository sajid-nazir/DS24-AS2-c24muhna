FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY mlruns /app/mlruns
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
