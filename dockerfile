FROM python:3.10-slim

# Java yükle
RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
