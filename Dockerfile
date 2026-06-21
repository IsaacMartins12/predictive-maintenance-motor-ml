FROM python:3.10.6

WORKDIR /app

# build-essential é necessário pra compilar dependências como scikit-learn/xgboost/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501
