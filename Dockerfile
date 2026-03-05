FROM python:3.11-slim-bookworm

WORKDIR /app

# システム依存（必要に応じて）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python依存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 環境変数
COPY ./.env .

# アプリケーション
COPY debate_app.py .
COPY debate_app.html .
COPY login.html .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "debate_app:app", "--host", "0.0.0.0", "--port", "8000"]
