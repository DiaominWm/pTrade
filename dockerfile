FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt* ./
RUN pip install --no-cache-dir akshare

COPY . .

CMD ["python", "src/backtest_etf_t.py"]