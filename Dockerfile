# ==========================================
# 阶段 1：构建阶段 (Builder)
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /build

# 1. 安装编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 将依赖编译并打包成 wheel 文件，存入 /build/wheels 目录
RUN pip wheel --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --wheel-dir /build/wheels \
    akshare baostock tushare yfinance pandas numpy scipy \
    openpyxl xlrd statsmodels scikit-learn empyrical-reloaded \
    requests tqdm python-dateutil loguru

# ==========================================
# 阶段 2：运行阶段 (Runtime)
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# 1. 从 builder 阶段把编译好的 wheels 复制过来
COPY --from=builder /build/wheels /wheels

# 2. 安装这些 wheel（因为已经是编译好的，所以极快，且不需要 gcc）
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# 3. 复制代码
COPY . .

CMD ["bash"]