# pTrade
programming tarde

# running
git clone
cd pTrade
docker build -t ptrade-py312 . # new env run once

# 跑 backtest_etf_t.py
docker run --rm -v "$(pwd):/app" ptrade-py312 python src/backtest_etf_t.py

# 跑 backtest.py
docker run --rm -v "$(pwd):/app" ptrade-py312 python src/backtest.py

# 带参数
docker run --rm -v "$(pwd):/app" ptrade-py312 python src/backtest_etf_t.py --start 2024-01-01

# 进容器手动操作（如果 CMD 是 bash，或不带命令直接 bash）
docker run --rm -it -v "$(pwd):/app" ptrade-py312 bash

# 使用脚本运行
./run.sh src/backtest.py
./run.sh src/backtest_etf_t.py