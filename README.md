# pTrade
programming tarde

# running
git clone URL
cd pTrade
docker build -t ptrade-py312 .     # 只需一次
docker run --rm -v "$(pwd):/app" ptrade-py312