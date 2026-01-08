:: MIAA-2025-2026
:: by lufer

docker build -t prediction-api .
docker run -d --name miaa-fastapi -p 8000:8000 prediction-api