#!/bin/bash
# Entrypoint that starts monitoring services in background, then runs training

set -e

# Start TensorBoard in background
echo "Starting TensorBoard on port 6006..."
python -m tensorboard.main \
  --logdir /app/.mutriangle_data/MuTriangle/runs \
  --host 0.0.0.0 \
  --port 6006 \
  > /tmp/tensorboard.log 2>&1 &

# Start MLflow in background
echo "Starting MLflow on port 5000..."
python -m mlflow.server \
  --backend-store-uri file:///app/.mutriangle_data/MuTriangle/mlruns \
  --host 0.0.0.0 \
  --port 5000 \
  > /tmp/mlflow.log 2>&1 &

# Wait a bit for servers to start
sleep 3

echo "Monitoring services started"
echo "  - TensorBoard: http://localhost:6006"
echo "  - MLflow: http://localhost:5000"
echo ""
echo "Starting training..."

# Run the main command (training)
exec mutriangle "$@"

