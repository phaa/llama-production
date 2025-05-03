#!/bin/bash

echo "Construindo os containers..."
docker compose build

echo "Subindo os serviços API, Prometheus e Grafana..."
docker compose up -d

echo "Verificando status dos containers..."
docker compose ps

echo ""
echo "Serviços inicializados:"
echo "  - API LLama:      http://localhost:8000"
echo "  - Prometheus:     http://localhost:9090"
echo "  - Grafana:        http://localhost:3000 (login: admin / senha: admin)"
echo ""
