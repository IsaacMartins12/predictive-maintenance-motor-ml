# MLOps — MLflow

Documentação da integração de MLflow no projeto de manutenção preditiva de motores industriais.

---

## Visão Geral

O **MLflow** é utilizado para tracking de experimentos: parâmetros, métricas, artefatos (modelos) e comparação entre runs via UI.

O projeto treina 4 modelos em paralelo (Logistic Regression, Decision Tree, Random Forest, XGBoost), cada um logado como um run separado no MLflow para comparação.

---

## Pré-requisitos

- Docker e Docker Compose instalados
- Para rodar localmente (sem Docker): `pip install -r requirements.txt`

---

## MLflow Server (Docker)

O MLflow server roda como serviço no `docker-compose.yml`:

- **UI:** [http://localhost:5000](http://localhost:5000)
- **Backend:** SQLite (persistido em volume Docker `mlflow_data`)
- **Artefatos:** Salvos em `/app/mlflow/artifacts` dentro do container

### Subir apenas o MLflow

```bash
docker compose up -d mlflow
```

### Subir toda a stack (MLflow + API + Dashboard)

```bash
docker compose up -d
```

### Verificar se está rodando

```bash
docker compose ps
docker compose logs mlflow
```

O log deve mostrar: `Listening at: http://0.0.0.0:5000`

---

## Executando um Experimento

### 1. Suba o MLflow server

```bash
docker compose up -d mlflow
```

Aguarde o container ficar healthy e confirme acessando [http://localhost:5000](http://localhost:5000).

### 2. Edite os hiperparâmetros

Os parâmetros ficam centralizados em `params.yaml`, individualizados por modelo:

```yaml
data:
  test_size: 0.2
  random_state: 42

models:
  logistic_regression:
    enabled: true
    max_iter: 10000

  decision_tree:
    enabled: true
    random_state: 42

  random_forest:
    enabled: true
    n_estimators: 100
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: 42

  xgboost:
    enabled: true
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.3
    random_state: 42
```

Para desabilitar um modelo, basta setar `enabled: false`.

### 3. Rode o treinamento

**Via Docker** (recomendado):

```bash
docker compose --profile training up training
```

> Nota: se alterou o `src/train.py` ou `requirements.txt`, adicione `--build`:
> ```bash
> docker compose --profile training up --build training
> ```

**Localmente** (com MLflow server rodando):

```bash
python src/train.py
```

### 4. Analise os resultados

- **Terminal:** métricas aparecem no stdout com resumo comparativo ao final
- **MLflow UI:** acesse [http://localhost:5000](http://localhost:5000), selecione o experimento `predictive-maintenance-motor` e compare runs lado a lado

---

## O que é Logado no MLflow (por modelo)

| Tipo | Conteúdo |
|------|----------|
| **Tags** | model_type |
| **Parâmetros** | Todos os hiperparâmetros do modelo + test_size, random_state, dataset_size, features |
| **Métricas** | accuracy, precision_weighted, recall_weighted, f1_weighted |
| **Artefatos** | modelo serializado (.pkl), feature_importance.json (quando disponível) |

---

## Modelos Salvos

Após o treinamento, os modelos ficam em `models/`:

```
models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
└── xgboost.pkl
```

A API de produção usa o `random_forest.pkl` por padrão (configurável via `MODEL_PATH` no docker-compose).

---

## Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Endereço do MLflow server |
| `MLFLOW_EXPERIMENT_NAME` | `predictive-maintenance-motor` | Nome do experimento |

---

## Fluxo Completo de Experimento

```bash
# 1. Subir MLflow
docker compose up -d mlflow

# 2. Editar hiperparâmetros
# (edite params.yaml)

# 3. Treinar (todos os modelos habilitados)
docker compose --profile training up training

# 4. Comparar na UI
# http://localhost:5000

# 5. Se satisfeito, versionar
git add params.yaml
git commit -m "experiment: rf n_estimators=200, xgb learning_rate=0.1"
```

---

## Estrutura de Arquivos

```
.
├── docs/
│   └── MLOPS.md           # Esta documentação
├── src/
│   └── train.py           # Script de treinamento multi-modelo com MLflow
├── models/
│   ├── random_forest.pkl  # Modelo em produção (API)
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   └── xgboost.pkl
├── params.yaml            # Hiperparâmetros centralizados por modelo
└── docker-compose.yml     # MLflow server + training service
```

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| MLflow UI não abre | Verifique se o container está rodando: `docker compose ps` |
| Erro "unable to open database file" | Rode `docker compose down -v` e suba novamente |
| Erro de conexão (Connection refused) | O MLflow ainda não está pronto. Espere ficar healthy: `docker compose ps` |
| Erro "Invalid Host header" | Verifique que `MLFLOW_HOST_HEADER_VALIDATION_DISABLED=true` está no environment do mlflow |
| Container training não sobe | Precisa do `--profile training`: `docker compose --profile training up training` |
| Alterações no código não refletem | Adicione `--build`: `docker compose --profile training up --build training` |
