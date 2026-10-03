# Extrator de NFS-e

[English](README.md) · [CI](https://github.com/JeandePaula/nfse-yolov8/actions)

API pequena com FastAPI para localizar campos de NFS-e com YOLO e ler seus textos com Tesseract em português. Processa imagens e PDFs localmente, sem banco de dados, OCR externo ou fila de tarefas.

## Executar no Linux / WSL

Requer Python 3.12–3.14 e [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/JeandePaula/nfse-yolov8.git
cd nfse-yolov8
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por libgl1 libglib2.0-0
uv sync --locked
uv run uvicorn nfse.api:app --host 127.0.0.1 --port 8000
```

Acesse a [documentação interativa](http://127.0.0.1:8000/docs). Os comandos `uv run python main.py` e `uv run uvicorn main:app` continuam disponíveis. O modelo carrega uma vez na inicialização. Pesos ou idiomas de OCR ausentes impedem a inicialização com um erro explícito.

## Docker

```bash
docker build -t nfse .
docker run --rm -p 127.0.0.1:8000:8000 nfse
```

A imagem inclui o checkpoint, PyTorch para CPU, PDFium e Tesseract em português. Executa com usuário sem privilégios e verifica `/health`. Para configurar, copie `.env.example` para `.env` e acrescente `--env-file .env`. Caminhos personalizados do modelo devem existir dentro do contêiner; use um volume somente de leitura para disponibilizar outros pesos.

## API

```bash
curl --fail-with-body http://127.0.0.1:8000/extract -F 'file=@nota.pdf'
curl --fail-with-body http://127.0.0.1:8000/extract -F 'file=@nota.png'
curl http://127.0.0.1:8000/health
```

`POST /extract` preserva cada detecção e separa as páginas do PDF. Exemplo ilustrativo:

```json
{
  "pages": [
    {
      "page": 1,
      "width": 1240,
      "height": 1754,
      "fields": [
        {
          "label": "nf-numero",
          "text": "00000001",
          "confidence": 0.94,
          "bbox": [910, 80, 1150, 135]
        }
      ]
    }
  ]
}
```

Os rótulos mantêm os nomes originais do checkpoint, inclusive as grafias presentes em `data.yaml`. As coordenadas são medidas em pixels da página renderizada, após a orientação EXIF. A confiança se refere à **detecção**, não à precisão do OCR. Páginas sem detecções retornam `fields: []`.

`POST /extract_data` mantém o contrato antigo `{rotulo: texto}`. Em campos repetidos, continua prevalecendo a última detecção, inclusive entre páginas. Para novas integrações, use `/extract`.

Formatos: JPEG, PNG, WebP, TIFF e BMP de um quadro, além de PDF com várias páginas. O formato é identificado pelo conteúdo, independentemente do nome ou MIME declarado. PDFs criptografados, arquivos corrompidos e formatos incompatíveis geram erro. Converta imagens com múltiplos quadros para PDF antes do envio.

| Status | Significado |
| --- | --- |
| 400 | Arquivo vazio ou multipart malformado |
| 413 | Limite de arquivo, requisição, páginas ou pixels excedido |
| 415 | Formato incompatível ou não reconhecido |
| 422 | Documento ilegível ou campo de upload ausente |
| 503 | Processo ocupado; repetir após o intervalo `Retry-After` |
| 504 | Tempo de OCR excedido |

## Configuração

Copie `.env.example` para `.env`. Variáveis de ambiente têm prioridade sobre o arquivo.

| Variável | Padrão | Finalidade |
| --- | --- | --- |
| `NFSE_MODEL_PATH` | `best-nfse.pt` do repositório | Checkpoint local confiável |
| `NFSE_DEVICE` | `cpu` | Dispositivo de inferência |
| `NFSE_CONFIDENCE` | `0.25` | Limiar de detecção |
| `NFSE_IMAGE_SIZE` | `640` | Entrada do YOLO, múltiplo de 32 |
| `NFSE_OCR_LANGUAGE` | `por` | Idiomas instalados no Tesseract |
| `NFSE_OCR_TIMEOUT` | `15` | Segundos por região detectada |
| `NFSE_MAX_FILE_MB` | `5` | Limite do arquivo em MiB |
| `NFSE_MAX_PAGES` | `10` | Máximo de páginas do PDF |
| `NFSE_MAX_PIXELS` | `20000000` | Máximo de pixels por página |
| `NFSE_MAX_DETECTIONS` | `100` | Máximo de detecções por página |
| `NFSE_PDF_DPI` | `150` | Resolução de renderização do PDF |

A requisição tem o limite do arquivo mais 64 KiB para o multipart, inclusive em uploads enviados em blocos. PDFs são renderizados uma página por vez. Detecção e OCR executam fora do loop assíncrono da API. Cada processo atende uma extração por vez para proteger o estado do YOLO e do PDFium; extrações simultâneas recebem 503. Mais workers do Uvicorn carregam cópias independentes do modelo e consomem mais RAM.

A API não tem autenticação própria: use rede confiável ou proxy autenticado na implantação. O parser pode criar arquivos temporários durante o upload, fechados ao terminar a requisição. A aplicação não grava notas anotadas nem os textos extraídos; a política de logs de acesso/proxy depende da implantação. A orientação EXIF é corrigida; rotação arbitrária de páginas digitalizadas não é. A antiga checagem por palavras-chave foi removida porque não validava de forma confiável a orientação ou a autenticidade da nota.

## Modelo e treinamento

O `best-nfse.pt` incluído é o checkpoint YOLOv8 original. Seu treinamento limitado ainda afeta a precisão e pode gerar falsas detecções até em páginas em branco. A reconstrução atualiza o software; não melhora automaticamente os pesos. Como não há dataset rotulado no repositório, não foi realizado novo treinamento nem aferida a precisão. `result/` e `output_image.png` são registros históricos, não resultados desta versão.

O treinamento novo usa [YOLO26 nano](https://docs.ultralytics.com/models/yolo26/) por padrão. Prepare conjuntos distintos de treino e validação:

```text
datasets/nfse/
  images/train/  images/val/
  labels/train/  labels/val/
```

Cada linha dos rótulos YOLO contém `class_id x_center y_center width height`, normalizados de 0 a 1. O `data.yaml` preserva os 38 IDs originais. Seu `path` segue as regras de resolução de datasets do Ultralytics; use caminho absoluto para um dataset em outro local. Não repita as amostras de treino na validação.

```bash
uv run python train.py --data data.yaml --epochs 100 --device cpu
# Para continuar com a arquitetura existente:
uv run python train.py --model best-nfse.pt --data data.yaml
```

O Ultralytics baixa os pesos-base no primeiro treinamento. Depois de avaliar o `best.pt` gerado em notas separadas do treino, aponte `NFSE_MODEL_PATH` para ele. A instalação travada usa CPU. Para CUDA, altere as duas fontes PyTorch no `pyproject.toml` conforme o [seletor oficial](https://pytorch.org/get-started/locally/), regenere `uv.lock` e selecione `--device 0` / `NFSE_DEVICE=0`.

## Desenvolvimento e manutenção

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
# Testes rápidos, sem carregar os pesos reais ou invocar OCR:
uv run pytest -q -m 'not integration'
```

Os testes de integração exigem Tesseract em português e DejaVu Sans (`fonts-dejavu-core` no Ubuntu). Exercitam o checkpoint real, inferência em imagem/PDF, OCR e inicialização da aplicação. Verificam funcionamento, não precisão de extração fiscal. A CI executa em Python 3.12 e 3.14 e constrói/inicia o contêiner Docker.

Em `nfse/`, `api.py` concentra HTTP e inicialização; `config.py`, configurações; `documents.py`, leitura limitada; `service.py`, detecção/OCR; `schemas.py`, respostas. `train.py` é o comando independente de treinamento. Dependências ficam no `pyproject.toml` e a resolução completa no `uv.lock`. Para atualizar: `uv lock --upgrade`, `uv sync --locked` e os testes acima. O Dependabot verifica atualizações mensalmente.

Versões resolvidas a partir das versões estáveis atuais em 03/10/2026: FastAPI 0.142.2, Ultralytics 8.4.172, PyTorch 2.14.1, NumPy 2.5.3, OpenCV 5.0.0.93 e bindings PDFium 5.13.0. O lockfile é a referência exata.

## Licença

A licença MIT original permanece em [LICENSE](LICENSE). Dependências e artefatos de modelo têm termos próprios; consulte o [licenciamento Ultralytics](https://docs.ultralytics.com/help/license/).
