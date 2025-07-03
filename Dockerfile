# Dockerfile

# Etapa 1: Imagem Base
# Usamos uma imagem oficial do Python, versão 3.10, do tipo "slim", que é otimizada em tamanho.
FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner. Todos os comandos seguintes serão executados a partir daqui.
WORKDIR /app

# Define variáveis de ambiente para otimizar o Poetry e o Python
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

# Etapa 2: Instalar Dependências do Sistema
# CORREÇÃO: A imagem 'slim' não vem com a biblioteca libgomp1, que é necessária
# para o processamento paralelo do LightGBM. Nós a instalamos aqui.
RUN apt-get update && apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Etapa 3: Instalar o Poetry
# Instalamos o Poetry, que é o nosso gerenciador de dependências.
RUN pip install poetry

# Etapa 4: Instalar as Dependências do Projeto
# Copiamos apenas os arquivos de definição de dependências primeiro.
# Isso aproveita o cache do Docker: se esses arquivos não mudarem, o Docker não reinstalará tudo.
COPY pyproject.toml poetry.lock ./

# Instala apenas as dependências de produção, ignorando as de desenvolvimento (como pytest).
RUN poetry install --no-root --without dev

# Etapa 5: Copiar o Código da Aplicação
# Agora, copiamos todo o resto do projeto para dentro do contêiner.
# Os arquivos listados no .dockerignore não serão copiados.
COPY . .

# Etapa 6: Expor a Porta
# Informa ao Docker que a aplicação dentro do contêiner estará escutando na porta 8000.
EXPOSE 8000

# Etapa 7: Comando de Execução
# Define o comando que será executado quando o contêiner iniciar.
# Ele inicia o servidor Uvicorn para rodar nossa API FastAPI.
# O host 0.0.0.0 é necessário para que a API seja acessível de fora do contêiner.
CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
