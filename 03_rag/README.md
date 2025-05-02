# RAG LangGraph

Este projeto RAG baseado em [LangGraph](https://github.com/langchain-ai/langgraph) e [Langchain-LiteLLM](https://docs.litellm.ai/docs/), utilizando o gerenciador de pacotes [uv](https://docs.astral.sh/uv/reference/cli/) para dependências.

## 📦 Pré-requisitos

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) instalado (`pip install uv`)

## 🚀 Instalação

1. Clone este repositório e acesse o diretório do projeto:

```bash
git clone https://github.com/wandealves/introducao_langGraph.git
cd introducao_langGraph/01_simples_chat_multiplos_agentes
```

2. Inicialize o ambiente com `uv`:

```bash
uv init
```

3. Adicione as dependências necessárias:

```bash
uv add langgraph
uv add langchain_community
uv add langchain-openai
uv add python-dotenv
uv add langchain-litellm
uv add chromadb
uv add beautifulsoup4
uv add tavily-python
uv add langchainhub
uv add langchain
```

4. Certifique-se de configurar um arquivo `.env` com as variáveis de ambiente necessárias (como chave de API, se aplicável).

## 🧠 Execução do Chatbot

Para iniciar o chatbot, use:

```bash
uv run chat_bot.py
```

## 📝 Licença

Este projeto está licenciado sob a licença MIT.
