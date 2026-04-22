📈 TZFinance
O TZFinance é uma plataforma de investimentos focada em educação financeira, análise fundamentalista e inteligência preditiva. O objetivo é fornecer ferramentas profissionais de análise de forma simples e 100% gratuita.

🚀 O que o site proporciona?
💼 Gestão de Carteira: Monitore seus ativos, preço médio e rentabilidade total em tempo real, com interface inspirada no Investidor 10.

📰 Radar de Notícias com IA: Feed de notícias financeiras globais com análise de sentimento (Positivo/Negativo) processada por IA em tempo real.

🔍 Valuation (Preço Justo): Cálculo automático do Número de Graham e Margem de Segurança para ações, utilizando dados diretos do Yahoo Finance.

📊 Projeções Estatísticas: Simulações de Monte Carlo para prever tendências de preços com base em volatilidade histórica.

💬 Assistente Virtual: Chat inteligente integrado ao Llama-3 (Groq) para tirar dúvidas sobre conceitos de investimentos e educação financeira.

🛠️ Tecnologias Utilizadas
Backend: Python com FastAPI.

Banco de Dados: DuckDB (Rápido, analítico e local).

Frontend: HTML5, Tailwind CSS e JavaScript Puro.

IA: Llama-3 via Groq API.

Dados: Integração com a biblioteca yfinance.

Gerenciador: uv (para instalação e execução ultra-rápida).

💻 Como rodar o projeto
Instale as dependências:

Bash
uv pip install -r requirements_free.txt
Inicie o Backend:

Bash
uv run uvicorn backend:app --reload
Acesse o Frontend:
Basta abrir o arquivo frontend.html no seu navegador.
