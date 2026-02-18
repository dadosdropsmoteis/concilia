# 🏦 Conciliação Bancária — OFX × Rede

Dashboard para conciliação automática entre extrato bancário (`.ofx`) e extrato da Rede (`.xlsx`).

## ✨ Funcionalidades

- **Importação** de arquivo OFX (banco) e XLSX (Rede)
- **Agrupamento automático** das transações da Rede por Data + Bandeira + Tipo
- **Conciliação automática** cruzando os totais agrupados com os lançamentos bancários
- **Identificação** de lançamentos não conciliados (banco e Rede)
- **Resumo por bandeira e tipo** (VISA Crédito, Mastercard Débito, etc.)
- **Exportação** do resultado em Excel com formatação colorida por status

## 🚀 Como usar (online)

Acesse o link do Streamlit Cloud e faça upload dos arquivos diretamente na interface.

## 🛠️ Como rodar localmente

### Pré-requisitos
- Python 3.10 ou superior instalado

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/conciliacao-bancaria.git
cd conciliacao-bancaria

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode o app
streamlit run conciliacao_bancaria.py
```

O app abrirá automaticamente em `http://localhost:8501`

## 📁 Estrutura do projeto

```
conciliacao-bancaria/
├── conciliacao_bancaria.py   # App principal
├── requirements.txt          # Dependências Python
└── README.md                 # Este arquivo
```

## 📋 Formato esperado dos arquivos

### OFX (Extrato Bancário)
Arquivo padrão OFX exportado pelo seu banco. Os lançamentos devem conter no campo `MEMO` a bandeira e tipo, ex:
```
REDE VISA CREDITO
REDE MASTERCARD DEBITO
```

### XLSX (Extrato Rede)
Relatório exportado pelo portal da Rede. O sistema detecta automaticamente as colunas por nome. Colunas esperadas:
- Data da Transação / Data do Pagamento
- Bandeira
- Tipo de Transação / Produto
- Valor Bruto
- Valor Líquido
- NSU (opcional)

## ⚙️ Parâmetros configuráveis

| Parâmetro | Padrão | Descrição |
|---|---|---|
| Tolerância de data | 1 dia | Margem de dias para considerar match |
| Tolerância de valor | 5% | Margem percentual de diferença aceita |
| Base de comparação | Valor Líquido | Comparar OFX com valor bruto ou líquido da Rede |

## 🎨 Status da conciliação

| Status | Significado |
|---|---|
| ✅ Conciliado | Match perfeito encontrado |
| ⚠️ Conciliado c/ Divergência | Par encontrado, mas valores diferem |
| ❌ Não Conciliado (banco) | Lançamento do banco sem par na Rede |
| ❌ Não Conciliado (Rede) | Lançamento da Rede sem par no banco |
