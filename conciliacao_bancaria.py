"""
Dashboard de Conciliação Bancária
OFX (Extrato Bancário) x XLSX (Extrato Rede)
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
import re

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Conciliação Bancária",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Conciliação Bancária")
st.caption("OFX (Extrato Banco) × XLS (Intermediadora / Rede)")

# ─────────────────────────────────────────────
# FUNÇÕES DE PARSE
# ─────────────────────────────────────────────

def parse_ofx(file_bytes: bytes) -> pd.DataFrame:
    """Parser manual do OFX. Extrai: data, valor, descrição, fitid."""
    content = file_bytes.decode("latin-1", errors="ignore")
    transactions = []

    pattern = re.compile(r"<STMTTRN>(.*?)</STMTTRN>", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(content):
        block = match.group(1)

        def get_field(tag):
            m = re.search(rf"<{tag}>(.*?)(?=<|\Z)", block, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        dtposted = get_field("DTPOSTED")
        trnamt   = get_field("TRNAMT")
        memo     = get_field("MEMO") or get_field("NAME")
        fitid    = get_field("FITID")
        trntype  = get_field("TRNTYPE")

        try:
            date = datetime.strptime(dtposted[:8], "%Y%m%d").date()
        except Exception:
            continue
        try:
            valor = float(trnamt.replace(",", "."))
        except Exception:
            continue

        transactions.append({
            "data": date, "valor_ofx": valor,
            "memo": memo, "fitid": fitid, "tipo_lancamento": trntype,
        })

    return pd.DataFrame(transactions)


def detectar_bandeira_tipo(memo: str):
    """Extrai bandeira e tipo (CREDITO/DEBITO) do memo do OFX."""
    memo_upper = memo.upper()

    bandeiras = ["VISA", "MASTERCARD", "MASTER", "ELO", "AMEX",
                 "AMERICAN EXPRESS", "HIPERCARD", "HIPER", "CABAL", "DINERS"]
    tipos = ["CREDITO", "CRÉDITO", "DEBITO", "DÉBITO", "CREDIT", "DEBIT"]

    bandeira, tipo = "OUTROS", "OUTROS"

    for b in bandeiras:
        if b in memo_upper:
            bandeira = "MASTERCARD" if b == "MASTER" else b
            bandeira = "AMEX" if b == "AMERICAN EXPRESS" else bandeira
            bandeira = "HIPERCARD" if b == "HIPER" else bandeira
            break

    for t in tipos:
        if t in memo_upper:
            tipo = "CREDITO" if t in ["CREDITO", "CRÉDITO", "CREDIT"] else "DEBITO"
            break

    return bandeira, tipo


def parse_intermediadora_xls(file) -> pd.DataFrame:
    """
    Parser específico para o relatório TSV da intermediadora (extensão .xls).
    Colunas mapeadas:
      DATA VENDA, BANDEIRA, TRANSAÇÃO, VALOR BRUTO, VALOR LÍQUIDO,
      TAXA FINAL (R$), TAXA FINAL (%), STATUS VENDA, C.V., ESTABELECIMENTO
    """
    def to_float(s):
        if pd.isna(s) or str(s).strip() in ("", "-", "nan"): return 0.0
        s = str(s).strip()
        # Remove conteúdo entre parênteses: "(1.26%)" etc.
        s = re.sub(r"\(.*?\)", "", s)
        # Extrai apenas sequências numéricas com vírgula/ponto
        nums = re.findall(r"[\d.,]+", s)
        if not nums: return 0.0
        n = nums[0].replace(".", "").replace(",", ".")
        try: return float(n)
        except: return 0.0

    # Arquivo é TSV com encoding latin-1
    try:
        content = file.read()
        if isinstance(content, bytes):
            text = content.decode("latin-1", errors="ignore")
        else:
            text = content
        df_raw = pd.read_csv(
            io.StringIO(text),
            sep="\t",
            dtype=str,
            encoding=None,
        )
    except Exception as e:
        raise ValueError(f"Erro ao ler arquivo da intermediadora: {e}")

    df_raw.columns = [c.strip() for c in df_raw.columns]

    # Mapeamento direto das colunas conhecidas
    df = pd.DataFrame()
    df["data"]          = pd.to_datetime(df_raw["DATA VENDA"], dayfirst=True, errors="coerce").dt.date
    df["bandeira"]      = df_raw["BANDEIRA"].astype(str).str.strip().str.upper()
    df["tipo"]          = df_raw["TRANSAÇÃO"].astype(str).str.strip()
    df["valor_bruto"]   = df_raw["VALOR BRUTO"].apply(to_float)
    df["valor_liquido"] = df_raw["VALOR LÍQUIDO"].apply(to_float)
    df["taxa_final"]    = df_raw["TAXA FINAL (R$)"].apply(to_float)
    df["taxa_pct"]      = df_raw["TAXA FINAL (%)"].apply(to_float)
    df["status"]        = df_raw.get("STATUS VENDA", pd.Series([""] * len(df_raw))).astype(str)
    df["cv"]            = df_raw.get("C.V.", pd.Series([""] * len(df_raw))).astype(str)
    df["estabelecimento"] = df_raw.get("ESTABELECIMENTO", pd.Series([""] * len(df_raw))).astype(str)

    # Normaliza bandeira
    df["bandeira"] = df["bandeira"].replace({
        "MASTER":           "MASTERCARD",
        "MC":               "MASTERCARD",
        "AMERICAN EXPRESS": "AMEX",
        "AX":               "AMEX",
        "VI":               "VISA",
    })

    # Normaliza tipo → CREDITO / DEBITO
    df["tipo_norm"] = df["tipo"].apply(
        lambda x: "CREDITO" if "CRÉD" in x.upper() or "CRED" in x.upper()
                  else ("DEBITO" if "DÉB" in x.upper() or "DEB" in x.upper() else x.upper())
    )

    # Remove linhas sem data
    df = df.dropna(subset=["data"])
    df = df[df["valor_bruto"] > 0]

    return df


def agrupar_rede(df_rede: pd.DataFrame) -> pd.DataFrame:
    """Agrupa transações por data + bandeira + tipo, somando valores e contando transações."""
    group_cols = [c for c in ["data", "bandeira", "tipo_norm"] if c in df_rede.columns]
    agg = {}
    if "valor_bruto"   in df_rede.columns: agg["valor_bruto"]   = "sum"
    if "valor_liquido" in df_rede.columns: agg["valor_liquido"] = "sum"
    if "taxa_final"    in df_rede.columns: agg["taxa_final"]    = "sum"
    agg["cv"] = "count"

    df_group = df_rede.groupby(group_cols, as_index=False).agg(agg)
    df_group = df_group.rename(columns={"cv": "qtd_transacoes"})
    return df_group


def conciliar(df_ofx: pd.DataFrame, df_rede_grupo: pd.DataFrame,
              tolerancia_dias: int = 1, tolerancia_valor: float = 0.05) -> pd.DataFrame:
    """Cruza OFX com Rede agrupado por data + bandeira + tipo + valor líquido."""
    df_ofx = df_ofx.copy()
    bandeira_tipo = df_ofx["memo"].apply(lambda m: pd.Series(detectar_bandeira_tipo(m)))
    df_ofx[["bandeira_ofx", "tipo_ofx"]] = bandeira_tipo

    resultados = []
    rede_usados = set()
    rede_rows = df_rede_grupo.reset_index(drop=True)

    for i_ofx, row_ofx in df_ofx.iterrows():
        melhor_match = None
        melhor_diff  = float("inf")

        for i_rede, row_rede in rede_rows.iterrows():
            if i_rede in rede_usados:
                continue

            # Filtro bandeira
            if (row_ofx["bandeira_ofx"] not in ("OUTROS", "") and
                    row_rede.get("bandeira", "") not in ("OUTROS", "") and
                    row_ofx["bandeira_ofx"] != row_rede.get("bandeira", "")):
                continue

            # Filtro tipo
            if (row_ofx["tipo_ofx"] not in ("OUTROS", "") and
                    row_rede.get("tipo_norm", "") not in ("OUTROS", "") and
                    row_ofx["tipo_ofx"] != row_rede.get("tipo_norm", "")):
                continue

            # Filtro data
            diff_dias = abs((row_ofx["data"] - row_rede["data"]).days)
            if diff_dias > tolerancia_dias:
                continue

            # Filtro valor
            val_ofx  = abs(row_ofx["valor_ofx"])
            val_rede = abs(row_rede.get("valor_liquido", row_rede.get("valor_bruto", 0)))
            if val_rede == 0:
                continue
            diff_perc = abs(val_ofx - val_rede) / val_rede
            if diff_perc > tolerancia_valor:
                continue

            score = diff_dias + diff_perc
            if score < melhor_diff:
                melhor_diff  = score
                melhor_match = i_rede

        row_rede_matched = {}
        if melhor_match is not None:
            rede_usados.add(melhor_match)
            row_rede_matched = rede_rows.loc[melhor_match].to_dict()
            diff_valor = abs(row_ofx["valor_ofx"]) - abs(row_rede_matched.get("valor_liquido", 0))
            status = "⚠️ Conciliado c/ Divergência" if abs(diff_valor) > 0.01 else "✅ Conciliado"
        else:
            status     = "❌ Não Conciliado (banco)"
            diff_valor = None

        resultados.append({
            "Status":             status,
            "Data OFX":           row_ofx["data"],
            "Valor OFX":          row_ofx["valor_ofx"],
            "Memo OFX":           row_ofx["memo"],
            "Bandeira OFX":       row_ofx["bandeira_ofx"],
            "Tipo OFX":           row_ofx["tipo_ofx"],
            "Data Rede":          row_rede_matched.get("data", ""),
            "Bandeira Rede":      row_rede_matched.get("bandeira", ""),
            "Tipo Rede":          row_rede_matched.get("tipo_norm", ""),
            "Valor Bruto Rede":   row_rede_matched.get("valor_bruto", ""),
            "Valor Líq. Rede":    row_rede_matched.get("valor_liquido", ""),
            "Qtd Transações":     row_rede_matched.get("qtd_transacoes", ""),
            "Diferença (R$)":     diff_valor,
        })

    # Não conciliados da Rede
    for i_rede, row_rede in rede_rows.iterrows():
        if i_rede not in rede_usados:
            resultados.append({
                "Status":             "❌ Não Conciliado (Rede)",
                "Data OFX":           "", "Valor OFX": "", "Memo OFX": "",
                "Bandeira OFX":       "", "Tipo OFX": "",
                "Data Rede":          row_rede.get("data", ""),
                "Bandeira Rede":      row_rede.get("bandeira", ""),
                "Tipo Rede":          row_rede.get("tipo_norm", ""),
                "Valor Bruto Rede":   row_rede.get("valor_bruto", ""),
                "Valor Líq. Rede":    row_rede.get("valor_liquido", ""),
                "Qtd Transações":     row_rede.get("qtd_transacoes", ""),
                "Diferença (R$)":     None,
            })

    return pd.DataFrame(resultados)


def exportar_excel(df_result: pd.DataFrame, df_rede_orig: pd.DataFrame,
                   df_rede_grupo: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_result.to_excel(writer,    sheet_name="Conciliação",    index=False)
        df_rede_grupo.to_excel(writer, sheet_name="Rede - Agrupado", index=False)
        df_rede_orig.to_excel(writer,  sheet_name="Rede - Detalhe",  index=False)

        wb = writer.book
        fmt_h    = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1})
        fmt_ok   = wb.add_format({"bg_color": "#C6EFCE"})
        fmt_warn = wb.add_format({"bg_color": "#FFEB9C"})
        fmt_err  = wb.add_format({"bg_color": "#FFC7CE"})

        ws = writer.sheets["Conciliação"]
        for col_num, col_name in enumerate(df_result.columns):
            ws.write(0, col_num, col_name, fmt_h)
            ws.set_column(col_num, col_num, 22)
        for row_num in range(1, len(df_result) + 1):
            status = str(df_result.iloc[row_num - 1]["Status"])
            fmt = fmt_ok if "✅" in status else (fmt_warn if "⚠️" in status else fmt_err)
            ws.set_row(row_num, None, fmt)

    return output.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Importar Arquivos")
    file_ofx  = st.file_uploader("Extrato Bancário (.ofx)", type=["ofx", "OFX"])
    file_rede = st.file_uploader("Extrato Intermediadora (.xls)", type=["xls", "xlsx", "tsv", "txt"])

    st.divider()
    st.header("⚙️ Parâmetros")
    tolerancia_dias  = st.slider("Tolerância de data (dias)", 0, 5, 1)
    tolerancia_valor = st.slider("Tolerância de valor (%)",   0, 10, 5) / 100
    col_valor_rede   = st.radio("Comparar OFX com:", ["Valor Líquido", "Valor Bruto"])

    st.divider()
    st.markdown("**Legenda:**")
    st.markdown("✅ Conciliado | ⚠️ Divergência | ❌ Não Conciliado")

# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────
if not file_ofx or not file_rede:
    st.info("👈 Importe o arquivo OFX e o extrato da Rede para iniciar.")
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
**Arquivo OFX** — Extrato bancário exportado pelo seu banco.  
Contém créditos agrupados por bandeira e tipo (ex: `REDE VISA CREDITO`).

**Arquivo XLS** — Relatório da intermediadora (formato TSV com extensão .xls).  
Colunas utilizadas: `DATA VENDA`, `BANDEIRA`, `TRANSAÇÃO`, `VALOR BRUTO`, `VALOR LÍQUIDO`, `TAXA FINAL (R$)`.

**Lógica de conciliação:**
- Transações da intermediadora são **agrupadas** por Data + Bandeira + Tipo e somadas
- O total líquido é comparado com cada lançamento do OFX
- Tolerâncias de data e valor são configuráveis na barra lateral

**Dica:** Use *Tolerância de valor 5%* para absorver pequenas diferenças de taxa.
        """)
    st.stop()

with st.spinner("Processando arquivos..."):
    try:
        df_ofx = parse_ofx(file_ofx.read())
        if df_ofx.empty:
            st.error("Nenhuma transação encontrada no OFX."); st.stop()
    except Exception as e:
        st.error(f"Erro ao ler OFX: {e}"); st.stop()

    try:
        df_rede_orig = parse_intermediadora_xls(file_rede)
        if df_rede_orig.empty:
            st.error("Nenhuma transação encontrada no arquivo da intermediadora."); st.stop()
    except Exception as e:
        st.error(f"Erro ao ler arquivo da intermediadora: {e}"); st.stop()

if col_valor_rede == "Valor Bruto" and "valor_bruto" in df_rede_orig.columns:
    df_rede_orig["valor_liquido"] = df_rede_orig["valor_bruto"]

df_rede_grupo = agrupar_rede(df_rede_orig)
df_result     = conciliar(df_ofx, df_rede_grupo, tolerancia_dias, tolerancia_valor)

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
total       = len(df_result)
conciliados = df_result["Status"].str.contains("✅").sum()
divergentes = df_result["Status"].str.contains("⚠️").sum()
nao_conc    = df_result["Status"].str.contains("❌").sum()
pct = lambda n: f"{n/total*100:.1f}%" if total else "0%"

c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Total Lançamentos",  total)
c2.metric("✅ Conciliados",        f"{conciliados} ({pct(conciliados)})")
c3.metric("⚠️ Com Divergência",    f"{divergentes} ({pct(divergentes)})")
c4.metric("❌ Não Conciliados",    f"{nao_conc} ({pct(nao_conc)})")

# Valor total OFX vs Rede
val_ofx_total  = df_ofx["valor_ofx"].apply(lambda v: v if v > 0 else 0).sum()
val_rede_total = df_rede_orig["valor_liquido"].sum() if "valor_liquido" in df_rede_orig.columns else 0
val_taxa_total = df_rede_orig["taxa_final"].sum()   if "taxa_final"    in df_rede_orig.columns else 0

cv1, cv2, cv3, cv4 = st.columns(4)
cv1.metric("💰 Total Créditos OFX",    f"R$ {val_ofx_total:,.2f}")
cv2.metric("💳 Total Bruto Intermediadora", f"R$ {df_rede_orig['valor_bruto'].sum():,.2f}" if "valor_bruto" in df_rede_orig.columns else "—")
cv3.metric("🏦 Total Líquido Intermediadora", f"R$ {val_rede_total:,.2f}")
cv4.metric("📊 Diferença OFX × Líquido", f"R$ {val_ofx_total - val_rede_total:,.2f}")

st.divider()

# ─────────────────────────────────────────────
# RESUMO POR BANDEIRA E TIPO
# ─────────────────────────────────────────────
st.subheader("📊 Resumo por Bandeira e Tipo (Intermediadora)")

tab1, tab2 = st.tabs(["Agrupado", "Detalhe por Transação"])

with tab1:
    df_resumo = df_rede_grupo.copy()
    rename = {
        "bandeira":        "Bandeira",
        "tipo_norm":       "Tipo",
        "valor_bruto":     "Valor Bruto (R$)",
        "valor_liquido":   "Valor Líquido (R$)",
        "taxa_final":      "Total Taxas (R$)",
        "qtd_transacoes":  "Qtd Transações",
    }
    df_resumo = df_resumo.rename(columns={k: v for k, v in rename.items() if k in df_resumo.columns})
    fmt_cols = {c: "R$ {:,.2f}" for c in ["Valor Bruto (R$)", "Valor Líquido (R$)", "Total Taxas (R$)"] if c in df_resumo.columns}
    st.dataframe(df_resumo.style.format(fmt_cols), use_container_width=True)

with tab2:
    cols_show = [c for c in ["data", "bandeira", "tipo_norm", "tipo", "valor_bruto",
                              "taxa_final", "taxa_pct", "valor_liquido", "cv", "estabelecimento", "status"]
                 if c in df_rede_orig.columns]
    rename_det = {
        "data": "Data", "bandeira": "Bandeira", "tipo_norm": "Tipo",
        "tipo": "Descrição Transação", "valor_bruto": "Valor Bruto",
        "taxa_final": "Taxa (R$)", "taxa_pct": "Taxa (%)",
        "valor_liquido": "Valor Líquido", "cv": "C.V.",
        "estabelecimento": "Estabelecimento", "status": "Status",
    }
    st.dataframe(
        df_rede_orig[cols_show].rename(columns=rename_det),
        use_container_width=True, height=300
    )

st.divider()

# ─────────────────────────────────────────────
# RESULTADO DA CONCILIAÇÃO
# ─────────────────────────────────────────────
st.subheader("🔍 Resultado da Conciliação")

status_opcoes = df_result["Status"].unique().tolist()
filtro = st.multiselect("Filtrar por status:", options=status_opcoes, default=status_opcoes)
df_filtrado = df_result[df_result["Status"].isin(filtro)]

def fmt_brl(v):
    try:
        if v == "" or pd.isna(v): return ""
        return f"R$ {float(v):,.2f}"
    except: return v

df_display = df_filtrado.copy()
for c in ["Valor OFX", "Valor Bruto Rede", "Valor Líq. Rede", "Diferença (R$)"]:
    if c in df_display.columns:
        df_display[c] = df_display[c].apply(fmt_brl)

st.dataframe(df_display, use_container_width=True, height=400)

st.divider()

# ─────────────────────────────────────────────
# EXPORTAR
# ─────────────────────────────────────────────
excel_bytes = exportar_excel(df_result, df_rede_orig, df_rede_grupo)
st.download_button(
    label="⬇️ Exportar Resultado Excel",
    data=excel_bytes,
    file_name=f"conciliacao_{datetime.today().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
