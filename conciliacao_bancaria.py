"""
Dashboard de Conciliação Bancária
OFX (Extrato Bancário) x XLS (Intermediadora / Rede)
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
            bandeira = "AMEX"       if b == "AMERICAN EXPRESS" else bandeira
            bandeira = "HIPERCARD"  if b == "HIPER" else bandeira
            break
    for t in tipos:
        if t in memo_upper:
            tipo = "CREDITO" if t in ["CREDITO", "CRÉDITO", "CREDIT"] else "DEBITO"
            break
    return bandeira, tipo


def parse_intermediadora_xls(file) -> pd.DataFrame:
    """Parser para o relatório TSV da intermediadora (.xls)."""
    def to_float(s):
        if pd.isna(s) or str(s).strip() in ("", "-", "nan"): return 0.0
        s = re.sub(r"\(.*?\)", "", str(s).strip())
        nums = re.findall(r"[\d.,]+", s)
        if not nums: return 0.0
        try: return float(nums[0].replace(".", "").replace(",", "."))
        except: return 0.0

    try:
        content = file.read()
        text = content.decode("latin-1", errors="ignore") if isinstance(content, bytes) else content
        df_raw = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, encoding=None)
    except Exception as e:
        raise ValueError(f"Erro ao ler arquivo da intermediadora: {e}")

    df_raw.columns = [c.strip() for c in df_raw.columns]

    df = pd.DataFrame()
    df["data"]          = pd.to_datetime(df_raw["DATA VENDA"], dayfirst=True, errors="coerce").dt.date
    df["bandeira"]      = df_raw["BANDEIRA"].astype(str).str.strip().str.upper()
    df["tipo"]          = df_raw["TRANSACAO"].astype(str).str.strip() if "TRANSACAO" in df_raw.columns else df_raw["TRANSAÇÃO"].astype(str).str.strip()
    df["valor_bruto"]   = df_raw["VALOR BRUTO"].apply(to_float)
    df["valor_liquido"] = df_raw["VALOR LÍQUIDO"].apply(to_float) if "VALOR LÍQUIDO" in df_raw.columns else df_raw["VALOR LIQUIDO"].apply(to_float)
    df["taxa_final"]    = df_raw["TAXA FINAL (R$)"].apply(to_float)
    df["taxa_pct"]      = df_raw["TAXA FINAL (%)"].apply(to_float)
    df["status_adq"]    = df_raw.get("STATUS VENDA", pd.Series([""] * len(df_raw))).astype(str)
    df["cv"]            = df_raw.get("C.V.", pd.Series([""] * len(df_raw))).astype(str)
    df["estabelecimento"] = df_raw.get("ESTABELECIMENTO", pd.Series([""] * len(df_raw))).astype(str)

    df["bandeira"] = df["bandeira"].replace({
        "MASTER": "MASTERCARD", "MC": "MASTERCARD",
        "AMERICAN EXPRESS": "AMEX", "AX": "AMEX", "VI": "VISA",
    })
    df["tipo_norm"] = df["tipo"].apply(
        lambda x: "CREDITO" if "CRÉD" in x.upper() or "CRED" in x.upper()
                  else ("DEBITO" if "DÉB" in x.upper() or "DEB" in x.upper() else x.upper())
    )

    df = df.dropna(subset=["data"])
    df = df[df["valor_bruto"] > 0].reset_index(drop=True)
    df["idx_transacao"] = df.index
    return df


def agrupar_rede(df_rede: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por data + bandeira + tipo, preservando lista de índices por grupo."""
    group_cols = ["data", "bandeira", "tipo_norm"]
    agg = {}
    for c in ["valor_bruto", "valor_liquido", "taxa_final"]:
        if c in df_rede.columns: agg[c] = "sum"
    agg["idx_transacao"] = lambda x: list(x)

    df_group = df_rede.groupby(group_cols, as_index=False).agg(agg)
    df_group["qtd_transacoes"] = df_group["idx_transacao"].apply(len)
    df_group = df_group.reset_index(drop=True)
    df_group["idx_grupo"] = df_group.index.astype(int)
    return df_group


def conciliar(df_ofx: pd.DataFrame, df_rede_grupo: pd.DataFrame,
              tolerancia_dias: int = 1, tolerancia_valor: float = 0.05) -> pd.DataFrame:
    """Cruza OFX com grupos da intermediadora. Mantém idx_grupo para rastreio."""
    df_ofx = df_ofx.copy()
    bandeira_tipo = df_ofx["memo"].apply(lambda m: pd.Series(detectar_bandeira_tipo(m)))
    df_ofx[["bandeira_ofx", "tipo_ofx"]] = bandeira_tipo

    resultados  = []
    rede_usados = set()
    rede_rows   = df_rede_grupo.reset_index(drop=True)

    for _, row_ofx in df_ofx.iterrows():
        melhor_match = None
        melhor_diff  = float("inf")

        for i_rede, row_rede in rede_rows.iterrows():
            if i_rede in rede_usados: continue
            if (row_ofx["bandeira_ofx"] not in ("OUTROS", "") and
                    row_rede.get("bandeira", "") not in ("OUTROS", "") and
                    row_ofx["bandeira_ofx"] != row_rede.get("bandeira", "")): continue
            if (row_ofx["tipo_ofx"] not in ("OUTROS", "") and
                    row_rede.get("tipo_norm", "") not in ("OUTROS", "") and
                    row_ofx["tipo_ofx"] != row_rede.get("tipo_norm", "")): continue
            diff_dias = abs((row_ofx["data"] - row_rede["data"]).days)
            if diff_dias > tolerancia_dias: continue
            val_ofx  = abs(row_ofx["valor_ofx"])
            val_rede = abs(row_rede.get("valor_liquido", row_rede.get("valor_bruto", 0)))
            if val_rede == 0: continue
            diff_perc = abs(val_ofx - val_rede) / val_rede
            if diff_perc > tolerancia_valor: continue
            score = diff_dias + diff_perc
            if score < melhor_diff:
                melhor_diff  = score
                melhor_match = i_rede

        row_rede_matched  = {}
        idx_grupo_matched = None

        if melhor_match is not None:
            rede_usados.add(melhor_match)
            row_rede_matched  = rede_rows.loc[melhor_match].to_dict()
            idx_grupo_matched = row_rede_matched.get("idx_grupo")
            diff_valor = abs(row_ofx["valor_ofx"]) - abs(row_rede_matched.get("valor_liquido", 0))
            status = "⚠️ Conciliado c/ Divergência" if abs(diff_valor) > 0.01 else "✅ Conciliado"
        else:
            status     = "❌ Não Conciliado (banco)"
            diff_valor = None

        resultados.append({
            "Status":           status,
            "idx_grupo":        idx_grupo_matched,
            "Data OFX":         row_ofx["data"],
            "Valor OFX":        row_ofx["valor_ofx"],
            "Memo OFX":         row_ofx["memo"],
            "Bandeira OFX":     row_ofx["bandeira_ofx"],
            "Tipo OFX":         row_ofx["tipo_ofx"],
            "Data Rede":        row_rede_matched.get("data", ""),
            "Bandeira Rede":    row_rede_matched.get("bandeira", ""),
            "Tipo Rede":        row_rede_matched.get("tipo_norm", ""),
            "Valor Bruto Rede": row_rede_matched.get("valor_bruto", ""),
            "Valor Líq. Rede":  row_rede_matched.get("valor_liquido", ""),
            "Qtd Transações":   row_rede_matched.get("qtd_transacoes", ""),
            "Diferença (R$)":   diff_valor,
        })

    for i_rede, row_rede in rede_rows.iterrows():
        if i_rede not in rede_usados:
            resultados.append({
                "Status":           "❌ Não Conciliado (Rede)",
                "idx_grupo":        row_rede.get("idx_grupo"),
                "Data OFX":         "", "Valor OFX": "", "Memo OFX": "",
                "Bandeira OFX":     "", "Tipo OFX": "",
                "Data Rede":        row_rede.get("data", ""),
                "Bandeira Rede":    row_rede.get("bandeira", ""),
                "Tipo Rede":        row_rede.get("tipo_norm", ""),
                "Valor Bruto Rede": row_rede.get("valor_bruto", ""),
                "Valor Líq. Rede":  row_rede.get("valor_liquido", ""),
                "Qtd Transações":   row_rede.get("qtd_transacoes", ""),
                "Diferença (R$)":   None,
            })

    return pd.DataFrame(resultados)


def build_status_transacao(df_rede_orig: pd.DataFrame,
                            df_rede_grupo: pd.DataFrame,
                            df_result: pd.DataFrame,
                            vinculos_manuais: dict) -> pd.DataFrame:
    """Propaga status do grupo (auto + manual) para cada transação individual.
    Vínculos manuais podem ser por idx_grupo (legado) ou por idx_transacao direta (virtual).
    """
    # ── Status automáticos: idx_grupo → status ──
    status_por_grupo = {}
    memo_por_grupo   = {}
    valor_por_grupo  = {}
    for _, row in df_result.iterrows():
        ig = row["idx_grupo"]
        if ig is not None:
            status_por_grupo[ig] = row["Status"]
            memo_por_grupo[ig]   = row.get("Memo OFX", "")
            valor_por_grupo[ig]  = row.get("Valor OFX", "")

    # ── Vínculos manuais: podem ser por idx_grupo ou por idx_transacao ──
    # status_por_transacao tem prioridade sobre status_por_grupo
    status_por_transacao = {}
    memo_por_transacao   = {}
    valor_por_transacao  = {}

    for chave, info in vinculos_manuais.items():
        if info.get("virtual"):
            # Vínculo direto por transação individual
            for idx_t in info.get("idx_transacoes", []):
                status_por_transacao[idx_t] = info["status"]
                memo_por_transacao[idx_t]   = info.get("memo_ofx", "")
                valor_por_transacao[idx_t]  = info.get("valor_ofx", "")
        else:
            # Vínculo por grupo (legado)
            try:
                ig = int(chave)
                status_por_grupo[ig] = info["status"]
                memo_por_grupo[ig]   = info.get("memo_ofx", "")
                valor_por_grupo[ig]  = info.get("valor_ofx", "")
            except (ValueError, TypeError):
                pass

    # Mapa idx_transacao → idx_grupo
    mapa_idx = {}
    for _, row in df_rede_grupo.iterrows():
        for idx_t in row["idx_transacao"]:
            mapa_idx[idx_t] = row["idx_grupo"]

    df = df_rede_orig.copy()
    df["idx_grupo"] = df["idx_transacao"].map(mapa_idx)

    def get_status(row):
        idx_t = row["idx_transacao"]
        if idx_t in status_por_transacao:
            return status_por_transacao[idx_t]
        ig = row["idx_grupo"]
        return status_por_grupo.get(ig, "❌ Não Conciliado (Rede)")

    def get_memo(row):
        idx_t = row["idx_transacao"]
        if idx_t in memo_por_transacao:
            return memo_por_transacao[idx_t]
        ig = row["idx_grupo"]
        return memo_por_grupo.get(ig, "")

    def get_valor(row):
        idx_t = row["idx_transacao"]
        if idx_t in valor_por_transacao:
            return valor_por_transacao[idx_t]
        ig = row["idx_grupo"]
        return valor_por_grupo.get(ig, "")

    df["Status Conciliação"] = df.apply(get_status, axis=1)
    df["Memo OFX Vinculado"]  = df.apply(get_memo,   axis=1)
    df["Valor OFX Vinculado"] = df.apply(get_valor,  axis=1)

    return df


def exportar_excel(df_result: pd.DataFrame,
                   df_detalhe_status: pd.DataFrame,
                   df_rede_grupo: pd.DataFrame,
                   vinculos_manuais: dict) -> bytes:
    output = io.BytesIO()

    df_result_exp = df_result.drop(columns=["idx_grupo"], errors="ignore")
    df_grupo_exp  = df_rede_grupo.drop(columns=["idx_transacao", "idx_grupo"], errors="ignore")
    cols_det      = [c for c in df_detalhe_status.columns if c not in ("idx_transacao", "idx_grupo")]
    df_det_exp    = df_detalhe_status[cols_det]

    # Aba de vínculos manuais
    registros_manual = []
    for ig, info in vinculos_manuais.items():
        registros_manual.append({
            "Status":          info["status"],
            "Memo OFX":        info.get("memo_ofx", ""),
            "Valor OFX":       info.get("valor_ofx", ""),
            "Data OFX":        str(info.get("data_ofx", "")),
            "Diferença (R$)":  info.get("diff_valor", ""),
            "Observação":      info.get("observacao", ""),
        })
    df_manual = pd.DataFrame(registros_manual) if registros_manual else pd.DataFrame()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_result_exp.to_excel(writer, sheet_name="Conciliação",        index=False)
        df_grupo_exp.to_excel(writer,  sheet_name="Grupos",             index=False)
        df_det_exp.to_excel(writer,    sheet_name="Detalhe Transações", index=False)
        if not df_manual.empty:
            df_manual.to_excel(writer, sheet_name="Vínculos Manuais",   index=False)

        wb    = writer.book
        fmt_h    = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1})
        fmt_ok   = wb.add_format({"bg_color": "#C6EFCE"})
        fmt_warn = wb.add_format({"bg_color": "#FFEB9C"})
        fmt_man  = wb.add_format({"bg_color": "#DDEBF7"})
        fmt_err  = wb.add_format({"bg_color": "#FFC7CE"})

        def aplicar_formato(ws, df_exp, status_col):
            for col_num, col_name in enumerate(df_exp.columns):
                ws.write(0, col_num, col_name, fmt_h)
                ws.set_column(col_num, col_num, 22)
            if status_col in df_exp.columns:
                for row_num in range(1, len(df_exp) + 1):
                    s = str(df_exp.iloc[row_num - 1][status_col])
                    if "✅" in s:    fmt = fmt_ok
                    elif "⚠️" in s: fmt = fmt_warn
                    elif "🔗" in s: fmt = fmt_man
                    else:            fmt = fmt_err
                    ws.set_row(row_num, None, fmt)

        aplicar_formato(writer.sheets["Conciliação"],        df_result_exp, "Status")
        aplicar_formato(writer.sheets["Detalhe Transações"], df_det_exp,    "Status Conciliação")

    return output.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Importar Arquivos")
    file_ofx  = st.file_uploader("Extrato Bancário (.ofx)",       type=["ofx", "OFX"])
    file_rede = st.file_uploader("Extrato Intermediadora (.xls)", type=["xls", "xlsx", "tsv", "txt"])

    st.divider()
    st.header("⚙️ Parâmetros")
    tolerancia_dias  = st.slider("Tolerância de data (dias)", 0, 5, 1)
    tolerancia_valor = st.slider("Tolerância de valor (%)",   0, 10, 5) / 100
    col_valor_rede   = st.radio("Comparar OFX com:", ["Valor Líquido", "Valor Bruto"])

    st.divider()
    st.markdown("**Legenda:**")
    st.markdown("✅ Conciliado automaticamente")
    st.markdown("⚠️ Conciliado c/ divergência")
    st.markdown("🔗 Vinculado manualmente")
    st.markdown("❌ Não conciliado")

# ─────────────────────────────────────────────
# ESTADO DA SESSÃO
# ─────────────────────────────────────────────
if "vinculos_manuais" not in st.session_state:
    st.session_state["vinculos_manuais"] = {}

# ─────────────────────────────────────────────
# AGUARDA ARQUIVOS
# ─────────────────────────────────────────────
if not file_ofx or not file_rede:
    st.info("👈 Importe o arquivo OFX e o extrato da intermediadora para iniciar.")
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
**Arquivo OFX** — Extrato bancário exportado pelo banco.  
Lançamentos com "REDE" no memo são usados na conciliação. "SALDO TOTAL" é ignorado.

**Arquivo XLS** — Relatório da intermediadora (TSV com extensão .xls).

**Conciliação automática:** agrupa por Data + Bandeira + Tipo e cruza com o OFX.

**Vinculação manual:** para grupos não conciliados automaticamente, associe a um  
lançamento OFX pendente. Vínculos são exportados no Excel.
        """)
    st.stop()

# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────
with st.spinner("Processando arquivos..."):
    try:
        df_ofx_raw = parse_ofx(file_ofx.read())
        if df_ofx_raw.empty:
            st.error("Nenhuma transação encontrada no OFX."); st.stop()
    except Exception as e:
        st.error(f"Erro ao ler OFX: {e}"); st.stop()

    try:
        df_rede_orig = parse_intermediadora_xls(file_rede)
        if df_rede_orig.empty:
            st.error("Nenhuma transação encontrada no arquivo da intermediadora."); st.stop()
    except Exception as e:
        st.error(f"Erro ao ler arquivo da intermediadora: {e}"); st.stop()

# Filtros OFX
df_ofx        = df_ofx_raw[~df_ofx_raw["memo"].str.upper().str.contains("SALDO TOTAL", na=False)].copy()
df_ofx_rede   = df_ofx[df_ofx["memo"].str.upper().str.contains("REDE", na=False)].copy()
df_ofx_outros = df_ofx[~df_ofx["memo"].str.upper().str.contains("REDE", na=False)].copy()

if col_valor_rede == "Valor Bruto" and "valor_bruto" in df_rede_orig.columns:
    df_rede_orig["valor_liquido"] = df_rede_orig["valor_bruto"]

df_rede_grupo = agrupar_rede(df_rede_orig)
df_result     = conciliar(df_ofx_rede, df_rede_grupo, tolerancia_dias, tolerancia_valor)

# Lançamentos OFX REDE pendentes (não conciliados automaticamente nem manualmente)
memos_conciliados = set(
    df_result[df_result["Status"].str.startswith(("✅", "⚠️"))]["Memo OFX"]
)
# Adiciona memos já vinculados manualmente
memos_vinculados_manual = set(
    info.get("memo_ofx", "") for info in st.session_state["vinculos_manuais"].values()
)
memos_conciliados = memos_conciliados | memos_vinculados_manual

df_ofx_pendentes = df_ofx_rede[~df_ofx_rede["memo"].isin(memos_conciliados)].copy()

# Grupos não conciliados automaticamente
# Apenas grupos com idx_grupo válido (lado Rede não conciliado)
grupos_nao_conc = df_result[
    df_result["Status"].str.contains("❌") &
    df_result["idx_grupo"].notna()
].copy()
grupos_nao_conc["idx_grupo"] = grupos_nao_conc["idx_grupo"].astype(int)
grupos_vinculados_manual = set(st.session_state["vinculos_manuais"].keys())
# Índices de transações já vinculadas manualmente (virtuais)
transacoes_ja_vinculadas = set()
for info in st.session_state["vinculos_manuais"].values():
    for idx_t in info.get("idx_transacoes", []):
        transacoes_ja_vinculadas.add(idx_t)

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
total_grupos    = len(df_rede_grupo)
total_conc_auto = df_result["Status"].str.contains("✅|⚠️", regex=True).sum()
total_manual    = len(st.session_state["vinculos_manuais"])
total_pendentes = max(len(grupos_nao_conc) - total_manual, 0)
pct = lambda n: f"{n/total_grupos*100:.1f}%" if total_grupos else "0%"

c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Grupos Intermediadora",  total_grupos)
c2.metric("✅ Conciliados auto",        f"{total_conc_auto} ({pct(total_conc_auto)})")
c3.metric("🔗 Vinculados manualmente", f"{total_manual} ({pct(total_manual)})")
c4.metric("❌ Pendentes",              f"{total_pendentes} ({pct(total_pendentes)})")

val_ofx_rede  = df_ofx_rede["valor_ofx"].apply(lambda v: v if v > 0 else 0).sum()
val_rede_bruto = df_rede_orig["valor_bruto"].sum()  if "valor_bruto"   in df_rede_orig.columns else 0
val_rede_liq   = df_rede_orig["valor_liquido"].sum() if "valor_liquido" in df_rede_orig.columns else 0

cv1, cv2, cv3, cv4 = st.columns(4)
cv1.metric("💰 OFX — Lançamentos REDE",          f"R$ {val_ofx_rede:,.2f}")
cv2.metric("💳 Intermediadora — Valor Bruto",     f"R$ {val_rede_bruto:,.2f}")
cv3.metric("🏦 Intermediadora — Valor Líquido",   f"R$ {val_rede_liq:,.2f}")
cv4.metric("📊 Diferença OFX × Líquido",
           f"R$ {val_ofx_rede - val_rede_liq:,.2f}",
           delta=f"{((val_ofx_rede - val_rede_liq)/val_rede_liq*100):.2f}%" if val_rede_liq else None)

st.divider()

# ─────────────────────────────────────────────
# ABAS PRINCIPAIS
# ─────────────────────────────────────────────
aba_result, aba_detalhe, aba_manual, aba_outros = st.tabs([
    "🔍 Conciliação",
    "📋 Detalhe por Transação",
    "🔗 Vinculação Manual",
    "📄 Outros Lançamentos OFX",
])

# ════════════════════════════════════════════
# ABA 1 — RESULTADO DA CONCILIAÇÃO
# ════════════════════════════════════════════
with aba_result:
    st.subheader("Resultado — Grupos (Intermediadora × OFX)")

    # Aplica vínculos manuais na exibição
    # Vínculos virtuais (por idx_transacao) não têm idx_grupo — são ignorados aqui,
    # pois o status deles aparece no Detalhe por Transação
    df_result_display = df_result.copy()
    for ig, info in st.session_state["vinculos_manuais"].items():
        if info.get("virtual"):
            continue  # vínculo por transação individual — sem idx_grupo no resultado
        try:
            ig_int = int(ig)
        except (ValueError, TypeError):
            continue
        mask = df_result_display["idx_grupo"] == ig_int
        df_result_display.loc[mask, "Status"]    = info["status"]
        df_result_display.loc[mask, "Memo OFX"]  = info.get("memo_ofx", "")
        df_result_display.loc[mask, "Valor OFX"] = info.get("valor_ofx", "")
        df_result_display.loc[mask, "Data OFX"]  = info.get("data_ofx", "")

    status_opcoes = df_result_display["Status"].unique().tolist()
    filtro = st.multiselect("Filtrar por status:", options=status_opcoes,
                             default=status_opcoes, key="filtro_resultado")
    df_filtrado = df_result_display[df_result_display["Status"].isin(filtro)]

    def fmt_brl(v):
        try:
            if v == "" or pd.isna(v): return ""
            return f"R$ {float(v):,.2f}"
        except: return v

    df_show = df_filtrado.drop(columns=["idx_grupo"], errors="ignore").copy()
    for c in ["Valor OFX", "Valor Bruto Rede", "Valor Líq. Rede", "Diferença (R$)"]:
        if c in df_show.columns:
            df_show[c] = df_show[c].apply(fmt_brl)

    st.dataframe(df_show, use_container_width=True, height=420)

# ════════════════════════════════════════════
# ABA 2 — DETALHE POR TRANSAÇÃO
# ════════════════════════════════════════════
with aba_detalhe:
    st.subheader("Detalhe por Transação Individual")
    st.caption("Cada transação herda o status do seu grupo (Data + Bandeira + Tipo).")

    df_detalhe = build_status_transacao(
        df_rede_orig, df_rede_grupo, df_result,
        st.session_state["vinculos_manuais"]
    )

    status_det_opcoes = df_detalhe["Status Conciliação"].unique().tolist()
    filtro_det = st.multiselect("Filtrar por status:", options=status_det_opcoes,
                                 default=status_det_opcoes, key="filtro_detalhe")
    df_det_filtrado = df_detalhe[df_detalhe["Status Conciliação"].isin(filtro_det)]

    cols_show = ["Status Conciliação", "data", "bandeira", "tipo_norm", "tipo",
                 "valor_bruto", "taxa_final", "taxa_pct", "valor_liquido",
                 "cv", "estabelecimento", "status_adq",
                 "Memo OFX Vinculado", "Valor OFX Vinculado"]
    cols_show = [c for c in cols_show if c in df_det_filtrado.columns]

    rename_det = {
        "Status Conciliação":  "Status",
        "data":                "Data",
        "bandeira":            "Bandeira",
        "tipo_norm":           "Tipo",
        "tipo":                "Descrição",
        "valor_bruto":         "Valor Bruto",
        "taxa_final":          "Taxa (R$)",
        "taxa_pct":            "Taxa (%)",
        "valor_liquido":       "Valor Líquido",
        "cv":                  "C.V.",
        "estabelecimento":     "Estabelecimento",
        "status_adq":          "Status Adquirente",
        "Memo OFX Vinculado":  "Memo OFX",
        "Valor OFX Vinculado": "Valor OFX",
    }

    df_det_show = df_det_filtrado[cols_show].rename(columns=rename_det).copy()
    for c in ["Valor Bruto", "Valor Líquido", "Taxa (R$)"]:
        if c in df_det_show.columns:
            df_det_show[c] = df_det_show[c].apply(
                lambda v: f"R$ {v:,.2f}" if pd.notna(v) and v != "" else ""
            )

    st.dataframe(df_det_show, use_container_width=True, height=450)

    with st.expander("📊 Resumo por status"):
        resumo_status = df_detalhe["Status Conciliação"].value_counts().reset_index()
        resumo_status.columns = ["Status", "Qtd Transações"]
        st.dataframe(resumo_status, use_container_width=True)

# ════════════════════════════════════════════
# ABA 3 — VINCULAÇÃO MANUAL
# ════════════════════════════════════════════
with aba_manual:
    st.subheader("🔗 Vinculação Manual de Lançamentos")
    st.info(
        "Selecione um **lançamento OFX pendente**, depois marque as **transações individuais** "
        "da intermediadora que compõem esse valor. Somente lançamentos ❌ estão disponíveis.",
        icon="ℹ️"
    )

    # Índices de transações já vinculadas manualmente (não podem ser reusadas)
    transacoes_vinculadas = set()
    for info in st.session_state["vinculos_manuais"].values():
        for idx_t in info.get("idx_transacoes", []):
            transacoes_vinculadas.add(idx_t)

    if df_ofx_pendentes.empty:
        st.success("✅ Não há lançamentos OFX pendentes para vinculação manual.")
    else:
        # ── 1. Seleção do lançamento OFX ────────
        st.markdown("#### 1. Selecione o lançamento OFX pendente")

        mapa_ofx = {}
        for _, row in df_ofx_pendentes.iterrows():
            label = f"{row['data']}  |  {row['memo']}  |  R$ {abs(row['valor_ofx']):,.2f}"
            mapa_ofx[label] = row

        sel_ofx_label = st.selectbox("Lançamento OFX:", list(mapa_ofx.keys()), key="sel_ofx")
        sel_ofx_row   = mapa_ofx.get(sel_ofx_label)

        if sel_ofx_row is None:
            st.warning("Selecione um lançamento OFX para continuar.")
            st.stop()

        val_ofx_s = float(sel_ofx_row["valor_ofx"])

        # ── 2. Seleção das transações da intermediadora ──
        st.markdown("#### 2. Selecione as transações da intermediadora")
        st.caption("Marque transações até atingir o valor do lançamento OFX acima.")

        # Transações ainda não vinculadas a nenhum OFX (automático ou manual)
        # Exclui transações que já foram conciliadas automaticamente
        grupos_conciliados_auto = set(
            df_result[df_result["Status"].str.startswith(("✅", "⚠️"))]["idx_grupo"].dropna().astype(int)
        )
        idxs_auto_conciliados = set()
        for ig in grupos_conciliados_auto:
            rows = df_rede_grupo[df_rede_grupo["idx_grupo"] == ig]
            if not rows.empty:
                for idx_t in rows.iloc[0]["idx_transacao"]:
                    idxs_auto_conciliados.add(idx_t)

        df_trans_disponiveis = df_rede_orig[
            ~df_rede_orig["idx_transacao"].isin(idxs_auto_conciliados) &
            ~df_rede_orig["idx_transacao"].isin(transacoes_vinculadas)
        ].copy().reset_index(drop=True)

        if df_trans_disponiveis.empty:
            st.warning("Não há transações disponíveis para vinculação.")
        else:
            # ── Tabela com checkbox (data_editor) ──
            cols_tabela = ["data", "bandeira", "tipo_norm", "cv",
                           "valor_bruto", "taxa_final", "valor_liquido"]
            cols_tabela = [c for c in cols_tabela if c in df_trans_disponiveis.columns]

            df_editor = df_trans_disponiveis[cols_tabela].copy()
            # Converte data para string para evitar exibição como timestamp no data_editor
            df_editor["data"] = df_editor["data"].apply(lambda d: d.strftime("%d/%m/%Y") if pd.notna(d) and hasattr(d, "strftime") else str(d))
            df_editor.insert(0, "✔", False)   # coluna de seleção

            # Formata valores para exibição
            for c, col in [("valor_bruto", "Valor Bruto"), ("taxa_final", "Taxa (R$)"),
                           ("valor_liquido", "Valor Líquido")]:
                if c in df_editor.columns:
                    df_editor[col] = df_editor[c].apply(lambda v: f"R$ {v:,.2f}")
                    df_editor = df_editor.drop(columns=[c])

            df_editor = df_editor.rename(columns={
                "data": "Data", "bandeira": "Bandeira",
                "tipo_norm": "Tipo", "cv": "C.V.",
            })

            edited = st.data_editor(
                df_editor,
                column_config={
                    "✔": st.column_config.CheckboxColumn("✔", help="Marque para incluir", width="small"),
                    "Data":          st.column_config.TextColumn("Data",       width="small"),
                    "Bandeira":      st.column_config.TextColumn("Bandeira",   width="small"),
                    "Tipo":          st.column_config.TextColumn("Tipo",       width="small"),
                    "C.V.":          st.column_config.TextColumn("C.V.",       width="medium"),
                    "Valor Bruto":   st.column_config.TextColumn("Valor Bruto",  width="medium"),
                    "Taxa (R$)":     st.column_config.TextColumn("Taxa (R$)",    width="small"),
                    "Valor Líquido": st.column_config.TextColumn("Valor Líquido", width="medium"),
                },
                disabled=["Data", "Bandeira", "Tipo", "C.V.",
                          "Valor Bruto", "Taxa (R$)", "Valor Líquido"],
                hide_index=True,
                use_container_width=True,
                key="editor_transacoes",
                height=min(400, 45 + len(df_editor) * 35),
            )

            # Recupera índices das linhas marcadas
            idx_marcados = edited[edited["✔"]].index.tolist()
            df_sel = df_trans_disponiveis.iloc[idx_marcados]
            total_liq_sel   = df_sel["valor_liquido"].sum()
            total_bruto_sel = df_sel["valor_bruto"].sum()
            total_taxa_sel  = df_sel["taxa_final"].sum()
            selecionadas    = idx_marcados   # usado apenas para len()

            # ── 3. Conferência ───────────────────
            st.markdown("#### 3. Conferência de valores")
            diff_val = abs(val_ofx_s) - total_liq_sel
            diff_pct = (diff_val / abs(val_ofx_s) * 100) if val_ofx_s else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Valor OFX",         f"R$ {abs(val_ofx_s):,.2f}")
            m2.metric("Selecionado Bruto",  f"R$ {total_bruto_sel:,.2f}")
            m3.metric("Selecionado Líq.",   f"R$ {total_liq_sel:,.2f}",
                      help="Deve ser igual ou próximo ao Valor OFX")
            m4.metric("Diferença",          f"R$ {diff_val:,.2f}",
                      delta=f"{diff_pct:.2f}%" if selecionadas else None,
                      delta_color="off" if abs(diff_val) < 0.01 else "inverse")

            obs = st.text_input("Observação (opcional):", key="obs_manual",
                                placeholder="Ex: transações do dia 04/01 referentes ao OFX de 06/01")

            # Validações
            nenhuma_sel   = len(selecionadas) == 0
            diff_bloqueio = abs(diff_pct) > 10 and not nenhuma_sel

            if nenhuma_sel:
                st.info("Selecione ao menos uma transação para confirmar o vínculo.")
            elif diff_bloqueio:
                st.error(f"⛔ Diferença de {diff_pct:.1f}% acima de 10%. "
                          "Revise as transações selecionadas.")
            elif abs(diff_val) > 0.01:
                st.warning(f"⚠️ Diferença de R$ {diff_val:,.2f} ({diff_pct:.2f}%). "
                            "O vínculo será marcado como divergente.")

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                confirmar = st.button(
                    "✅ Confirmar Vínculo", type="primary",
                    use_container_width=True,
                    disabled=(nenhuma_sel or diff_bloqueio)
                )

            if confirmar:
                # Cria um grupo virtual com as transações selecionadas
                idx_virtual = f"manual_{len(st.session_state['vinculos_manuais'])}"
                status_manual = "🔗 Vinculado Manualmente" if abs(diff_val) < 0.01 else "🔗 Vinculado c/ Divergência"

                st.session_state["vinculos_manuais"][idx_virtual] = {
                    "status":         status_manual,
                    "memo_ofx":       sel_ofx_row["memo"],
                    "valor_ofx":      sel_ofx_row["valor_ofx"],
                    "data_ofx":       sel_ofx_row["data"],
                    "observacao":     obs,
                    "diff_valor":     diff_val,
                    "idx_transacoes": list(df_sel["idx_transacao"]),
                    "total_liq":      total_liq_sel,
                    "total_bruto":    total_bruto_sel,
                    "total_taxa":     total_taxa_sel,
                    "virtual":        True,   # não corresponde a idx_grupo real
                }
                st.success(f"✅ {len(selecionadas)} transação(ões) vinculada(s) ao OFX **{sel_ofx_row['memo']}**!")
                st.rerun()

    # ── Lista de vínculos registrados ───────────
    if st.session_state["vinculos_manuais"]:
        st.divider()
        st.markdown("#### Vínculos manuais registrados nesta sessão")

        registros = []
        for chave, info in st.session_state["vinculos_manuais"].items():
            qtd_trans = len(info.get("idx_transacoes", []))
            registros.append({
                "Status":          info["status"],
                "Memo OFX":        info.get("memo_ofx", ""),
                "Data OFX":        str(info.get("data_ofx", "")),
                "Valor OFX":       f"R$ {abs(float(info.get('valor_ofx', 0))):,.2f}",
                "Valor Líq. Sel.": f"R$ {float(info.get('total_liq', 0)):,.2f}",
                "Diferença":       f"R$ {float(info.get('diff_valor', 0)):,.2f}",
                "Qtd Transações":  qtd_trans,
                "Observação":      info.get("observacao", ""),
            })

        df_vinculos = pd.DataFrame(registros)
        st.dataframe(df_vinculos, use_container_width=True)

        col_limpar, _ = st.columns([1, 3])
        with col_limpar:
            if st.button("🗑️ Limpar todos os vínculos manuais", type="secondary",
                         use_container_width=True):
                st.session_state["vinculos_manuais"] = {}
                st.rerun()

# ════════════════════════════════════════════
# ABA 4 — OUTROS LANÇAMENTOS OFX
# ════════════════════════════════════════════
with aba_outros:
    st.subheader("Outros Lançamentos OFX — Não Relacionados à Rede")
    if df_ofx_outros.empty:
        st.info("Nenhum lançamento OFX fora do escopo da Rede.")
    else:
        df_out = df_ofx_outros[["data", "valor_ofx", "memo"]].copy()
        df_out["valor_ofx"] = df_out["valor_ofx"].apply(lambda v: f"R$ {v:,.2f}")
        df_out = df_out.rename(columns={"data": "Data", "valor_ofx": "Valor", "memo": "Memo"})
        st.dataframe(df_out, use_container_width=True)
        total_outros = df_ofx_outros["valor_ofx"].apply(lambda v: v if v > 0 else 0).sum()
        st.metric("Total créditos", f"R$ {total_outros:,.2f}")

st.divider()

# ─────────────────────────────────────────────
# EXPORTAR
# ─────────────────────────────────────────────
df_detalhe_export = build_status_transacao(
    df_rede_orig, df_rede_grupo, df_result,
    st.session_state["vinculos_manuais"]
)
excel_bytes = exportar_excel(
    df_result, df_detalhe_export, df_rede_grupo,
    st.session_state["vinculos_manuais"]
)
st.download_button(
    label="⬇️ Exportar Resultado Completo (Excel)",
    data=excel_bytes,
    file_name=f"conciliacao_{datetime.today().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
