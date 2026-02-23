import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import json

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Journey Retail Media AI", layout="wide")
st.title("🚀 Journey Campaign AI Insight Generator")
st.markdown("Upload Excel + Enter Groq API Key → Generate Executive PDF")

groq_key = st.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])


# --------------------------------------------------
# SMART SHEET CLEANING (CRITICAL FIX)
# --------------------------------------------------
def clean_sheet(file, sheet_name):

    df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)

    # Find the row where real headers exist
    header_row = None
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).str.lower()
        if any("impression" in x or "click" in x or "trip" in x for x in row):
            header_row = i
            break

    if header_row is None:
        return pd.DataFrame()

    df = pd.read_excel(file, sheet_name=sheet_name, header=header_row)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Drop completely empty rows
    df = df.dropna(how="all")

    df = df.reset_index(drop=True)

    return df


# --------------------------------------------------
# GRANULARITY DETECTION
# --------------------------------------------------
def detect_granularity(sheet_name):
    name = sheet_name.lower()

    if "overall" in name:
        return "Overall Campaign Level"
    elif "ad_group" in name:
        return "Ad Group Level"
    elif "city" in name:
        return "City Level"
    elif "categorie" in name:
        return "Category Level"
    elif "top_places" in name:
        return "Top Places Level"
    elif "car" in name:
        return "Car Type Level"
    elif "day" in name:
        return "Day of Week Level"
    else:
        return "Other"


# --------------------------------------------------
# KPI ENGINE (PROPER VERSION)
# --------------------------------------------------
def compute_metrics(df):

    metrics = {}

    # Remove duplicate columns (CRITICAL FIX)
    df = df.loc[:, ~df.columns.duplicated()]

    # Standardize column names
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Detect important columns
    impressions_col = next((c for c in df.columns if "impression" in c), None)
    clicks_col = next((c for c in df.columns if c == "clicks"), None)
    trips_col = next((c for c in df.columns if "trip" in c and "%" not in c), None)
    ctr_col = next((c for c in df.columns if c == "ctr"), None)

    # -----------------------------
    # SAFE NUMERIC CLEANING
    # -----------------------------
    def safe_numeric(series):
        if isinstance(series, pd.Series):
            return (
                series.astype(str)
                .str.replace(",", "", regex=False)
                .replace("", "0")
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
        return series

    if impressions_col and impressions_col in df.columns:
        df[impressions_col] = safe_numeric(df[impressions_col])

    if clicks_col and clicks_col in df.columns:
        df[clicks_col] = safe_numeric(df[clicks_col])

    if trips_col and trips_col in df.columns:
        df[trips_col] = safe_numeric(df[trips_col])

    if ctr_col and ctr_col in df.columns:
        df[ctr_col] = safe_numeric(df[ctr_col])

    # -----------------------------
    # TOTALS
    # -----------------------------
    total_impressions = df[impressions_col].sum() if impressions_col else 0
    total_clicks = df[clicks_col].sum() if clicks_col else 0
    total_trips = df[trips_col].sum() if trips_col else 0

    metrics["total_impressions"] = float(total_impressions)
    metrics["total_clicks"] = float(total_clicks)
    metrics["total_trips"] = float(total_trips)

    # -----------------------------
    # CTR %
    # -----------------------------
    if ctr_col:
        metrics["avg_ctr_percent"] = float(df[ctr_col].mean() * 100)
    elif impressions_col and clicks_col and total_impressions > 0:
        df["ctr_calc"] = df[clicks_col] / df[impressions_col]
        metrics["avg_ctr_percent"] = float(df["ctr_calc"].mean() * 100)

    # -----------------------------
    # TRIP RATE %
    # -----------------------------
    if clicks_col and trips_col and total_clicks > 0:
        df["trip_rate"] = df[trips_col] / df[clicks_col]
        metrics["avg_trip_rate_percent"] = float(df["trip_rate"].mean() * 100)

    # -----------------------------
    # ENTITY DETECTION
    # -----------------------------
    identifier_candidates = [
        "ad_group_name", "ad_group",
        "category", "city",
        "place_name", "car_type",
        "ad_name", "day"
    ]

    id_col = next((col for col in identifier_candidates if col in df.columns), None)

    # -----------------------------
    # TOP & BOTTOM ENTITIES
    # -----------------------------
    if id_col and trips_col and total_trips > 0:

        df_sorted = df.sort_values(trips_col, ascending=False)

        top = df_sorted.head(3)
        bottom = df_sorted.tail(3)

        metrics["top_entities"] = [
            {
                "name": str(row[id_col]),
                "trips": float(row[trips_col]),
                "ctr_percent": float((row[ctr_col] * 100) if ctr_col else 0)
            }
            for _, row in top.iterrows()
        ]

        metrics["bottom_entities"] = [
            {
                "name": str(row[id_col]),
                "trips": float(row[trips_col]),
                "ctr_percent": float((row[ctr_col] * 100) if ctr_col else 0)
            }
            for _, row in bottom.iterrows()
        ]

    # -----------------------------
    # CONCENTRATION RISK
    # -----------------------------
    if id_col and trips_col and total_trips > 0:
        top3_share = df.sort_values(trips_col, ascending=False).head(3)[trips_col].sum() / total_trips
        metrics["top3_trip_contribution_percent"] = float(top3_share * 100)

    metrics["row_count"] = len(df)

    return metrics


# --------------------------------------------------
# INGEST EXCEL
# --------------------------------------------------
def ingest_excel(file):

    xls = pd.ExcelFile(file)
    documents = []

    for sheet in xls.sheet_names:

        df = clean_sheet(file, sheet)

        if df.empty:
            continue

        granularity = detect_granularity(sheet)
        metrics = compute_metrics(df)

        structured_content = f"""
Granularity: {granularity}
Sheet Name: {sheet}

KPIs:
{json.dumps(metrics, indent=2)}
"""

        doc = Document(
            page_content=structured_content,
            metadata={"sheet": sheet}
        )

        documents.append(doc)

    return documents


# --------------------------------------------------
# LLM GENERATION
# --------------------------------------------------
def generate_insights(documents, groq_key):

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are a Senior Retail Media Strategist.

Below is campaign data:

{context}

Generate strong executive insights using:
- Total Impressions
- Total Clicks
- Total Trips
- CTR %
- Trip Rate %

For each level:
Highlight TOP 3 and BOTTOM 3 entities.
Be decisive.
No generic language.
Use actual numbers.
"""

    response = llm.invoke(prompt)
    return response.content


# --------------------------------------------------
# PDF
# --------------------------------------------------
def generate_pdf(text):

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Journey Campaign Report", styles["Heading1"]))
    elements.append(Spacer(1, 20))

    for line in text.split("\n"):
        elements.append(Paragraph(line.strip(), styles["Normal"]))
        elements.append(Spacer(1, 8))

    doc.build(elements)

    return temp_pdf.name


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if st.button("Generate AI Insights"):

    if not uploaded_file:
        st.error("Please upload Excel file.")
        st.stop()

    if not groq_key:
        st.error("Please enter Groq API key.")
        st.stop()

    documents = ingest_excel(uploaded_file)

    if not documents:
        st.error("No valid sheets found.")
        st.stop()

    insights = generate_insights(documents, groq_key)

    pdf_path = generate_pdf(insights)

    st.success("Report Generated Successfully")

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="Journey_Report.pdf",
            mime="application/pdf"
        )

    st.subheader("Insight Preview")
    st.write(insights)