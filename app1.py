import streamlit as st
import pandas as pd
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
st.title(":rocket: Journey Campaign AI Insight Generator")
st.markdown("Upload Excel + Enter Groq API Key → Generate Executive PDF")

groq_key = st.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

# --------------------------------------------------
# SMART SHEET CLEANING
# --------------------------------------------------
def clean_sheet(file, sheet_name):

    df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)

    header_row = None
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].astype(str).str.lower()
        if any("impression" in x or "click" in x or "trip" in x for x in row):
            header_row = i
            break

    if header_row is None:
        return pd.DataFrame()

    df = pd.read_excel(file, sheet_name=sheet_name, header=header_row)

    df.columns = df.columns.astype(str).str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all").reset_index(drop=True)

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
    elif "top_place" in name:
        return "Top Places Level"
    elif "car" in name:
        return "Car Type Level"
    elif "day" in name:
        return "Day of Week Level"
    else:
        return "Other"



# --------------------------------------------------
# KPI ENGINE
# --------------------------------------------------
def compute_metrics(df):

    metrics = {}

    impressions_col = next((c for c in df.columns if "impression" in c), None)
    clicks_col = next((c for c in df.columns if c == "clicks"), None)
    trips_col = next((c for c in df.columns if "trip" in c and "%" not in c), None)
    ctr_col = next((c for c in df.columns if c == "ctr"), None)

    def clean_numeric(series):
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .replace(["", "nan", "None"], "0")
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

    for col in [impressions_col, clicks_col, trips_col, ctr_col]:
        if col and col in df.columns:
            df[col] = clean_numeric(df[col])

    if impressions_col:
        df = df[df[impressions_col] > 0]

    total_impressions = df[impressions_col].sum() if impressions_col else 0
    total_clicks = df[clicks_col].sum() if clicks_col else 0
    total_trips = df[trips_col].sum() if trips_col else 0

    metrics["total_impressions"] = float(total_impressions)
    metrics["total_clicks"] = float(total_clicks)
    metrics["total_trips"] = float(total_trips)

    if ctr_col:
        df["ctr_percent"] = df[ctr_col] * 100
    elif impressions_col and clicks_col:
        df["ctr_percent"] = (df[clicks_col] / df[impressions_col]) * 100

    if "ctr_percent" in df.columns:
        metrics["avg_ctr_percent"] = round(float(df["ctr_percent"].mean()), 2)

    identifier_candidates = [
        "ad_group_name", "ad_group",
        "category", "city",
        "place_name", "car_type",
        "ad_name", "day"
    ]

    id_col = next((col for col in identifier_candidates if col in df.columns), None)

    ignore_values = ["0", "multi unit", "\\n", "\\N", "nan", None]

    if id_col:
        df = df[~df[id_col].astype(str).str.lower().isin(ignore_values)]

    if "ctr_percent" in df.columns:
        df = df[df["ctr_percent"] > 0]

    if id_col and len(df) > 1:

        df_sorted = df.sort_values(trips_col, ascending=False)

        top = df_sorted.head(3)
        bottom = df_sorted.tail(3)

        metrics["top_entities"] = [
            {
                "name": str(row[id_col]),
                "trips": float(row[trips_col]),
                "ctr_percent": round(float(row["ctr_percent"]), 2)
            }
            for _, row in top.iterrows()
        ]

        metrics["bottom_entities"] = [
            {
                "name": str(row[id_col]),
                "trips": float(row[trips_col]),
                "ctr_percent": round(float(row["ctr_percent"]), 2)
            }
            for _, row in bottom.iterrows()
        ]

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
# LLM STRATEGIC GENERATION
# --------------------------------------------------
def generate_insights(documents, groq_key):

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.15
    )

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
You are a Senior Reatil media Architect, you are good at insight generation and provide strategic recommendation .

Below is structured campaign KPI data:

{context}

IMPORTANT RULES:

1. Use ONLY:
   - Total Impressions
   - Total Clicks
   - Total Trips
   - CTR %
   - Top Entities
   - Bottom Entities

2. NEVER:
   - Recommend increasing budget for weak entities
   - Call high CTR (above campaign average) as weak
   - Repeat generic "moderate CTR risk" statements
   - Mention Trip Rate

3. Logic Rules:
   - Strong = Higher CTR + Higher Total Trips
   - Weak = Lower CTR + Lower Total Trips
   - If metrics are similar across entities → state “Performance is evenly distributed”

4. Category Level:
   - Focus ONLY on Top 5 categories by Total Trips
   - Call out 2 strongest based on good trip count as well as good CTR  and 2 weakest properly based on CTR

5. City Level:
   - Focus ONLY on Top 10 cities by Total Trips and good CTR
   - Identify cities with BOTH high CTR and high Trips as strong
   - Identify cities with low CTR and low Trips as weak

6. Weekday Level:
   - Rank days by Total Trips
   - Identify Peak Day (good amount of trips + strong CTR)
   - Identify Lowest Day (lowest trips)
   - Recommend weekday budget shift strategy

7. Budget Strategy:
   - Suggest realistic shift (5%–20%)
   - Shift budget FROM weak TO strong only

Structure output EXACTLY as:

=== Overall Campaign Level ===

Performance Diagnosis:

=== Ad Group Level ===

Performance Diagnosis:

=== Category Level (Top 5 Only) ===

Strong Categories:
Weak Categories:
Budget Reallocation:
Scaling Opportunity:
Risk:

=== City Level (Top 10 Only) ===

Strong Cities:
Weak Cities:
Geo Budget Strategy:
Scaling Plan:
Risk:

=== Car Type Level ===

Strong Ride Types:
Weak Ride Types:
Allocation Strategy:
Risk:

=== Weekday Level ===

Peak Day:
Lowest Day:
Weekday Budget Strategy:
Risk:

=== 30 DAY EXECUTION PLAN ===

Week 1:
Week 2:
Week 3:
Week 4:

=== Strategic Recommendation ===

provide overall takeaways and strategic recommendation

Be data-driven.
Use actual numbers.
Avoid repetition.
Executive tone.
"""

    response = llm.invoke(prompt)
    return response.content



# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------
def generate_pdf(text):

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Journey Campaign Strategic Report", styles["Heading1"]))
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

    st.success(":white_check_mark: Strategic Report Generated Successfully")

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="Journey_Strategic_Report.pdf",
            mime="application/pdf"
        )

    st.subheader(":mag_right: Insight Preview")
    st.write(insights)
