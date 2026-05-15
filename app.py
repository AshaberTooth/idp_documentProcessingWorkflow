import os
import json
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from openai import AzureOpenAI


load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

# Azure Document Intelligence
DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
DOCUMENTINTELLIGENCE_KEY = os.getenv("DOCUMENTINTELLIGENCE_KEY")

# Azure OpenAI from Azure AI Foundry deployment page
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_content_type(file_path: str) -> str:
    ext = file_path.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return "application/pdf"

    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"

    if ext == "png":
        return "image/png"

    raise ValueError("Unsupported file type. Please upload PDF, JPG, JPEG, or PNG.")


def get_document_intelligence_client() -> DocumentIntelligenceClient:
    if not DOCUMENTINTELLIGENCE_ENDPOINT:
        raise ValueError("DOCUMENTINTELLIGENCE_ENDPOINT is missing in .env")

    if not DOCUMENTINTELLIGENCE_KEY:
        raise ValueError("DOCUMENTINTELLIGENCE_KEY is missing in .env")

    return DocumentIntelligenceClient(
        endpoint=DOCUMENTINTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(DOCUMENTINTELLIGENCE_KEY),
    )


def analyze_document_with_doc_intelligence(file_path: str) -> dict:
    """
    Extracts text, pages, and tables using Azure AI Document Intelligence.
    """

    file_size = os.path.getsize(file_path)

    print("Sending file to Document Intelligence:", file_path)
    print("File size:", file_size)

    if file_size == 0:
        raise ValueError(
            "Uploaded file is empty. Check that your HTML form has enctype='multipart/form-data'."
        )

    client = get_document_intelligence_client()
    content_type = get_content_type(file_path)

    with open(file_path, "rb") as document:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=document,
            content_type=content_type,
        )

    result = poller.result()

    extracted_lines = []
    pages = []

    if result.pages:
        for page in result.pages:
            page_lines = []

            if page.lines:
                for line in page.lines:
                    extracted_lines.append(line.content)
                    page_lines.append(line.content)

            pages.append(
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit,
                    "lines": page_lines,
                }
            )

    tables = []

    if result.tables:
        for table in result.tables:
            table_data = {
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": [],
            }

            if table.cells:
                for cell in table.cells:
                    table_data["cells"].append(
                        {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "content": cell.content,
                        }
                    )

            tables.append(table_data)

    extracted_text = "\n".join(extracted_lines)

    return {
        "text": extracted_text,
        "pages": pages,
        "tables": tables,
    }


def get_azure_openai_client() -> AzureOpenAI:
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT is missing in .env")

    if not AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY is missing in .env")

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def call_foundry_model(system_prompt: str, user_prompt: str) -> str:
    """
    Calls your Azure OpenAI model deployment from Azure AI Foundry.
    """

    if not AZURE_OPENAI_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT is missing in .env")

    client = get_azure_openai_client()

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=3000,
    )

    return response.choices[0].message.content


def extract_structured_json(extracted_payload: dict) -> dict:
    system_prompt = """
You are an intelligent document processing extraction agent.

You receive OCR/layout output from Azure AI Document Intelligence.
Your task is to identify the document type and extract relevant fields.

Return only valid JSON.

Required JSON schema:
{
  "document_type": "invoice|contract|id|receipt|form|letter|report|unknown",
  "language": "",
  "title": "",
  "parties": [],
  "dates": [],
  "amounts": [],
  "key_value_pairs": {},
  "tables": [],
  "important_clauses_or_sections": [],
  "confidence_notes": [],
  "missing_or_unclear_fields": []
}

Rules:
- Preserve original values.
- Do not invent missing fields.
- If a field is missing, leave it empty or mention it in missing_or_unclear_fields.
- Return JSON only.
- Do not include markdown.
"""

    user_prompt = f"""
Analyze the following extracted document payload and return structured JSON only:

{json.dumps(extracted_payload, ensure_ascii=False)}
"""

    raw_response = call_foundry_model(system_prompt, user_prompt)

    cleaned = raw_response.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "document_type": "unknown",
            "language": "",
            "title": "",
            "parties": [],
            "dates": [],
            "amounts": [],
            "key_value_pairs": {},
            "tables": [],
            "important_clauses_or_sections": [],
            "confidence_notes": [
                "The model response could not be parsed as valid JSON."
            ],
            "missing_or_unclear_fields": [],
            "raw_model_response": raw_response,
        }


def summarize_document(extracted_payload: dict, structured_json: dict) -> str:
    system_prompt = """
You are a document summarization agent.

Create a clear business summary from the extracted document text and structured JSON.

Return markdown with these sections:
1. Document Type
2. Executive Summary
3. Key Entities
4. Key Dates
5. Financial Values
6. Obligations, Risks, or Action Items
7. Missing or Unclear Information

Rules:
- Do not invent information.
- If something is not available, say "Not found in the document."
"""

    user_prompt = f"""
Extracted document text:
{extracted_payload.get("text", "")}

Structured JSON:
{json.dumps(structured_json, ensure_ascii=False)}
"""

    return call_foundry_model(system_prompt, user_prompt)


def answer_question(question: str, extracted_payload: dict, structured_json: dict) -> str:
    system_prompt = """
You are a grounded document QnA assistant.

Answer the user's question using only the uploaded document content.

Rules:
- Do not use outside knowledge.
- If the answer is not in the document, say:
  "I could not find this in the uploaded document."
- Keep the answer concise.
"""

    user_prompt = f"""
Document text:
{extracted_payload.get("text", "")}

Structured JSON:
{json.dumps(structured_json, ensure_ascii=False)}

User question:
{question}
"""

    return call_foundry_model(system_prompt, user_prompt)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return render_template(
                "index.html",
                error="No file field found in the request.",
            )

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return render_template(
                "index.html",
                error="No file selected.",
            )

        if not allowed_file(uploaded_file.filename):
            return render_template(
                "index.html",
                error="Only PDF, JPG, JPEG, and PNG files are supported.",
            )

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        uploaded_file.save(file_path)

        file_size = os.path.getsize(file_path)

        print("Uploaded file:", file_path)
        print("Uploaded file size:", file_size)

        if file_size == 0:
            return render_template(
                "index.html",
                error="Uploaded file is 0 bytes. Make sure your form uses enctype='multipart/form-data'.",
            )

        extracted_payload = analyze_document_with_doc_intelligence(file_path)

        extracted_text = extracted_payload.get("text", "")

        if not extracted_text.strip():
            return render_template(
                "index.html",
                error="Document Intelligence processed the file, but no readable text was extracted.",
            )

        structured_json = extract_structured_json(extracted_payload)
        summary = summarize_document(extracted_payload, structured_json)

        return render_template(
            "index.html",
            filename=filename,
            extracted_text=extracted_text,
            structured_json=json.dumps(structured_json, indent=2, ensure_ascii=False),
            summary=summary,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Analyze failed: {str(e)}",
        )


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form.get("question", "").strip()
        extracted_text = request.form.get("extracted_text", "").strip()
        structured_json_text = request.form.get("structured_json", "").strip()

        if not question:
            return render_template(
                "index.html",
                error="Please enter a question.",
            )

        if not extracted_text:
            return render_template(
                "index.html",
                error="No document content found. Please upload and analyze a document first.",
            )

        try:
            structured_json = json.loads(structured_json_text)
        except Exception:
            structured_json = {}

        extracted_payload = {
            "text": extracted_text,
        }

        answer = answer_question(
            question=question,
            extracted_payload=extracted_payload,
            structured_json=structured_json,
        )

        return render_template(
            "index.html",
            extracted_text=extracted_text,
            structured_json=structured_json_text,
            question=question,
            answer=answer,
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"QnA failed: {str(e)}",
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8400, debug=True)