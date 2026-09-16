import os
import asyncio
import json
import io
import traceback
import pypdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Charge les variables du fichier .env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du client avec la variable d'environnement uniquement
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- MODÈLES QCM ---
class QuestionModel(BaseModel):
    text: str = Field(..., description="Le texte de la question")
    options: list[str] = Field(..., description="Liste de 4 choix ou propositions de réponses uniques")
    correctIndex: int = Field(..., description="L'index (0, 1, 2 ou 3) de la bonne réponse")

class QuizContainer(BaseModel):
    questions: list[QuestionModel]

# --- MODÈLES FLASHCARD ---
class FlashcardModel(BaseModel):
    question: str = Field(..., description="La question synthétique posée au recto de la carte")
    answer: str = Field(..., description="La réponse claire et concise figurant au verso")

class FlashcardContainer(BaseModel):
    cards: list[FlashcardModel]

# --- MODÈLES RÉSUMÉ ---
class SummaryModel(BaseModel):
    title: str = Field(..., description="Un titre clair et accrocheur résumant le thème principal du document")
    bulletPoints: list[str] = Field(..., description="Liste de points clés synthétiques et essentiels extraits du cours")

# --- NOUVEAUX MODÈLES VRAI / FAUX ---
class TrueFalseItemModel(BaseModel):
    statement: str = Field(..., description="L'affirmation ou la proposition à évaluer")
    isTrue: bool = Field(..., description="True si l'affirmation est VRAIE, False si elle est FAUSSE")
    explanation: str = Field(..., description="Une explication courte et pédagogique de la réponse")

class TrueFalseContainer(BaseModel):
    items: list[TrueFalseItemModel]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        return " ".join(lines)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture PDF: {str(e)}")


# --- ENDPOINT QCM ---
@app.post("/generate")
async def generate_qcm(file: UploadFile = File(...), count: int = Form(5)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        content_type = file.content_type or ""

        contents_payload = []

        if filename.endswith(".pdf") or "pdf" in content_type:
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, _extract_text_from_pdf, file_bytes)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce PDF.")
            contents_payload.append(f"Texte source du cours :\n{raw_text[:20000]}")
        else:
            mime = content_type if content_type.startswith("image/") else "image/jpeg"
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
            contents_payload.append(image_part)

        prompt_instructions = f"""
        Tu es un enseignant expert. Analyse rigoureusement le cours fourni (texte ou image) et génère exactement {count} questions de QCM.
        Chaque question doit posséder 4 options réalistes mais distinctes, dont une seule est correcte.
        L'index 'correctIndex' doit correspondre strictly à la position de la bonne réponse dans la liste 'options' (0 pour le premier choix, etc.).
        Base-toi uniquement sur les faits réels du document.
        """
        contents_payload.append(prompt_instructions)

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizContainer,
                temperature=0.3,
            ),
        )

        result_data = json.loads(response.text)
        return JSONResponse(content=result_data.get('questions', []))

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n--- ERREUR GENERATE QCM ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


# --- ENDPOINT FLASHCARDS ---
@app.post("/generate-flashcards")
async def generate_flashcards(file: UploadFile = File(...), count: int = Form(5)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        content_type = file.content_type or ""

        contents_payload = []

        if filename.endswith(".pdf") or "pdf" in content_type:
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, _extract_text_from_pdf, file_bytes)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce PDF.")
            contents_payload.append(f"Texte source du cours :\n{raw_text[:20000]}")
        else:
            mime = content_type if content_type.startswith("image/") else "image/jpeg"
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
            contents_payload.append(image_part)

        prompt_instructions = f"""
        Tu es un expert en pédagogie et mémorisation.
        Analyse le document/image fourni et génère exactement {count} flashcards de révision (paire Question / Réponse).
        - La question doit être ciblée, précise et inviter à tester une notion clé.
        - La réponse doit être concise, exacte et facile à mémoriser.
        """
        contents_payload.append(prompt_instructions)

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FlashcardContainer,
                temperature=0.3,
            ),
        )

        result_data = json.loads(response.text)
        return JSONResponse(content=result_data.get('cards', []))

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n--- ERREUR GENERATE FLASHCARDS ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne Flashcards : {str(e)}")


# --- ENDPOINT GENERATE SUMMARY ---
@app.post("/generate-summary")
async def generate_summary(file: UploadFile = File(...), count: int = Form(5)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        content_type = file.content_type or ""

        contents_payload = []

        if filename.endswith(".pdf") or "pdf" in content_type:
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, _extract_text_from_pdf, file_bytes)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce PDF.")
            contents_payload.append(f"Texte source du cours :\n{raw_text[:20000]}")
        else:
            mime = content_type if content_type.startswith("image/") else "image/jpeg"
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
            contents_payload.append(image_part)

        prompt_instructions = f"""
        Tu es un expert en synthèse de cours.
        Analyse le document/image fourni et génère un résumé structuré :
        1. Donnes-en un titre représentatif.
        2. Extrais exactement {count} points clés essentiels sous forme de puces informatives, compréhensibles et autonomes.
        """
        contents_payload.append(prompt_instructions)

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SummaryModel,
                temperature=0.3,
            ),
        )

        result_data = json.loads(response.text)
        return JSONResponse(content=result_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n--- ERREUR GENERATE SUMMARY ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne Résumé : {str(e)}")


# --- ENDPOINT GENERATE VRAI / FAUX ---
@app.post("/generate-true-false")
async def generate_true_false(file: UploadFile = File(...), count: int = Form(5)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower() if file.filename else ""
        content_type = file.content_type or ""

        contents_payload = []

        if filename.endswith(".pdf") or "pdf" in content_type:
            loop = asyncio.get_event_loop()
            raw_text = await loop.run_in_executor(None, _extract_text_from_pdf, file_bytes)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="Impossible d'extraire du texte de ce PDF.")
            contents_payload.append(f"Texte source du cours :\n{raw_text[:20000]}")
        else:
            mime = content_type if content_type.startswith("image/") else "image/jpeg"
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
            contents_payload.append(image_part)

        prompt_instructions = f"""
        Tu es un enseignant évaluateur.
        Analyse le document/image fourni et génère exactement {count} questions sous forme de 'Vrai ou Faux'.
        - 'statement': L'affirmation testée. Varie les propositions vraies et fausses de manière équilibrée.
        - 'isTrue': True si la proposition est vraie d'après le cours, False si elle est fausse.
        - 'explanation': Une explication claire, exacte et pédagogique récapitulant pourquoi c'est vrai ou faux.
        """
        contents_payload.append(prompt_instructions)

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TrueFalseContainer,
                temperature=0.3,
            ),
        )

        result_data = json.loads(response.text)
        return JSONResponse(content=result_data.get('items', []))

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n--- ERREUR GENERATE VRAI/FAUX ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne Vrai/Faux : {str(e)}")

# --- ENDPOINT RACINE ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "API MentorMe Python opérationnelle"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)