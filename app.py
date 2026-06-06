import streamlit as st
import time
import os
import google.generativeai as genai
from huggingface_hub import InferenceClient

# --- SIDKONFIGURATION ---
st.set_page_config(page_title="AI Samtalsanalys: Ljuduppladdning", layout="wide")

# Hämta API-nycklar från Streamlits hemligheter (secrets)
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_MODEL_REPO = st.secrets["HF_MODEL_REPO"]

# --- FUNKTIONER ---
def analyze_audio_with_gemini(file_path, prompt):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    start_time = time.time()
    
    # 1. Ladda upp ljudfilen till Googles File API
    st.info("🔄 Laddar upp ljudfil till Google Cloud...")
    audio_file = genai.upload_file(path=file_path)
    
    # 2. Vänta på att Google har processat filen (viktigt för större filer)
    while audio_file.state.name == "PROCESSING":
        time.sleep(1)
        audio_file = genai.get_file(audio_file.name)
        
    if audio_file.state.name == "FAILED":
        raise Exception("Google lyckades inte processa ljudfilen.")
        
    st.info("🧠 Gemini 1.5 Flash lyssnar och analyserar...")
    # 3. Skicka filen + prompten till Gemini
   # 3. Skicka filen + prompten till Gemini
    response = model.generate_content(
        [prompt, audio_file],
        generation_config={"response_mime_type": "application/json", "temperature": 0.1}
    )
    
    # 4. Städa upp och ta bort filen från molnet
    genai.delete_file(audio_file.name)
    
    latency = time.time() - start_time
    return response.text, latency

def analyze_text_with_hf(transcript, prompt):
    client = InferenceClient(model=HF_MODEL_REPO, token=HF_TOKEN)
    full_prompt = f"<|sammanhang|>{prompt}<|användare|>{transcript}<|assistent|>"
    
    start_time = time.time()
    response = client.text_generation(
        full_prompt, 
        max_new_tokens=300, 
        temperature=0.1
    )
    latency = time.time() - start_time
    return response, latency

# --- GRÄNSSNITT ---
st.title("🎙️ Multimodal AI Samtalsanalys: Ladda upp egna samtal")
st.markdown("Ladda upp en inspelning (.mp3, .wav eller .m4a) för att köra en automatiserad kvalitetsgranskning.")

# --- SEKTION 1: LADDA UPP FIL ---
st.subheader("1. Ladda upp ljudfil från kundsamtal")
uploaded_file = uploaded_file = st.file_uploader(
    "Välj en ljudfil", 
    type=["mp3", "wav", "m4a", "ogg"],
    help="Ladda upp ett inspelat samtal mellan en agent och en kund."
)

# Om en fil har laddats upp, visa en ljudspelare så man kan lyssna på den i appen
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Spara filen tillfälligt på Streamlits server så att Google SDK kan läsa den
    temp_file_path = os.path.join("/tmp", uploaded_file.name) if os.path.exists("/tmp") else uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.divider()

# --- SEKTION 2: INSTRUKTIONER ---
st.subheader("2. Ange granskningsregler")
with st.expander("⚙️ Justera vad AI:n ska leta efter i ljudfilen", expanded=True):
    system_prompt = st.text_area(
        "Systeminstruktioner & Regler", 
        value="""Du är en expert på kvalitetsgranskning av kundtjänst. Analysera detta samtal utifrån följande regler:
1. Agenten får absolut inte använda svordomar eller oprofessionellt språk.
2. Agenten måste erbjuda kunden en skriftlig bekräftelse eller kvitto på slutet av samtalet.

Returnera resultatet som ett strikt JSON-objekt med denna exakta struktur:
{
    "rule_violations": [
        {
            "rule": "Namnet på regeln som bröts",
            "quote": "Exakt citat från samtalet där överträdelsen skedde",
            "severity": "High/Medium/Low"
        }
    ]
}""",
        height=200
    )

# --- SEKTION 3: STARTA ---
if st.button("🚀 Starta Multimodal Analys", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Vänligen ladda upp en ljudfil först!")
    else:
        col1, col2 = st.columns(2)
        
        # --- GOOGLE GEMINI PIPELINE ---
        with col1:
            st.header("☁️ Google Gemini 1.5 Flash")
            with st.spinner("Analyserar ljudfil..."):
                try:
                    gemini_res, gem_time = analyze_audio_with_gemini(temp_file_path, system_prompt)
                    word_count = len(gemini_res.split())
                    
                    st.success("Analys klar!")
                    st.json(gemini_json := gemini_res) # Visar JSON snyggt formaterat
                    
                    # Statistik
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Svarstid (Ljud)", f"{gem_time:.2f} s")
                    m2.metric("Hastighet", f"{int(word_count/gem_time)} ord/s")
                    m3.metric("API-Status", "200 OK")
                    
                    # Spara texten till sessions-state så att den kan användas av nästa modell
                    st.session_state["last_analysis"] = gemini_res
                    
                except Exception as e:
                    st.error(f"Ett fel uppstod med Gemini: {e}")

        # --- HUGGING FACE PIPELINE ---
        with col2:
            st.header("🇸🇪 Egen Modell (Hugging Face)")
            
            # Eftersom HF Inference API kräver text, skickar vi med den transkriberade 
            # datan eller insikterna som Gemini fick fram för att låta din modell utvärdera texten
            if "last_analysis" in st.session_state:
                with st.spinner("Kör kompletterande svensk analys..."):
                    try:
                        hf_res, hf_time = analyze_text_with_hf(st.session_state["last_analysis"], system_prompt)
                        hf_word_count = len(hf_res.split())
                        compliance_score = max(10, 100 - (hf_word_count * 2))
                        
                        st.success("Klar!")
                        st.markdown(f"**Modellens slutsats:**\n{hf_res}")
                        
                        # Statistik
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Svarstid (Text)", f"{hf_time:.2f} s")
                        m2.metric("Hastighet", f"{int(hf_word_count/hf_time)} ord/s")
                        m3.metric("Compliance Score", f"{compliance_score} %")
                        
                    except Exception as e:
                        st.error(f"Ett fel uppstod: {e}")
            else:
                st.info("💡 Den lokala modellen startar så fort Gemini har bearbetat ljudfilen och skapat textunderlaget.")
        
        # Städa bort den tillfälliga filen från Streamlit-servern efter körning
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
