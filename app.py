import streamlit as st
import time
import google.generativeai as genai
from huggingface_hub import InferenceClient

# --- SIDKONFIGURATION ---
st.set_page_config(page_title="AI Samtalsanalys: Moln vs Lokal", layout="wide")

# Hämta API-nycklar från Streamlits hemligheter (secrets)
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_MODEL_REPO = st.secrets["HF_MODEL_REPO"] # T.ex. "DittAnvändarnamn/min-fina-modell"

# --- FUNKTIONER ---
def analyze_with_gemini(transcript, prompt):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    start_time = time.time()
    response = model.generate_content(f"{prompt}\n\nTranskription:\n{transcript}")
    latency = time.time() - start_time
    
    return response.text, latency

def analyze_with_hf(transcript, prompt):
    client = InferenceClient(model=HF_MODEL_REPO, token=HF_TOKEN)
    full_prompt = f"<|sammanhang|>{prompt}<|användare|>{transcript}<|assistent|>"
    
    start_time = time.time()
    # Anropa din tränade modell via Hugging Face
    response = client.text_generation(
        full_prompt, 
        max_new_tokens=300, 
        temperature=0.1
    )
    latency = time.time() - start_time
    
    return response, latency

# --- GRÄNSSNITT ---
st.title("🎙️ AI Samtalsanalys: Gemini vs Egen Modell")
st.markdown("Testa hur Googles kommersiella API presterar jämfört med en skräddarsydd, svensk modell.")

# Input-sektion
st.subheader("1. Klistra in samtalstranskription")
default_transcript = """Kunden: Hej, min faktura är fel.
Agenten: Jaha, fan vad irriterande. Låt mig kolla. Ja, det var fel. Jag fixar.
Kunden: Okej tack.
Agenten: Hejdå."""

transcript = st.text_area("Samtalstext", value=default_transcript, height=150)

# System-prompt gömd i en expander för att hålla gränssnittet rent
with st.expander("⚙️ Visa/Redigera Systeminstruktioner (Regler)"):
    system_prompt = st.text_area(
        "Instruktioner", 
        value="Hitta regelbrott i samtalet. Leta efter svordomar och oprofessionellt beteende. Svara i kortfattad punktform.",
        height=100
    )

if st.button("🚀 Starta Analys", use_container_width=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("☁️ Google Gemini 1.5 Flash")
        with st.spinner("Väntar på Google..."):
            try:
                gemini_res, gem_time = analyze_with_gemini(transcript, system_prompt)
                word_count = len(gemini_res.split())
                
                # Visa resultat
                st.success("Klar!")
                st.markdown(f"**Resultat:**\n{gemini_res}")
                
                # Statistik
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Svarstid", f"{gem_time:.2f} s")
                m2.metric("Hastighet", f"{int(word_count/gem_time)} ord/s")
                m3.metric("Est. Kostnad", "< 0.01 kr")
                
            except Exception as e:
                st.error(f"Ett fel uppstod: {e}")

    with col2:
        st.header("🇸🇪 Egen Modell (Hugging Face)")
        with st.spinner("Väntar på Hugging Face..."):
            try:
                hf_res, hf_time = analyze_with_hf(transcript, system_prompt)
                hf_word_count = len(hf_res.split())
                
                # Räkna ut en rolig Compliance Score baserat på längden på svaret (färre klagomål = högre score)
                compliance_score = max(10, 100 - (hf_word_count * 2))
                
                # Visa resultat
                st.success("Klar!")
                st.markdown(f"**Resultat:**\n{hf_res}")
                
                # Statistik
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Svarstid", f"{hf_time:.2f} s")
                m2.metric("Hastighet", f"{int(hf_word_count/hf_time)} ord/s")
                m3.metric("Compliance Score", f"{compliance_score} %", delta="-Kräver åtgärd" if compliance_score < 50 else "Godkänt")
                
            except Exception as e:
                st.error(f"Ett fel uppstod: {e}\n\n(Obs: Om modellen precis har vaknat kan det ta en minut extra!)")
