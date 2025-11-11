import os
import streamlit as st

# Detect Streamlit Cloud environment
def _running_on_streamlit_cloud():
    return (
        os.environ.get("STREAMLIT_RUNTIME") == "true"
        or "STC_DEPLOYMENT" in os.environ
        or "STREALTIME" in os.environ
        or "STREAMLIT_SERVER_ENABLED" in os.environ
    )

# Try loading pyttsx3 only for local runs
if not _running_on_streamlit_cloud():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except Exception:
        engine = None
else:
    engine = None


def alert_user(msg="Pothole ahead!"):
    print(msg)

    # Cloud environment → disable pyttsx3, avoid crash
    if _running_on_streamlit_cloud():
        st.warning(msg)
        return

    # Local machine with working pyttsx3
    if engine is not None:
        try:
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            # fallback UI message
            st.warning(f"(Voice unavailable) {msg}")
    else:
        st.warning(msg)
