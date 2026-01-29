# app.py
import streamlit as st
import subprocess
from pathlib import Path
import sys
import threading
import time

from face_search import search_similar_faces, FaceSearchError

st.set_page_config(page_title="Voice → Forensic Sketch", layout="centered")

# ---------- Global layout & styling ----------
st.markdown(
    """
    <style>
    /* Animated background gradient */
    @keyframes gradientShift {
        0% {
            background-position: 0% 0%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 0%;
        }
    }

    /* Page background & layout */
    body, .stApp {
        background-image: linear-gradient(135deg, #050812, #151b3a, #050812);
        background-size: 250% 250%;
        animation: gradientShift 18s ease-in-out infinite;
        color: #f5f5f5;
    }
    .block-container {
        max-width: 980px !important;
        padding-top: 2.5rem !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5,8,18,0.98), rgba(13,18,38,0.98));
        border-left: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: -8px 0 24px rgba(0, 0, 0, 0.45);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2.0rem !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #e5e7ff;
    }
    section[data-testid="stSidebar"] p {
        color: #c5c9dd;
        font-size: 0.9rem;
    }

    /* Hide default Streamlit elements for a more app-like feel */
    #MainMenu, header, footer {
        visibility: hidden;
        height: 0px;
    }

    /* Top nav / brand bar */
    .app-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0.25rem 0.75rem 0.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 0.75rem;
    }
    .brand {
        font-weight: 700;
        letter-spacing: 0.06em;
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #9fb5ff;
    }
    .brand-pill {
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(159, 181, 255, 0.3);
        display: inline-block;
    }
    .nav-tagline {
        font-size: 0.8rem;
        color: #c9cedf;
    }

    /* Hero */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }
    .subtitle {
        color: #c4c7d4;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }

    /* Cards & sections */
    .card {
        background: linear-gradient(135deg, #101425, #050812);
        border-radius: 0.9rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.0rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #f3f4ff;
    }
    .status-badge {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        background-color: rgba(129, 140, 248, 0.15);
        color: #c7d2ff;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.4rem;
        border: 1px solid rgba(129, 140, 248, 0.5);
    }

    /* Footer */
    .app-footer {
        margin-top: 1.75rem;
        text-align: center;
        font-size: 0.75rem;
        color: #9ca3b5;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-nav">
      <div class="brand-pill">
        <span class="brand">VOICE2SKETCH</span>
      </div>
      <div class="nav-tagline">Voice‑driven forensic illustration</div>
    </div>
    <div class="main-title">🎙️ Voice‑driven Forensic Sketch Generation</div>
    <div class="subtitle">
      Convert an audio description into a forensic‑style facial sketch using speech‑to‑text and image generation.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with concise guidance
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        """
        1. Upload an audio description (WAV/MP3/M4A/OGG).  
        2. We transcribe it using your local `transcribe.py`.  
        3. We generate a forensic sketch via `generate_sketch.py`.  
        """
    )
    st.markdown("### Tips")
    st.markdown(
        """
        - Speak clearly and describe age, gender, hair, nose, mouth, and any marks.  
        - Longer, detailed descriptions tend to produce better sketches.
        """
    )

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-title">1. Upload audio</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported formats: WAV, MP3, M4A, OGG", type=["wav", "mp3", "m4a", "ogg"]
        )

    with col_right:
        st.markdown('<div class="section-title">Session status</div>', unsafe_allow_html=True)
        if uploaded_file:
            st.markdown("✅ Audio file selected")
        else:
            st.markdown("⬜ Awaiting audio upload")

    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Save audio as description.wav
    audio_path = Path("description.wav")
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Audio saved as description.wav")

    # ----------------------------------------
    # Run transcribe.py
    # ----------------------------------------
    st.info("Running transcribe.py...")
    transcribe_progress = st.progress(0, text="Transcribing audio...")

    # Run transcribe.py in a background thread so we can update the progress bar.
    transcribe_result_container = {}

    def _run_transcribe():
        try:
            transcribe_result_container["result"] = subprocess.run(
                [sys.executable, "transcribe.py"],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except Exception as e:
            transcribe_result_container["error"] = e

    t_thread = threading.Thread(target=_run_transcribe)
    t_thread.start()

    # Simple animated progress loop while the transcription is running.
    pct = 0
    while t_thread.is_alive():
        pct = (pct + 2) % 100
        transcribe_progress.progress(pct + 1, text="Transcribing audio...")
        time.sleep(0.2)

    t_thread.join()
    transcribe_progress.progress(100, text="Transcription complete")

    if "error" in transcribe_result_container:
        st.error(f"transcribe.py failed: {transcribe_result_container['error']}")
        st.stop()

    result = transcribe_result_container.get("result")
    if result is None:
        st.error("transcribe.py failed with no result.")
        st.stop()

    # If the script failed, show stderr; otherwise, show only stdout so that
    # warnings (like FP16/FP32) do not appear as the "transcription output".
    if result.returncode != 0:
        combined = ""
        if result.stdout:
            combined += result.stdout + "\n"
        if result.stderr:
            combined += result.stderr
        lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
        transcription_output = (
            lines[-1] if lines else "transcribe.py failed with no output."
        )
    else:
        # Successful run: prefer the last non‑empty line from stdout only.
        out_lines = [
            ln.strip() for ln in (result.stdout or "").splitlines() if ln.strip()
        ]
        transcription_output = out_lines[-1] if out_lines else "No transcription output."

    # Keep transcription ready; it will be displayed together with the sketch below.

    # ----------------------------------------
    # Run generate_sketch.py
    # ----------------------------------------
    st.markdown(
        '<div class="section-title">3. Forensic sketch generation<span class="status-badge">from generate_sketch.py</span></div>',
        unsafe_allow_html=True,
    )
    st.info("Running sketch generation...")

    sketch_progress = st.progress(0, text="Generating sketch...")

    sketch_result_container = {}

    def _run_sketch():
        try:
            # Pass the transcription text to generate_sketch.py so the image
            # reflects what was actually described (e.g., "old man, 52 years").
            sketch_result_container["result"] = subprocess.run(
                [sys.executable, "generate_sketch.py", transcription_output],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except Exception as e:
            sketch_result_container["error"] = e

    s_thread = threading.Thread(target=_run_sketch)
    s_thread.start()

    pct2 = 0
    while s_thread.is_alive():
        pct2 = (pct2 + 2) % 100
        sketch_progress.progress(pct2 + 1, text="Generating sketch...")
        time.sleep(0.2)

    s_thread.join()
    sketch_progress.progress(100, text="Sketch generation complete")

    if "error" in sketch_result_container:
        st.error(f"generate_sketch.py failed: {sketch_result_container['error']}")
        st.stop()

    result2 = sketch_result_container.get("result")
    if result2 is None:
        st.error("generate_sketch.py failed with no result.")
        st.stop()

    # ----------------------------------------
    # Load and show generated image
    # ----------------------------------------
    img_path = Path("criminal_sketch.png")

    st.markdown('<div class="section-title">4. Result</div>', unsafe_allow_html=True)

    if img_path.exists():
        st.success("Sketch generated successfully!")

        # Side‑by‑side layout: transcription on the left, sketch & matches on the right
        col_txt, col_img = st.columns([1.1, 1.6])

        with col_txt:
            st.markdown(
                '<div class="section-title">Transcription<span class="status-badge">from transcribe.py</span></div>',
                unsafe_allow_html=True,
            )
            st.code(transcription_output)

        with col_img:
            st.markdown(
                '<div class="section-title">Generated sketch</div>',
                unsafe_allow_html=True,
            )
            st.image(str(img_path), use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    label="Download sketch",
                    data=f,
                    file_name=img_path.name,
                    mime="image/png",
                )

            # ----------------------------------------
            # Optional: Retrieve similar mugshots
            # ----------------------------------------
            st.markdown(
                '<div class="section-title">5. Top‑5 similar mugshots</div>',
                unsafe_allow_html=True,
            )

            run_search = st.toggle(
                "Run mugshot similarity search (requires prebuilt FAISS index)",
                value=False,
            )

            if run_search:
                with st.spinner("Searching iDOC mugshot database (FAISS + SSIM)..."):
                    try:
                        matches = search_similar_faces(str(img_path), top_k=5)
                    except FileNotFoundError:
                        st.info(
                            "Mugshot index not found. From a terminal, run:\n\n"
                            "`python build_mugshot_index.py --dataset-dir \"PATH_TO_iDOC_DATASET\"`"
                        )
                    except FaceSearchError as e:
                        st.error(f"Face search error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected face search failure: {e}")
                    else:
                        if not matches:
                            st.warning("No candidate faces found in the index.")
                        else:
                            for rank, match in enumerate(matches, start=1):
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    st.image(
                                        match["image_path"],
                                        caption=f"Rank #{rank}",
                                        use_container_width=True,
                                    )
                                with c2:
                                    st.write(f"**Image:** `{match['image_path']}`")
                                    st.write(
                                        f"**SSIM similarity:** {match['ssim']:.3f}  "
                                    )
                                    st.write(
                                        f"**Embedding distance (FAISS):** {match['distance']:.4f}"
                                    )
    else:
        st.error("Error: criminal_sketch.png not found.")

    st.markdown(
        '<div class="app-footer">Built with Streamlit • Uses your local transcription & sketch generation models</div>',
        unsafe_allow_html=True,
    )
