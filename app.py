import streamlit as st
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Lost & Found Face Match – Demo",
    layout="centered"
)

# ---------------- LOADING MESSAGE ----------------
st.info("⏳ App is loading, please wait… Greatness takes time 🙂")

# ---------------- IMPORT AI MODULES ----------------
from backend.detector import detect_face
from backend.embedder import get_face_embedding
from backend.matcher import find_top_matches

# ---------------- TITLE ----------------
st.title("🧒 Lost & Found Face Match (Demo Mode)")
st.write(
    "Register a person using one photo, then upload another photo "
    "to identify them. Demo data is stored only for this session."
)

st.divider()

# ---------------- SESSION STATE ----------------
if "demo_embeddings" not in st.session_state:
    st.session_state.demo_embeddings = []

if "demo_metadata" not in st.session_state:
    st.session_state.demo_metadata = []

# ================= REGISTER =================
st.subheader("🔹 1. Register Person")

reg_file = st.file_uploader(
    "Upload a photo to register",
    type=["jpg", "jpeg", "png"],
    key="register"
)

reg_name = st.text_input("Enter name", key="reg_name")

if reg_file and reg_name:
    img = Image.open(reg_file).convert("RGB")
    st.image(img, caption="Registered Image", use_container_width=True)

    face = detect_face(img)

    if face is None:
        st.error("❌ No face detected.")
    else:
        embedding = get_face_embedding(img, face["box"])

        st.session_state.demo_embeddings.append(embedding)
        st.session_state.demo_metadata.append({"name": reg_name})

        st.success("✅ Unique face embedding generated")
        st.caption(f"Registered as: {reg_name}")

st.divider()

# ================= IDENTIFY =================
st.subheader("🔹 2. Identify Person")

id_file = st.file_uploader(
    "Upload another photo to identify",
    type=["jpg", "jpeg", "png"],
    key="identify"
)

if id_file:
    img = Image.open(id_file).convert("RGB")
    st.image(img, caption="Image to Identify", use_container_width=True)

    face = detect_face(img)

    if face is None:
        st.error("❌ No face detected.")
    else:
        query_embedding = get_face_embedding(img, face["box"])

        if not st.session_state.demo_embeddings:
            st.warning("⚠️ No registered faces yet.")
        else:
            matches = find_top_matches(
                query_embedding,
                st.session_state.demo_embeddings,
                st.session_state.demo_metadata,
                top_k=3
            )

            st.subheader("🔎 Match Results")

            best_score, best_match = matches[0]

            # ---- SAFE CONFIDENCE PRESENTATION ----
            # FaceNet similarity usually:
            # 0.35–0.45 = different person
            # 0.55–0.75 = same person
            confidence = round((best_score - 0.35) / (0.75 - 0.35) * 100, 2)
            confidence = max(0, min(confidence, 100))

            if best_score >= 0.55:
                if confidence > 80:
                    level = "High confidence"
                elif confidence > 60:
                    level = "Medium confidence"
                else:
                    level = "Low confidence"

                st.success(f"Name: {best_match['name']}")
                st.write(f"Confidence: {confidence}% ({level})")
            else:
                st.error("❌ No confident match found.")

st.caption("Demo data is session-based and cleared on refresh.")
