import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

# ============================================================
# DEFECT CONFIGURATION
# ============================================================

DEFECT_WEIGHTS = {
    "crazing": 0.8,
    "inclusion": 0.9,
    "patches": 0.7,
    "pitted_surface": 0.8,
    "rolled_in_scale": 1.0,
    "scratches": 1.0,
}

EXPECTED_CLASSES = list(DEFECT_WEIGHTS.keys())

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SteelGuard AI",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.title("🔍 SteelGuard AI")

st.markdown(
    "### Real-time stainless-steel surface defect inspection"
)

st.caption(
    "AI detection → confidence → severity → quality analytics"
)

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained YOLO model.

    The model path is resolved relative to this app.py file,
    so Streamlit can be launched from different directories.
    """

    if not MODEL_PATH.exists():
        st.error(
            "❌ SteelGuard AI model was not found."
        )

        st.code(str(MODEL_PATH))

        st.info(
            "Make sure best.pt exists at the path shown above."
        )

        return None

    try:
        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))

        return model

    except Exception as exc:
        st.error(
            f"❌ Failed to load SteelGuard AI model:\n\n{exc}"
        )

        return None


# ============================================================
# SEVERITY CALCULATION
# ============================================================

def severity_score(
    confidence,
    area_ratio,
    defect_type,
):
    """
    Calculate a prototype severity score.

    NOTE:
    This is a prototype business-rule score, not an
    industrially validated quality standard.
    """

    weight = DEFECT_WEIGHTS.get(
        defect_type.lower(),
        0.8,
    )

    score = min(
        100.0,
        confidence
        * 100
        * (0.55 + 2.0 * area_ratio)
        * weight,
    )

    if score >= 70:
        level = "CRITICAL"

    elif score >= 35:
        level = "HIGH"

    elif score >= 15:
        level = "MEDIUM"

    else:
        level = "LOW"

    return round(score, 1), level


# ============================================================
# YOLO PREDICTION
# ============================================================

def predict(
    image,
    model,
    confidence_threshold,
):
    """
    Run SteelGuard AI YOLO inference on an uploaded image.
    """

    if model is None:
        return None, []

    # --------------------------------------------------------
    # Convert PIL image to RGB numpy array
    # --------------------------------------------------------

    image_rgb = image.convert("RGB")

    image_array = np.array(image_rgb)

    # --------------------------------------------------------
    # Run YOLO inference
    # --------------------------------------------------------

    results = model.predict(
        source=image_array,
        conf=confidence_threshold,
        device="cpu",
        verbose=False,
    )

    # --------------------------------------------------------
    # Get first result
    # --------------------------------------------------------

    result = results[0]

    # --------------------------------------------------------
    # Draw YOLO bounding boxes
    # --------------------------------------------------------

    annotated = result.plot()

    detections = []

    # --------------------------------------------------------
    # Extract detections
    # --------------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        names = result.names

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
        )

        for box, conf, cls in zip(
            boxes,
            confidences,
            classes,
        ):

            class_id = int(cls)

            defect_type = str(
                names[class_id]
            ).lower().replace(" ", "_")

            detections.append(
                {
                    "type": defect_type,
                    "confidence": float(conf),
                    "box": tuple(
                        map(
                            int,
                            box,
                        )
                    ),
                }
            )

    return annotated, detections


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Inspection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    0.10,
    0.95,
    0.50,
    0.05,
)

review_threshold = st.sidebar.slider(
    "Human-review threshold",
    0.20,
    0.95,
    0.75,
    0.05,
)

st.sidebar.divider()

# ------------------------------------------------------------
# Model status
# ------------------------------------------------------------

if MODEL_PATH.exists():

    st.sidebar.success(
        "✅ Trained model found"
    )

else:

    st.sidebar.error(
        "❌ Model not found"
    )

st.sidebar.caption(
    f"Model: {MODEL_PATH.name}"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🧪 Inspection",
        "📊 Analytics",
        "⚙️ Model",
    ]
)

# ============================================================
# INSPECTION TAB
# ============================================================

with tab1:

    uploaded = st.file_uploader(
        "Upload a steel-strip image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
    )

    # --------------------------------------------------------
    # No image uploaded
    # --------------------------------------------------------

    if uploaded is None:

        st.info(
            "Upload a steel surface image to begin inspection."
        )

    # --------------------------------------------------------
    # Image uploaded
    # --------------------------------------------------------

    else:

        try:

            # ------------------------------------------------
            # Read uploaded image
            # ------------------------------------------------

            image = Image.open(uploaded).convert("RGB")

            # ------------------------------------------------
            # Display uploaded image information
            # ------------------------------------------------

            st.caption(
                f"Image: {uploaded.name}  |  "
                f"Resolution: {image.width} × {image.height}"
            )

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            model = load_model()

            # ------------------------------------------------
            # Stop if model unavailable
            # ------------------------------------------------

            if model is None:

                st.error(
                    "SteelGuard AI cannot perform real inference "
                    "because the trained model could not be loaded."
                )

                st.stop()

            # ------------------------------------------------
            # Run prediction
            # ------------------------------------------------

            start = time.perf_counter()

            annotated, raw_detections = predict(
                image,
                model,
                confidence_threshold,
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            # ------------------------------------------------
            # Process detections
            # ------------------------------------------------

            detections = []

            img_area = (
                image.width
                * image.height
            )

            for detection in raw_detections:

                x1, y1, x2, y2 = (
                    detection["box"]
                )

                box_width = max(
                    0,
                    x2 - x1,
                )

                box_height = max(
                    0,
                    y2 - y1,
                )

                box_area = (
                    box_width
                    * box_height
                )

                area_ratio = (
                    box_area
                    / img_area
                    if img_area > 0
                    else 0
                )

                score, severity = (
                    severity_score(
                        detection["confidence"],
                        area_ratio,
                        detection["type"],
                    )
                )

                # --------------------------------------------
                # Decide recommended action
                # --------------------------------------------

                if (
                    detection["confidence"]
                    < review_threshold
                ):

                    action = "HUMAN REVIEW"

                elif severity == "CRITICAL":

                    action = "IMMEDIATE ALERT"

                else:

                    action = "FLAG"

                detections.append(
                    {
                        "Defect": (
                            detection["type"]
                            .replace("_", " ")
                            .title()
                        ),

                        "Confidence": (
                            f"{detection['confidence'] * 100:.1f}%"
                        ),

                        "Severity": severity,

                        "Score": score,

                        "Action": action,
                    }
                )

            # ------------------------------------------------
            # Main display columns
            # ------------------------------------------------

            col1, col2 = st.columns(
                [2, 1]
            )

            # ------------------------------------------------
            # Annotated image
            # ------------------------------------------------

            with col1:

                st.subheader(
                    "AI Inspection"
                )

                st.image(
                    annotated,
                    caption="SteelGuard AI annotated inspection",
                    use_container_width=True,
                )

            # ------------------------------------------------
            # Inspection result
            # ------------------------------------------------

            with col2:

                st.subheader(
                    "Inspection Result"
                )

                st.metric(
                    "Defects detected",
                    len(detections),
                )

                st.metric(
                    "Inference latency",
                    f"{latency_ms:.1f} ms",
                )

                # --------------------------------------------
                # Highest severity defect
                # --------------------------------------------

                if detections:

                    highest = max(
                        detections,
                        key=lambda x: x["Score"],
                    )

                    if (
                        highest["Severity"]
                        == "CRITICAL"
                    ):

                        st.error(
                            f"🔴 {highest['Severity']}: "
                            f"{highest['Defect']}"
                        )

                    elif (
                        highest["Severity"]
                        == "HIGH"
                    ):

                        st.warning(
                            f"🟠 {highest['Severity']}: "
                            f"{highest['Defect']}"
                        )

                    else:

                        st.info(
                            f"🟡 {highest['Severity']}: "
                            f"{highest['Defect']}"
                        )

                else:

                    st.success(
                        "✅ NO DEFECT ABOVE THRESHOLD"
                    )

            # ------------------------------------------------
            # Defect details
            # ------------------------------------------------

            if detections:

                st.subheader(
                    "Defect Details"
                )

                detection_df = pd.DataFrame(
                    detections
                )

                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # --------------------------------------------
                # Defect distribution
                # --------------------------------------------

                st.subheader(
                    "Detected Defect Distribution"
                )

                defect_counts = (
                    detection_df[
                        "Defect"
                    ]
                    .value_counts()
                )

                st.bar_chart(
                    defect_counts
                )

        except Exception as exc:

            st.error(
                "❌ An error occurred while processing "
                "the uploaded image."
            )

            st.exception(exc)


# ============================================================
# ANALYTICS TAB
# ============================================================

with tab2:

    st.subheader(
        "Production Quality Dashboard"
    )

    # --------------------------------------------------------
    # Prototype analytics data
    # --------------------------------------------------------

    sample = pd.DataFrame(
        {
            "Defect": [
                "Scratch",
                "Rolled In Scale",
                "Inclusion",
                "Crazing",
                "Pitted Surface",
                "Patches",
            ],

            "Count": [
                31,
                18,
                12,
                9,
                7,
                5,
            ],
        }
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Inspected length",
        "1,840 m",
    )

    c2.metric(
        "Total defects",
        int(
            sample["Count"].sum()
        ),
    )

    c3.metric(
        "Critical defects",
        "7",
    )

    c4.metric(
        "False-alarm rate",
        "3.2%",
    )

    # --------------------------------------------------------
    # Defect distribution
    # --------------------------------------------------------

    st.bar_chart(
        sample.set_index("Defect")
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    st.subheader(
        "Defect Heatmap / Roll Timeline"
    )

    heat = np.zeros(
        (8, 40)
    )

    rng = np.random.default_rng(
        42
    )

    for _ in range(25):

        heat[
            rng.integers(
                0,
                8,
            ),
            rng.integers(
                0,
                40,
            ),
        ] += rng.integers(
            1,
            5,
        )

    st.image(
        heat,
        clamp=True,
        caption="Prototype defect-position heatmap",
    )

    # --------------------------------------------------------
    # Human-in-loop feedback
    # --------------------------------------------------------

    st.subheader(
        "Human-in-the-loop feedback"
    )

    feedback = pd.DataFrame(
        {
            "Prediction": [
                "Scratch",
                "Inclusion",
                "Scale",
                "Scratch",
            ],

            "AI confidence": [
                "96%",
                "61%",
                "88%",
                "54%",
            ],

            "Operator label": [
                "Correct",
                "False alarm",
                "Correct",
                "Correct",
            ],
        }
    )

    st.dataframe(
        feedback,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL TAB
# ============================================================

with tab3:

    st.subheader(
        "Model / Deployment Information"
    )

    # --------------------------------------------------------
    # Model status
    # --------------------------------------------------------

    model_exists = MODEL_PATH.exists()

    st.write(
        f"**Weights found:** "
        f"{'Yes ✅' if model_exists else 'No ❌'}"
    )

    st.write(
        f"**Model path:** `{MODEL_PATH}`"
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    st.write(
        "**Expected classes:** "
        + ", ".join(
            EXPECTED_CLASSES
        )
    )

    # --------------------------------------------------------
    # Deployment
    # --------------------------------------------------------

    st.write(
        "**Deployment:** "
        "Ultralytics YOLO11 → PyTorch GPU inference"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.write(
        "**Tracked metrics:** "
        "Precision, Recall, mAP50, mAP50-95, "
        "latency and FPS"
    )

    # --------------------------------------------------------
    # Current baseline
    # --------------------------------------------------------

    st.subheader(
        "Current Validation Baseline"
    )

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Precision",
        "77.6%",
    )

    metric_col2.metric(
        "Recall",
        "59.8%",
    )

    metric_col3, metric_col4 = st.columns(2)

    metric_col3.metric(
        "mAP@50",
        "70.9%",
    )

    metric_col4.metric(
        "mAP@50-95",
        "35.3%",
    )

    st.caption(
        "Baseline measured on the current validation set."
    )

    # --------------------------------------------------------
    # Run commands
    # --------------------------------------------------------

    st.subheader(
        "Run Commands"
    )

    st.code(
        """
Training:
    python train.py

Export:
    python export.py

Run dashboard:
    streamlit run app.py
        """,
        language="text",
    )
