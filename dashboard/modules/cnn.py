import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
from pathlib import Path
from utils.load_data import get_flood_stats, DATES, DATE_LABELS, MODEL_DIR

def render():
    st.title("CNN Model Results")
    st.caption("SUPPORTING EVIDENCE — U-Net with ResNet18, trained on Sen1Floods11 (India-prioritized 3x)")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Training Curves", "Flood Detection", "Architecture"])

    with tab1:
        render_training_curves()
    with tab2:
        render_flood_detection()
    with tab3:
        render_architecture()


def render_training_curves():
    st.subheader("Training History (30 Epochs)")

    history_path = MODEL_DIR / "training_history.json"
    if not history_path.exists():
        st.error("Training history not found")
        return

    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss", "IoU Score", "Accuracy"))

    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], name="Train Loss", line=dict(color="#98B06F", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], name="Val Loss", line=dict(color="#B6DC76", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_iou'], name="Train IoU", line=dict(color="#98B06F", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_iou'], name="Val IoU", line=dict(color="#B6DC76", width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_acc'], name="Train Acc", line=dict(color="#98B06F", width=2), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_acc'], name="Val Acc", line=dict(color="#B6DC76", width=2), showlegend=False), row=1, col=3)

    best_idx = df['val_iou'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_iou = df.loc[best_idx, 'val_iou']
    fig.add_annotation(x=best_epoch, y=best_iou, text=f"Best: {best_iou:.4f}",
                      showarrow=True, arrowhead=2, ax=30, ay=-30,
                      font=dict(color="#B6DC76"), row=1, col=2)

    fig.update_layout(height=350, template="plotly_dark", hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,39,0.5)",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="Epoch", gridcolor="#30363D")
    fig.update_yaxes(gridcolor="#30363D")

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Val IoU", f"{best_iou:.4f}", f"Epoch {best_epoch}")
    c2.metric("Best Val Acc", f"{df['val_acc'].max():.1%}")
    c3.metric("Final Train Loss", f"{df['train_loss'].iloc[-1]:.4f}")
    c4.metric("Total Epochs", "30")


def render_flood_detection():
    st.subheader("CNN Flood Water Detection")

    stats = get_flood_stats()
    dates = [s['date'] for s in stats]
    thresh = [s['threshold'] for s in stats]
    cnn = [s['cnn'] for s in stats]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=thresh, name="Threshold (primary)",
        line=dict(color="#98B06F", width=3), mode="lines+markers", marker=dict(size=8),
        hovertemplate="%{x}<br>Threshold: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=cnn, name="CNN (supporting)",
        line=dict(color="#B6DC76", width=3, dash="dash"), mode="lines+markers",
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="%{x}<br>CNN: %{y:.2f}%<extra></extra>"
    ))

    peak_t = thresh[3]
    peak_c = cnn[3]

    fig.add_annotation(x='2018-08-21', y=peak_t, text=f"Threshold: {peak_t:.2f}% (+73%)",
                      showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="#98B06F"))
    fig.add_annotation(x='2018-08-21', y=peak_c, text=f"CNN: {peak_c:.2f}% (+17%)",
                      showarrow=True, arrowhead=2, ax=0, ay=30, font=dict(color="#B6DC76"))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Flood Water %",
        template="plotly_dark", height=400, hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,39,0.5)",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_xaxes(gridcolor="#30363D")
    fig.update_yaxes(gridcolor="#30363D")

    st.plotly_chart(fig, use_container_width=True)

    st.warning("**Domain Gap Note:** CNN baseline (19.73%) is ~9x higher than threshold (2.28%). "
               "Trained on different dataset/domain. Both methods agree on relative trend — Aug 21 is peak.")


def render_architecture():
    st.subheader("Model Architecture")

    st.code("""
U-Net with ResNet18 Encoder
├── Encoder: ResNet18 (pretrained ImageNet)
│   ├── conv1: 3 → 64
│   ├── layer1: 64 → 64 (2 blocks)
│   ├── layer2: 64 → 128 (2 blocks)
│   ├── layer3: 128 → 256 (2 blocks)
│   └── layer4: 256 → 512 (2 blocks)
├── Decoder: UnetDecoder
│   ├── decoder4: 512 → 256
│   ├── decoder3: 256 → 128
│   ├── decoder2: 128 → 64
│   └── decoder1: 64 → 32
└── Head: 1×1 conv (32 → 1, sigmoid)
    """, language=None)

    st.subheader("Training Configuration")

    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
    | Loss | Focal Loss (α=0.75, γ=2) + BCE |
    | LR Scheduler | Cosine Annealing (T_max=30) |
    | Input | 128×128 patches, 2 channels (VH duplicated) |
    | Batch Size | 8 |
    | Augmentation | Random flip, 90° rotation |
    | India Oversampling | 3× weight (40 → 120 effective) |
    """)

    st.subheader("Training Data")

    st.markdown("""
    | Dataset | Samples | Source |
    |---------|---------|--------|
    | Training | 252 | Sen1Floods11 `flood_train_data.csv` |
    | Validation | 89 | Sen1Floods11 `flood_valid_data.csv` |
    | India Chips | 40 (~16%) | Oversampled 3× per epoch |
    | Effective/Epoch | 332 | 252 + (40 × 3) |
    """)
