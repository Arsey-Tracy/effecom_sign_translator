import streamlit as st
import os
import uuid
from typing import Dict, Any, List, Tuple
try:
    from ..config import config
except ImportError:
    from config import config


GREEN_PRIMARY = "#2E7D32"
GREEN_DARK = "#1B5E20"


def render_config_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown(
        f'<div class="health-header">🩺 Training Configuration</div>',
        unsafe_allow_html=True,
    )

    config_override = {}

    with st.sidebar.expander("📋 Actions & Data", expanded=True):
        if "configured_actions" not in st.session_state:
            st.session_state.configured_actions = [
                {"id": f"default_{index}", "name": action}
                for index, action in enumerate(config.actions)
            ]
        if "new_action_key" not in st.session_state:
            st.session_state.new_action_key = f"new_action_{uuid.uuid4().hex}"

        action_rows = st.session_state.configured_actions
        st.markdown("**Sign Classes**")

        new_actions = []
        for action_row in action_rows:
            action_id = action_row["id"]
            action_name = action_row["name"]
            col1, col2 = st.columns([4, 1])
            with col1:
                new_action = st.text_input(
                    "Sign class",
                    value=action_name,
                    key=f"action_{action_id}",
                    label_visibility="collapsed",
                )
            with col2:
                remove_action = st.button("✕", key=f"del_action_{action_id}", help="Remove this class")
            if remove_action:
                st.session_state.configured_actions = [
                    {
                        "id": row["id"],
                        "name": st.session_state.get(f"action_{row['id']}", row["name"]),
                    }
                    for row in action_rows
                    if row["id"] != action_id
                ]
                st.rerun()
            new_actions.append({"id": action_id, "name": new_action})

        st.session_state.configured_actions = new_actions
        new_action_name = st.text_input(
            "Add new class",
            placeholder="e.g., new_sign",
            key=st.session_state.new_action_key,
        )
        if st.button("➕ Add", key="add_action") and new_action_name:
            new_actions.append({"id": uuid.uuid4().hex, "name": new_action_name})
            st.session_state.configured_actions = new_actions
            st.session_state.new_action_key = f"new_action_{uuid.uuid4().hex}"
            st.rerun()

        config_override["actions"] = [
            row["name"] for row in new_actions if row["name"].strip()
        ]

        data_path = st.text_input(
            "Processed Data Path",
            value=config.data_path,
            key="data_path",
            help="Path to .npy sequence files",
        )
        config_override["data_path"] = data_path

        col1, col2 = st.columns(2)
        with col1:
            seq_len = st.number_input(
                "Sequence Length",
                min_value=10,
                max_value=100,
                value=config.sequence_length,
                step=5,
                key="seq_len",
                help="Frames per sequence",
            )
        with col2:
            no_seq = st.number_input(
                "Sequences/Class",
                min_value=5,
                max_value=200,
                value=config.no_sequences,
                step=5,
                key="no_seq",
                help="Sequences per sign class",
            )
        config_override["sequence_length"] = seq_len
        config_override["no_sequences"] = no_seq

    with st.sidebar.expander("🧠 Model Architecture", expanded=False):
        mp = config.model_params
        col1, col2 = st.columns(2)
        with col1:
            hidden_dim = st.slider(
                "Hidden Dim",
                min_value=16,
                max_value=256,
                value=mp.get("hidden_dim", 64),
                step=16,
                key="hidden_dim",
            )
            num_layers = st.slider(
                "LSTM Layers",
                min_value=1,
                max_value=4,
                value=mp.get("num_layers", 2),
                step=1,
                key="num_layers",
            )
        with col2:
            dropout = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=mp.get("dropout", 0.2),
                step=0.05,
                key="dropout",
                format="%.2f",
            )
            input_dim = st.number_input(
                "Input Dim",
                value=mp.get("input_dim", 225),
                disabled=True,
                key="input_dim",
                help="Fixed: 33 pose × 3 + 2 × 21 hand × 3 = 225",
            )
        
        config_override["model_params"] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    with st.sidebar.expander("⚙️ Training Hyperparameters", expanded=False):
        tp = config.training_params
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=10,
                max_value=500,
                value=tp.get("epochs", 100),
                step=10,
                key="epochs",
            )
            batch_size = st.selectbox(
                "Batch Size",
                options=[8, 16, 32, 64],
                index=[8, 16, 32, 64].index(tp.get("batch_size", 16)),
                key="batch_size",
            )
            lr = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=tp.get("learning_rate", 0.001),
                format_func=lambda x: f"{x:.0e}",
                key="lr",
            )
        with col2:
            val_split = st.slider(
                "Val Split",
                min_value=0.1,
                max_value=0.4,
                value=tp.get("val_split", 0.2),
                step=0.05,
                key="val_split",
                format="%.2f",
            )
            patience = st.number_input(
                "Early Stop Patience",
                min_value=5,
                max_value=30,
                value=tp.get("early_stopping_patience", 10),
                step=1,
                key="patience",
            )
            min_delta = st.number_input(
                "Min Delta",
                min_value=0.0001,
                max_value=0.01,
                value=tp.get("early_stopping_min_delta", 0.001),
                step=0.0001,
                format="%.4f",
                key="min_delta",
            )

        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except:
            pass
        
        device_options = ["cpu"]
        if cuda_available:
            device_options.append("cuda")
        
        device = st.radio(
            "Device",
            options=device_options,
            index=device_options.index(tp.get("device", "cpu")) if tp.get("device", "cpu") in device_options else 0,
            key="device",
            horizontal=True,
        )
        
        config_override["training_params"] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "val_split": val_split,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "device": device,
        }

    with st.sidebar.expander("📁 Model Paths", expanded=False):
        paths = config.paths
        hand_model = st.text_input("Hand Model", value=paths.get("hand_model", ""), key="hand_model_path")
        pose_model = st.text_input("Pose Model", value=paths.get("pose_model", ""), key="pose_model_path")
        model_weights = st.text_input("Model Weights", value=paths.get("model_weights", ""), key="model_weights_path")
        best_model_weights = st.text_input("Best Model Weights", value=paths.get("best_model_weights", ""), key="best_model_weights_path")
        
        config_override["paths"] = {
            "hand_model": hand_model,
            "pose_model": pose_model,
            "model_weights": model_weights,
            "best_model_weights": best_model_weights,
        }

    with st.sidebar.expander("🔧 Advanced", expanded=False):
        st.markdown("*Advanced options*")
        if st.button("🔄 Reset to Defaults", key="reset_config"):
            for key in list(st.session_state.keys()):
                if key.startswith(("action_", "data_path", "seq_len", "no_seq", "hidden_dim", 
                                  "num_layers", "dropout", "epochs", "batch_size", "lr",
                                  "val_split", "patience", "min_delta", "device",
                                  "hand_model_path", "pose_model_path", "model_weights_path",
                                  "best_model_weights_path", "new_action")):
                    del st.session_state[key]
            st.session_state.configured_actions = [
                {"id": f"default_{index}", "name": action}
                for index, action in enumerate(config.actions)
            ]
            st.session_state.new_action_key = f"new_action_{uuid.uuid4().hex}"
            st.rerun()

    return config_override


def validate_config(config_override: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    
    actions = config_override.get("actions", [])
    if not actions:
        errors.append("At least one action class is required")
    elif len(set(actions)) != len(actions):
        errors.append("Action names must be unique")
    
    data_path = config_override.get("data_path", "")
    if data_path and not os.path.exists(data_path):
        errors.append(f"Data path does not exist: {data_path}")
    
    seq_len = config_override.get("sequence_length", 30)
    if seq_len < 10:
        errors.append("Sequence length must be at least 10")
    
    model_params = config_override.get("model_params", {})
    if model_params.get("hidden_dim", 0) < 16:
        errors.append("Hidden dim must be at least 16")
    
    training_params = config_override.get("training_params", {})
    if training_params.get("epochs", 0) < 1:
        errors.append("Epochs must be at least 1")
    if training_params.get("batch_size", 0) < 1:
        errors.append("Batch size must be at least 1")
    if training_params.get("learning_rate", 0) <= 0:
        errors.append("Learning rate must be positive")
    if not 0 < training_params.get("val_split", 0) < 1:
        errors.append("Val split must be between 0 and 1")
    
    return len(errors) == 0, errors


def render_config_summary(config_override: Dict[str, Any]):
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div class="health-header">📊 Configuration Summary</div>',
        unsafe_allow_html=True,
    )
    
    actions = config_override.get("actions", [])
    st.sidebar.markdown(f"**Classes:** {len(actions)} ({', '.join(actions[:3])}{'...' if len(actions) > 3 else ''})")
    
    tp = config_override.get("training_params", {})
    st.sidebar.markdown(f"**Epochs:** {tp.get('epochs', 100)} | **Batch:** {tp.get('batch_size', 16)} | **LR:** {tp.get('learning_rate', 0.001):.0e}")
    
    mp = config_override.get("model_params", {})
    st.sidebar.markdown(f"**Hidden:** {mp.get('hidden_dim', 64)} | **Layers:** {mp.get('num_layers', 2)} | **Dropout:** {mp.get('dropout', 0.2):.2f}")
    
    device = tp.get("device", "cpu")
    st.sidebar.markdown(f"**Device:** {device.upper()}")