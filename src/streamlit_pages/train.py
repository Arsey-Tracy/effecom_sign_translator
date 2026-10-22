import streamlit as st
import time
import queue
import threading
import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from streamlit_utils.state import (
    init_training_state,
    get_training_state,
    reset_training_state,
    update_training_state,
    add_log_line,
    add_epoch_metrics,
    get_metrics_for_chart,
    get_elapsed_time,
    get_eta,
    format_time,
    TrainingStatus,
    EpochMetrics,
)
from streamlit_utils.worker import create_worker
from streamlit_utils.config_ui import render_config_sidebar, validate_config, render_config_summary


GREEN_PRIMARY = "#2E7D32"
GREEN_DARK = "#1B5E20"


def render_metrics_cards(state):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Epoch",
            f"{state.current_epoch}/{state.total_epochs}",
            delta=f"{state.current_epoch/state.total_epochs*100:.0f}%" if state.total_epochs > 0 else None,
        )
    
    with col2:
        if state.metrics_history:
            last = state.metrics_history[-1]
            st.metric("Train Loss", f"{last.train_loss:.4f}")
        else:
            st.metric("Train Loss", "—")
    
    with col3:
        if state.metrics_history:
            last = state.metrics_history[-1]
            st.metric("Val Loss", f"{last.val_loss:.4f}")
        else:
            st.metric("Val Loss", "—")
    
    with col4:
        if state.metrics_history:
            last = state.metrics_history[-1]
            st.metric("Val Acc", f"{last.val_acc:.2f}%")
        else:
            st.metric("Val Acc", "—")
    
    with col5:
        if state.best_val_loss != float("inf"):
            st.metric("Best Val Loss", f"{state.best_val_loss:.4f}", delta=f"Epoch {state.best_epoch}")
        else:
            st.metric("Best Val Loss", "—")


def render_charts(state):
    metrics = get_metrics_for_chart()
    
    if not metrics["epoch"]:
        st.info("📈 Charts will appear once training starts")
        return
    
    import pandas as pd
    
    df = pd.DataFrame({
        "Epoch": metrics["epoch"],
        "Train Loss": metrics["train_loss"],
        "Val Loss": metrics["val_loss"],
        "Val Accuracy (%)": metrics["val_acc"],
    })
    
    tab1, tab2 = st.tabs(["📉 Loss Curves", "📈 Accuracy"])
    
    with tab1:
        st.line_chart(df.set_index("Epoch")[["Train Loss", "Val Loss"]], height=300)
    
    with tab2:
        st.line_chart(df.set_index("Epoch")[["Val Accuracy (%)"]], height=300)
    
    if state.best_epoch > 0:
        st.caption(f"★ Best model at epoch {state.best_epoch} (Val Loss: {state.best_val_loss:.4f})")


def render_progress_bar(state):
    if state.total_epochs > 0:
        progress = state.current_epoch / state.total_epochs
        st.progress(progress, text=f"Epoch {state.current_epoch} of {state.total_epochs}")
    else:
        st.progress(0, text="Waiting to start...")


def render_time_info(state):
    elapsed = get_elapsed_time()
    eta = get_eta()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="health-card"><div class="health-header">⏱️ Elapsed</div><div class="health-text">{format_time(elapsed)}</div></div>', unsafe_allow_html=True)
    with col2:
        if eta is not None:
            st.markdown(f'<div class="health-card"><div class="health-header">⏳ ETA</div><div class="health-text">{format_time(eta)}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="health-card"><div class="health-header">⏳ ETA</div><div class="health-text">—</div></div>', unsafe_allow_html=True)
    with col3:
        if state.metrics_history:
            avg_epoch = elapsed / state.current_epoch if state.current_epoch > 0 else 0
            st.markdown(f'<div class="health-card"><div class="health-header">⚡ Avg/Epoch</div><div class="health-text">{format_time(avg_epoch)}</div></div>', unsafe_allow_html=True)


def render_data_summary(state):
    if state.data_summary:
        ds = state.data_summary
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sequences", ds.get("total_sequences", 0))
                st.metric("Classes", ds.get("num_classes", 0))
            with col2:
                dist = ds.get("class_distribution", {})
                if dist:
                    st.markdown("**Class Distribution:**")
                    for cls, count in dist.items():
                        st.text(f"  {cls}: {count}")


def collect_video_sequence(video_bytes, filename, action, data_path, sequence_length, paths):
    from extractor import USLFeatureExtractor

    suffix = os.path.splitext(filename)[1] or ".mp4"
    temporary_path = None
    capture = None
    extractor = None
    sequence_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < sequence_length:
            raise ValueError(f"Video has {frame_count} frames; at least {sequence_length} are required.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        sampled_frames = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
        extractor = USLFeatureExtractor(paths["hand_model"], paths["pose_model"])
        features = []
        for frame_index in sampled_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Could not read frame {frame_index} from the uploaded video.")
            timestamp_ms = int(frame_index * 1000 / fps)
            features.append(extractor.process_frame(frame, timestamp_ms))

        action_path = os.path.join(data_path, action)
        os.makedirs(action_path, exist_ok=True)
        existing = [
            int(name) for name in os.listdir(action_path)
            if name.isdigit() and os.path.isdir(os.path.join(action_path, name))
        ]
        sequence_path = os.path.join(action_path, str(max(existing, default=-1) + 1))
        os.makedirs(sequence_path)
        for frame_number, feature in enumerate(features):
            np.save(os.path.join(sequence_path, f"{frame_number}.npy"), feature)
        return sequence_path
    except Exception:
        if sequence_path and os.path.isdir(sequence_path):
            shutil.rmtree(sequence_path)
        raise
    finally:
        if extractor is not None:
            extractor.close()
        if capture is not None:
            capture.release()
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def render_collection_section(config_override):
    st.markdown("## Collect Training Data")
    actions = config_override["actions"]
    if not actions:
        st.warning("Add at least one sign class in the sidebar before collecting clips.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        action = st.selectbox("Sign class", actions, key="collection_action")
    with col2:
        uploaded_video = st.file_uploader(
            "Upload a recorded sign clip",
            type=["mp4", "mov", "avi", "mkv"],
            key="collection_video",
        )

    if st.button("Save clip as training sequence", type="primary", disabled=uploaded_video is None):
        try:
            with st.spinner("Extracting landmarks and saving sequence..."):
                saved_path = collect_video_sequence(
                    uploaded_video.getvalue(),
                    uploaded_video.name,
                    action,
                    config_override["data_path"],
                    int(config_override["sequence_length"]),
                    config_override["paths"],
                )
            st.success(f"Saved one {config_override['sequence_length']}-frame sequence for '{action}' to {saved_path}.")
        except Exception as error:
            st.error(f"Could not collect this clip: {error}")

    st.markdown("### Dataset Inventory")
    inventory = {}
    for class_name in actions:
        class_path = os.path.join(config_override["data_path"], class_name)
        complete = 0
        if os.path.isdir(class_path):
            for sequence_name in os.listdir(class_path):
                sequence_path = os.path.join(class_path, sequence_name)
                if os.path.isdir(sequence_path):
                    frame_count = sum(name.endswith(".npy") for name in os.listdir(sequence_path))
                    if frame_count == config_override["sequence_length"]:
                        complete += 1
        inventory[class_name] = complete
    st.dataframe(
        [{"Sign class": name, "Complete sequences": count} for name, count in inventory.items()],
        hide_index=True,
        use_container_width=True,
    )


def render_logs(state):
    st.markdown("### 📝 Training Logs")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR"], key="log_level")
    with col2:
        auto_scroll = st.checkbox("Auto-scroll", value=True, key="auto_scroll")
    with col3:
        if state.log_lines:
            log_text = "\n".join(state.log_lines)
            st.download_button(
                "💾 Download Logs",
                data=log_text,
                file_name=f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
    
    filtered_logs = state.log_lines
    if log_level != "All":
        filtered_logs = [l for l in state.log_lines if log_level in l]
    
    log_container = st.container()
    with log_container:
        if filtered_logs:
            st.code("\n".join(filtered_logs[-200:]), language="text", line_numbers=False)
        else:
            st.text("No logs yet...")
    
    if auto_scroll and filtered_logs:
        st.components.v1.html(
            """
            <script>
                var container = window.parent.document.querySelector('[data-testid="stCodeBlock"]:last-child');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            </script>
            """,
            height=0,
        )


def render_control_buttons(state):
    st.markdown("### 🎮 Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if state.status == TrainingStatus.IDLE:
            if st.button("▶️ Start Training", type="primary", use_container_width=True, key="btn_start"):
                return "start"
        elif state.status == TrainingStatus.RUNNING:
            if st.button("⏸️ Pause", type="secondary", use_container_width=True, key="btn_pause"):
                return "pause"
        elif state.status == TrainingStatus.PAUSED:
            if st.button("▶️ Resume", type="primary", use_container_width=True, key="btn_resume"):
                return "resume"
    
    with col2:
        if state.status in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True, key="btn_stop"):
                return "stop"
    
    with col3:
        if state.status == TrainingStatus.COMPLETED:
            if st.button("🔄 New Training", type="primary", use_container_width=True, key="btn_new"):
                return "reset"
    
    with col4:
        if state.status == TrainingStatus.ERROR:
            if st.button("🔄 Retry", type="primary", use_container_width=True, key="btn_retry"):
                return "reset"
    
    return None


def render_results(state):
    if state.status != TrainingStatus.COMPLETED:
        return
    
    st.markdown("---")
    st.markdown("## ✅ Training Completed")

    if state.final_results:
        st.caption(f"Completed {state.final_results.get('total_epochs_run', state.current_epoch)} epochs; best validation epoch: {state.final_results.get('best_epoch', state.best_epoch)}.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Loss", f"{state.metrics_history[-1].train_loss:.4f}" if state.metrics_history else "—")
    with col2:
        st.metric("Final Val Loss", f"{state.metrics_history[-1].val_loss:.4f}" if state.metrics_history else "—")
    with col3:
        st.metric("Final Val Acc", f"{state.metrics_history[-1].val_acc:.2f}%" if state.metrics_history else "—")
    with col4:
        st.metric("Best Val Loss", f"{state.best_val_loss:.4f}")
    
    if state.final_model_path and os.path.exists(state.final_model_path):
        with open(state.final_model_path, "rb") as f:
            st.download_button(
                "📥 Download Final Model",
                data=f.read(),
                file_name="usl_model.pt",
                mime="application/octet-stream",
            )
    
    if state.best_model_path and os.path.exists(state.best_model_path):
        with open(state.best_model_path, "rb") as f:
            st.download_button(
                "📥 Download Best Model",
                data=f.read(),
                file_name="usl_model_best.pt",
                mime="application/octet-stream",
            )
    
    config_snapshot = state.config_snapshot
    if config_snapshot:
        config_json = json.dumps(config_snapshot, indent=2)
        st.download_button(
            "📥 Download Config Snapshot",
            data=config_json,
            file_name=f"config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


def process_queues(log_queue: queue.Queue, metrics_queue: queue.Queue, stop_event: threading.Event, pause_event: threading.Event):
    while not log_queue.empty():
        try:
            line = log_queue.get_nowait()
            add_log_line(line)
        except queue.Empty:
            break
    
    while not metrics_queue.empty():
        try:
            msg_type, data = metrics_queue.get_nowait()
            
            if msg_type == "epoch_metrics":
                metrics = EpochMetrics(**data)
                add_epoch_metrics(metrics)
            elif msg_type == "total_epochs":
                update_training_state(total_epochs=data)
            elif msg_type == "data_summary":
                update_training_state(data_summary=data)
            elif msg_type == "best_model_saved":
                pass
            elif msg_type == "early_stop":
                pass
            elif msg_type == "completed":
                update_training_state(
                    status=TrainingStatus.COMPLETED,
                    final_model_path=data.get("final_model_path"),
                    best_model_path=data.get("best_model_path"),
                    final_results=data,
                )
            elif msg_type == "error":
                update_training_state(status=TrainingStatus.ERROR, error_message=data)
            elif msg_type == "paused":
                update_training_state(status=TrainingStatus.PAUSED)
            elif msg_type == "stopped":
                update_training_state(status=TrainingStatus.IDLE)
            elif msg_type == "best_model_path":
                update_training_state(best_model_path=data)
            elif msg_type == "final_model_path":
                update_training_state(final_model_path=data)
        except queue.Empty:
            break


def render_train_page():
    init_training_state()
    state = get_training_state()
    
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if "metrics_queue" not in st.session_state:
        st.session_state.metrics_queue = queue.Queue()
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if "pause_event" not in st.session_state:
        st.session_state.pause_event = threading.Event()
    if "worker" not in st.session_state:
        st.session_state.worker = None
    
    log_queue = st.session_state.log_queue
    metrics_queue = st.session_state.metrics_queue
    stop_event = st.session_state.stop_event
    pause_event = st.session_state.pause_event
    
    config_override = render_config_sidebar()
    render_config_summary(config_override)

    render_collection_section(config_override)
    
    is_valid, errors = validate_config(config_override)
    if not is_valid:
        for err in errors:
            st.sidebar.error(err)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        render_progress_bar(state)
        render_metrics_cards(state)
        render_time_info(state)
        render_charts(state)
        render_data_summary(state)
        render_logs(state)
    
    with col_right:
        action = render_control_buttons(state)
        
        if action == "start" and is_valid:
            reset_training_state()
            update_training_state(
                status=TrainingStatus.RUNNING,
                config_snapshot=config_override,
                start_time=time.time(),
            )
            stop_event.clear()
            pause_event.clear()
            
            worker = create_worker(config_override, log_queue, metrics_queue, stop_event, pause_event)
            st.session_state.worker = worker
            worker.start()
            st.rerun()
        
        elif action == "pause":
            pause_event.set()
            update_training_state(status=TrainingStatus.PAUSED)
            st.rerun()
        
        elif action == "resume":
            pause_event.clear()
            update_training_state(status=TrainingStatus.RUNNING)
            st.rerun()
        
        elif action == "stop":
            stop_event.set()
            if st.session_state.worker and st.session_state.worker.is_alive():
                st.session_state.worker.join(timeout=5)
            update_training_state(status=TrainingStatus.IDLE)
            st.rerun()
        
        elif action == "reset":
            reset_training_state()
            st.session_state.log_queue = queue.Queue()
            st.session_state.metrics_queue = queue.Queue()
            st.session_state.stop_event = threading.Event()
            st.session_state.pause_event = threading.Event()
            st.session_state.worker = None
            st.rerun()
    
    if state.status in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
        process_queues(log_queue, metrics_queue, stop_event, pause_event)
        time.sleep(0.5)
        st.rerun()
    
    elif state.status == TrainingStatus.ERROR:
        st.error(f"❌ Training Error: {state.error_message}")
        render_results(state)
    
    elif state.status == TrainingStatus.COMPLETED:
        render_results(state)
    
    if state.status == TrainingStatus.IDLE and not state.metrics_history:
        st.markdown(
            f"""
            <div class="health-card" style="text-align: center; padding: 3rem;">
                <div class="health-header" style="font-size: 1.5rem;">🩺 Ready to Train</div>
                <div class="health-text" style="font-size: 1.1rem; margin-top: 1rem;">
                    Configure your training parameters in the sidebar and click <strong>Start Training</strong> to begin.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_train_page()