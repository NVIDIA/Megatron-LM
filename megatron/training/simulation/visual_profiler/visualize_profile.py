"""
LLM Training Profile Visualizer

Interactive visualization tool for analyzing torch profiler timeline traces.
Visualizes VPP schedule timelines and module-level performance statistics.

Usage:
    streamlit run visualize_profile.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os

# Import the analyzer
from process_profile import MultiPPAnalyzer


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="LLM Profile Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_multi_pp_timeline_gantt(merged_result: Dict[str, Any]) -> go.Figure:
    """
    Create Gantt chart for multi-PP rank VPP schedule timeline.
    Each PP rank has two rows: Compute (forward/backward/P2P) and Optimizer.

    Args:
        merged_result: Result from MultiPPAnalyzer.analyze_all()

    Returns:
        Plotly figure with multi-rank Gantt chart
    """
    pp_degree = merged_result.get('pp_degree', 1)
    ranks_data = merged_result.get('ranks', {})

    if not ranks_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No multi-PP timeline data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Color scheme for compute row
    compute_color_map = {
        'Forward': '#3498db',
        'Backward': '#e74c3c',
        'P2P Send': '#2ecc71',
        'P2P Recv': '#f39c12',
    }
    # Color scheme for optimizer row
    opt_color_map = {
        'grad_reduce_scatter': '#9b59b6',
        'allreduce': '#1abc9c',
        'loss_postprocessing': '#f1c40f',
        'optimizer_step': '#e67e22',
        'training_log': '#7f8c8d',
    }
    opt_display_names = {
        'grad_reduce_scatter': 'GradRS',
        'allreduce': 'AllReduce',
        'loss_postprocessing': 'LossPost',
        'optimizer_step': 'OptStep',
        'training_log': 'TrainLog',
    }

    fig = go.Figure()

    # Check if any optimizer events exist
    has_optimizer = any(
        len(ranks_data.get(r, {}).get('optimizer_events', [])) > 0
        for r in range(pp_degree)
    )

    # Collect compute ops and optimizer ops per rank
    rank_compute_ops = {}
    rank_optimizer_ops = {}

    for rank_idx in range(pp_degree):
        rd = ranks_data.get(rank_idx, {})
        compute_label = f"PP Rank {rank_idx} - Compute" if has_optimizer else f"PP Rank {rank_idx}"
        optimizer_label = f"PP Rank {rank_idx} - Optimizer"
        compute_ops = []

        for step in rd.get('forward_steps', []):
            compute_ops.append({
                'start_ms': step['start_ms'], 'end_ms': step['end_ms'],
                'duration_ms': step['duration_ms'], 'type': 'Forward',
                'label': step.get('label', ''), 'rank': compute_label,
            })
        for step in rd.get('backward_steps', []):
            compute_ops.append({
                'start_ms': step['start_ms'], 'end_ms': step['end_ms'],
                'duration_ms': step['duration_ms'], 'type': 'Backward',
                'label': step.get('label', ''), 'rank': compute_label,
            })
        for step in rd.get('p2p_sends', []):
            compute_ops.append({
                'start_ms': step['start_ms'], 'end_ms': step['end_ms'],
                'duration_ms': step['duration_ms'], 'type': 'P2P Send',
                'label': step.get('label', 'Send'), 'rank': compute_label,
            })
        for step in rd.get('p2p_recvs', []):
            compute_ops.append({
                'start_ms': step['start_ms'], 'end_ms': step['end_ms'],
                'duration_ms': step['duration_ms'], 'type': 'P2P Recv',
                'label': step.get('label', 'Recv'), 'rank': compute_label,
            })

        rank_compute_ops[rank_idx] = sorted(compute_ops, key=lambda x: x['start_ms'])

        # Optimizer events
        opt_ops = []
        for evt in rd.get('optimizer_events', []):
            if evt.get('start_ms') is not None:
                opt_ops.append({
                    'start_ms': evt['start_ms'], 'end_ms': evt['end_ms'],
                    'duration_ms': evt['duration_ms'],
                    'phase': evt.get('phase', ''),
                    'label': evt.get('label', ''),
                    'rank': optimizer_label,
                })
        rank_optimizer_ops[rank_idx] = sorted(opt_ops, key=lambda x: x['start_ms'])

    # Calculate bubbles on compute row
    bubbles = []
    for rank_idx in range(pp_degree):
        ops = rank_compute_ops.get(rank_idx, [])
        for i in range(len(ops) - 1):
            bubble_start = ops[i]['end_ms']
            bubble_end = ops[i + 1]['start_ms']
            bubble_duration = bubble_end - bubble_start
            if bubble_duration > 0.5:
                bubbles.append({
                    'rank': ops[i]['rank'],
                    'start_ms': bubble_start,
                    'duration_ms': bubble_duration,
                    'end_ms': bubble_end,
                })

    if bubbles:
        fig.add_trace(go.Bar(
            x=[b['duration_ms'] for b in bubbles],
            y=[b['rank'] for b in bubbles],
            base=[b['start_ms'] for b in bubbles],
            orientation='h',
            name='Bubble (Idle)',
            marker=dict(
                color='rgba(200, 200, 200, 0.3)',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1)
            ),
            hovertemplate=(
                '<b>Bubble</b><br>'
                'Row: %{y}<br>'
                'Start: %{base:.2f} ms<br>'
                'End: %{customdata[0]:.2f} ms<br>'
                'Duration: %{x:.2f} ms<br>'
                '<extra></extra>'
            ),
            customdata=[[b['end_ms']] for b in bubbles],
            showlegend=True,
        ))

    # Add compute operation bars grouped by type
    for op_type in ['Forward', 'Backward', 'P2P Send', 'P2P Recv']:
        all_ops_of_type = []
        for rank_idx in range(pp_degree):
            for op in rank_compute_ops.get(rank_idx, []):
                if op['type'] == op_type:
                    all_ops_of_type.append(op)

        if not all_ops_of_type:
            continue

        fig.add_trace(go.Bar(
            x=[op['duration_ms'] for op in all_ops_of_type],
            y=[op['rank'] for op in all_ops_of_type],
            base=[op['start_ms'] for op in all_ops_of_type],
            orientation='h',
            name=op_type,
            marker=dict(
                color=compute_color_map.get(op_type, '#7f7f7f'),
                line=dict(color='white', width=1)
            ),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Type: ' + op_type + '<br>'
                'Row: %{y}<br>'
                'Start: %{base:.2f} ms<br>'
                'End: %{customdata[1]:.2f} ms<br>'
                'Duration: %{x:.2f} ms<br>'
                '<extra></extra>'
            ),
            customdata=[[op['label'], op['end_ms']] for op in all_ops_of_type],
            text=[op['label'] for op in all_ops_of_type],
            textposition='inside',
            textfont=dict(color='white', size=10, family='Arial Black'),
        ))

    # Add optimizer bars grouped by phase
    if has_optimizer:
        for phase_key in ['grad_reduce_scatter', 'allreduce', 'loss_postprocessing',
                          'optimizer_step', 'training_log']:
            all_phase_ops = []
            for rank_idx in range(pp_degree):
                for op in rank_optimizer_ops.get(rank_idx, []):
                    if op['phase'] == phase_key:
                        all_phase_ops.append(op)

            if not all_phase_ops:
                continue

            display_name = opt_display_names.get(phase_key, phase_key)
            fig.add_trace(go.Bar(
                x=[op['duration_ms'] for op in all_phase_ops],
                y=[op['rank'] for op in all_phase_ops],
                base=[op['start_ms'] for op in all_phase_ops],
                orientation='h',
                name=display_name,
                marker=dict(
                    color=opt_color_map.get(phase_key, '#7f7f7f'),
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Phase: ' + display_name + '<br>'
                    'Row: %{y}<br>'
                    'Start: %{base:.2f} ms<br>'
                    'End: %{customdata[1]:.2f} ms<br>'
                    'Duration: %{x:.2f} ms<br>'
                    '<extra></extra>'
                ),
                customdata=[[op['label'], op['end_ms']] for op in all_phase_ops],
                text=[op['label'] for op in all_phase_ops],
                textposition='inside',
                textfont=dict(color='white', size=10, family='Arial Black'),
            ))

    # Build Y-axis category order: Compute on top, Optimizer below for each rank
    # (reversed later so Rank 0 is at the top of the chart)
    y_labels = []
    for rank_idx in range(pp_degree):
        if has_optimizer:
            y_labels.append(f"PP Rank {rank_idx} - Compute")
            y_labels.append(f"PP Rank {rank_idx} - Optimizer")
        else:
            y_labels.append(f"PP Rank {rank_idx}")

    # Calculate total duration across all events
    total_duration = 0
    for rank_idx in range(pp_degree):
        for ops in [rank_compute_ops.get(rank_idx, []), rank_optimizer_ops.get(rank_idx, [])]:
            if ops:
                total_duration = max(total_duration, max(op['end_ms'] for op in ops))

    summary = merged_result.get('summary', {})
    bubble_ratio = summary.get('bubble_ratio', 0)

    num_rows = pp_degree * 2 if has_optimizer else pp_degree
    fig.update_layout(
        title={
            'text': f'VPP Timeline (PP={pp_degree}, '
                    f'Total={total_duration:.1f}ms, '
                    f'Bubble={bubble_ratio * 100:.1f}%)',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title='Relative Time (ms)',
        yaxis_title='',
        barmode='overlay',
        height=max(400, num_rows * 80 + 120),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        yaxis=dict(
            categoryorder='array',
            categoryarray=y_labels[::-1],
            gridcolor='rgba(200, 200, 200, 0.3)'
        ),
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.3)',
            zerolinewidth=1
        ),
        plot_bgcolor='rgba(250, 250, 250, 0.5)',
        paper_bgcolor='white',
    )

    return fig


def create_module_duration_chart(module_result: Dict[str, Any],
                                 selected_modules: List[str] = None) -> go.Figure:
    """
    Create bar chart for module durations

    Args:
        module_result: Result from analyze_modules()
        selected_modules: List of module names to display (None = all)

    Returns:
        Plotly figure with bar chart
    """
    # Prepare data
    module_stats = []

    for module_name, data in module_result.items():
        if selected_modules and module_name not in selected_modules:
            continue

        summary = data.get('summary', {})
        if summary.get('count', 0) > 0:
            module_stats.append({
                'Module': module_name,
                'Mean Duration (ms)': summary.get('mean_duration_ms', 0),
                'Std Duration (ms)': summary.get('std_duration_ms', 0),
                'Min Duration (ms)': summary.get('min_duration_ms', 0),
                'Max Duration (ms)': summary.get('max_duration_ms', 0),
                'Count': summary.get('count', 0),
                'Total Duration (ms)': summary.get('mean_duration_ms', 0) * summary.get('count', 0),
            })

    if not module_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No module data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    df = pd.DataFrame(module_stats)
    df = df.sort_values('Mean Duration (ms)', ascending=True)

    # Create bar chart with error bars
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['Module'],
        x=df['Mean Duration (ms)'],
        orientation='h',
        error_x=dict(
            type='data',
            array=df['Std Duration (ms)'],
            visible=True
        ),
        marker=dict(
            color=df['Mean Duration (ms)'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Mean Duration (ms)")
        ),
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Mean: %{x:.2f} ms<br>' +
            'Std: %{customdata[0]:.2f} ms<br>' +
            'Count: %{customdata[1]}<br>' +
            'Total: %{customdata[2]:.2f} ms<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['Std Duration (ms)'],
            df['Count'],
            df['Total Duration (ms)']
        )),
    ))

    fig.update_layout(
        title='Module Average Execution Time',
        xaxis_title='Mean Duration (ms)',
        yaxis_title='Module',
        height=max(400, len(df) * 40),
        showlegend=False,
    )

    return fig


def create_module_tflops_chart(module_result: Dict[str, Any],
                               selected_modules: List[str] = None) -> go.Figure:
    """
    Create bar chart for module TFLOPS

    Args:
        module_result: Result from analyze_modules()
        selected_modules: List of module names to display (None = all)

    Returns:
        Plotly figure with TFLOPS bar chart
    """
    module_stats = []

    for module_name, data in module_result.items():
        if selected_modules and module_name not in selected_modules:
            continue

        summary = data.get('summary', {})
        mean_tflops = summary.get('mean_tflops')
        if mean_tflops is not None and summary.get('count', 0) > 0:
            module_stats.append({
                'Module': module_name,
                'Mean TFLOPS': mean_tflops,
                'Min TFLOPS': summary.get('min_tflops', 0) or 0,
                'Max TFLOPS': summary.get('max_tflops', 0) or 0,
                'FLOPs': summary.get('flops', 0),
                'Mean Duration (ms)': summary.get('mean_duration_ms', 0),
            })

    if not module_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No TFLOPS data available (load model config to enable)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    df = pd.DataFrame(module_stats)
    df = df.sort_values('Mean TFLOPS', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['Module'],
        x=df['Mean TFLOPS'],
        orientation='h',
        marker=dict(
            color=df['Mean TFLOPS'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="TFLOPS")
        ),
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Mean TFLOPS: %{x:.1f}<br>' +
            'Min TFLOPS: %{customdata[0]:.1f}<br>' +
            'Max TFLOPS: %{customdata[1]:.1f}<br>' +
            'FLOPs: %{customdata[2]:.2e}<br>' +
            'Mean Duration: %{customdata[3]:.2f} ms<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['Min TFLOPS'],
            df['Max TFLOPS'],
            df['FLOPs'],
            df['Mean Duration (ms)'],
        )),
    ))

    fig.update_layout(
        title='Module TFLOPS (Forward Pass)',
        xaxis_title='TFLOPS',
        yaxis_title='Module',
        height=max(400, len(df) * 40),
        showlegend=False,
    )

    return fig


def create_module_bandwidth_chart(module_result: Dict[str, Any],
                                  selected_modules: List[str] = None) -> go.Figure:
    """
    Create bar chart for module bandwidth (GB/s)

    Args:
        module_result: Result from analyze_modules()
        selected_modules: List of module names to display (None = all)

    Returns:
        Plotly figure with bandwidth bar chart, or None if no bandwidth data
    """
    module_stats = []

    for module_name, data in module_result.items():
        if selected_modules and module_name not in selected_modules:
            continue

        summary = data.get('summary', {})
        mean_bw = summary.get('mean_bandwidth_gbs')
        if mean_bw is not None and summary.get('count', 0) > 0:
            module_stats.append({
                'Module': module_name,
                'Mean BW (GB/s)': mean_bw,
                'Min BW (GB/s)': summary.get('min_bandwidth_gbs', 0) or 0,
                'Max BW (GB/s)': summary.get('max_bandwidth_gbs', 0) or 0,
                'Comm Volume (MB)': (summary.get('comm_volume_bytes', 0) or 0) / 1e6,
                'Mean Duration (ms)': summary.get('mean_duration_ms', 0),
            })

    if not module_stats:
        return None

    df = pd.DataFrame(module_stats)
    df = df.sort_values('Mean BW (GB/s)', ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['Module'],
        x=df['Mean BW (GB/s)'],
        orientation='h',
        marker=dict(
            color='#2980b9',
            line=dict(color='#1a5276', width=1)
        ),
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'Mean BW: %{x:.1f} GB/s<br>' +
            'Min BW: %{customdata[0]:.1f} GB/s<br>' +
            'Max BW: %{customdata[1]:.1f} GB/s<br>' +
            'Comm Volume: %{customdata[2]:.1f} MB<br>' +
            'Mean Duration: %{customdata[3]:.2f} ms<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            df['Min BW (GB/s)'],
            df['Max BW (GB/s)'],
            df['Comm Volume (MB)'],
            df['Mean Duration (ms)'],
        )),
    ))

    fig.update_layout(
        title='Module Bandwidth (Dispatch / Combine)',
        xaxis_title='Bandwidth (GB/s)',
        yaxis_title='Module',
        height=max(300, len(df) * 60),
        showlegend=False,
    )

    return fig


def create_module_timeline(module_result: Dict[str, Any],
                          selected_module: str) -> go.Figure:
    """
    Create timeline for a specific module showing all events

    Args:
        module_result: Result from analyze_modules()
        selected_module: Module name to visualize

    Returns:
        Plotly figure with scatter plot timeline
    """
    events = module_result.get(selected_module, {}).get('events', [])

    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No events for {selected_module}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    df_events = pd.DataFrame(events)
    df_events['event_id'] = range(len(df_events))

    # Create scatter plot
    fig = go.Figure()

    # Add scatter points for start times
    fig.add_trace(go.Scatter(
        x=df_events['start_ms'],
        y=df_events['duration_ms'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_events['duration_ms'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Duration (ms)")
        ),
        text=df_events['event_id'],
        hovertemplate=(
            'Event ID: %{text}<br>' +
            'Start: %{x:.2f} ms<br>' +
            'Duration: %{y:.2f} ms<br>' +
            '<extra></extra>'
        ),
        name=selected_module
    ))

    fig.update_layout(
        title=f'{selected_module} - Event Timeline',
        xaxis_title='Start Time (ms)',
        yaxis_title='Duration (ms)',
        height=500,
        hovermode='closest',
    )

    return fig


def create_module_distribution(module_result: Dict[str, Any]) -> go.Figure:
    """
    Create pie chart showing total time distribution across modules

    Args:
        module_result: Result from analyze_modules()

    Returns:
        Plotly figure with pie chart
    """
    # Calculate total time per module
    module_totals = []

    for module_name, data in module_result.items():
        summary = data.get('summary', {})
        count = summary.get('count', 0)
        mean_duration = summary.get('mean_duration_ms', 0)

        if count > 0:
            total_time = mean_duration * count
            module_totals.append({
                'Module': module_name,
                'Total Time (ms)': total_time,
                'Count': count
            })

    if not module_totals:
        fig = go.Figure()
        fig.add_annotation(
            text="No module data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    df = pd.DataFrame(module_totals)
    df = df.sort_values('Total Time (ms)', ascending=False)

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=df['Module'],
        values=df['Total Time (ms)'],
        hovertemplate=(
            '<b>%{label}</b><br>' +
            'Total Time: %{value:.2f} ms<br>' +
            'Percentage: %{percent}<br>' +
            '<extra></extra>'
        ),
    )])

    fig.update_layout(
        title='Module Time Distribution',
        height=500,
    )

    return fig


# ============================================================================
# Main App
# ============================================================================

def main():
    st.title("📊 LLM Training Profile Visualizer")
    st.markdown("Interactive visualization for torch profiler timeline analysis")

    # ========================================================================
    # Sidebar - Configuration
    # ========================================================================

    st.sidebar.header("Configuration")

    # Profile directory — contains trace files + model_config.yaml
    profile_dir = st.sidebar.text_input(
        "Profile Directory",
        value="",
        help="Directory containing trace files (torch_profiler_rank*.json.gz) and model_config.yaml"
    )

    bin_path = st.sidebar.text_input(
        "Trace Processor Binary Path",
        #value="/Users/lisiyuan/Codes/ai_infra/profiles/mac-arm64/trace_processor_shell",
        value="",
        help="Path to trace_processor_shell binary"
    )

    # Validate paths
    bin_exists = os.path.exists(bin_path) if bin_path else False
    dir_exists = os.path.isdir(profile_dir) if profile_dir else False

    if profile_dir and not dir_exists:
        st.sidebar.error(f"Directory not found: {profile_dir}")
    if not bin_exists:
        st.sidebar.error(f"Binary not found: {bin_path}")

    # Auto-discover files in directory
    trace_files = []
    model_config_path = None
    if dir_exists:
        from process_profile import discover_trace_files
        trace_files = discover_trace_files(profile_dir)
        candidate_config = os.path.join(profile_dir, 'model_config.yaml')
        if os.path.exists(candidate_config):
            model_config_path = candidate_config

        if trace_files:
            st.sidebar.success(f"Found {len(trace_files)} trace file(s)")
            for i, f in enumerate(trace_files):
                st.sidebar.caption(f"  Rank {i}: {os.path.basename(f)}")
        else:
            st.sidebar.warning("No trace files found (torch_profiler_rank*.json.gz)")

        if model_config_path:
            st.sidebar.success(f"Found model_config.yaml")
        else:
            st.sidebar.info("No model_config.yaml found (TFLOPS will not be calculated)")

    # Analyze button — always use MultiPPAnalyzer (PP=1 is just a special case)
    can_analyze = dir_exists and bin_exists and len(trace_files) > 0

    if st.sidebar.button("Load and Analyze", disabled=not can_analyze):
        with st.spinner("Loading trace(s) and analyzing..."):
            try:
                analyzer = MultiPPAnalyzer(
                    trace_dir=profile_dir,
                    bin_path=bin_path,
                    model_config_path=model_config_path,
                )
                st.sidebar.info(f"PP={analyzer.pp_degree}, "
                               f"VPP chunks={analyzer.num_model_chunks}")

                result = analyzer.analyze_all()
                st.session_state['result'] = result

                st.sidebar.success(f"Analysis complete! PP={result['pp_degree']}")

            except Exception as e:
                st.sidebar.error(f"Analysis failed: {str(e)}")
                st.exception(e)

    # ========================================================================
    # Main Content
    # ========================================================================

    # Check if analysis results are available
    if 'result' not in st.session_state:
        st.info("Please configure the profile directory in the sidebar and click 'Load and Analyze'")

        st.markdown("""
        ### Getting Started

        This tool visualizes LLM training profiles from torch profiler timeline traces.

        **Features:**
        - **VPP Schedule Timeline**: Visualize forward/backward passes with bubble analysis (PP=1 or PP>1)
        - **Module Performance**: TFLOPS and bandwidth analysis for model components
        - **Event-level Details**: Drill down into individual module events

        **Instructions:**
        1. Enter the profile directory path (containing trace files + model_config.yaml)
        2. Enter the `trace_processor_shell` binary path
        3. Click "Load and Analyze"
        """)
        return

    # Get unified result
    result = st.session_state['result']
    module_result = result.get('module_result', {})

    # ========================================================================
    # Tab Layout
    # ========================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "VPP Timeline",
        "Module Performance",
        "Module Details",
        "Optimizer",
        "Summary Statistics",
    ])

    # ========================================================================
    # Tab 1: VPP Timeline (unified for PP=1 and PP>1)
    # ========================================================================

    with tab1:
        st.header("VPP Schedule Timeline")

        summary = result.get('summary', {})

        pp_degree = result.get('pp_degree', 1)
        per_rank_stats = summary.get('per_rank_stats', [])

        # Compute averages across ranks for top-level metrics
        fwd_means = [s['fwd_mean_ms'] for s in per_rank_stats if s['fwd_mean_ms'] > 0]
        bwd_means = [s['bwd_mean_ms'] for s in per_rank_stats if s['bwd_mean_ms'] > 0]
        avg_fwd = float(np.mean(fwd_means)) if fwd_means else 0.0
        avg_bwd = float(np.mean(bwd_means)) if bwd_means else 0.0

        # Top level metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("PP Degree", pp_degree)
        with col2:
            st.metric("Microbatches", result.get('num_microbatches', 0))
        with col3:
            st.metric("Total Time", f"{summary.get('total_time_ms', 0):.1f} ms")
        with col4:
            st.metric("Avg Forward", f"{avg_fwd:.1f} ms")
        with col5:
            st.metric("Bubble Ratio", f"{summary.get('bubble_ratio', 0) * 100:.1f}%")

        # Display timeline chart (unified multi-PP gantt, works for PP=1 too)
        st.subheader("Timeline Visualization")
        fig_timeline = create_multi_pp_timeline_gantt(result)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Per-rank statistics
        with st.expander("Per-Rank Statistics", expanded=True):
            if per_rank_stats:
                rank_df = pd.DataFrame(per_rank_stats)
                display_cols = {
                    'rank': 'Rank',
                    'fwd_mean_ms': 'Fwd Mean (ms)',
                    'bwd_mean_ms': 'Bwd Mean (ms)',
                    'compute_ms': 'Compute (ms)',
                    'bubble_ms': 'Bubble (ms)',
                    'bubble_ratio': 'Bubble Ratio',
                }
                rank_df = rank_df.rename(columns=display_cols)
                rank_df['Bubble Ratio'] = rank_df['Bubble Ratio'].apply(lambda x: f"{x * 100:.1f}%")
                st.dataframe(rank_df, use_container_width=True, hide_index=True)

        # Bubble analysis
        if pp_degree > 1:
            with st.expander("Bubble Analysis"):
                if per_rank_stats:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        fig_bubble = go.Figure()
                        fig_bubble.add_trace(go.Bar(
                            x=[f"Rank {s['rank']}" for s in per_rank_stats],
                            y=[s['bubble_ratio'] * 100 for s in per_rank_stats],
                            marker=dict(color='#95a5a6'),
                            text=[f"{s['bubble_ratio'] * 100:.1f}%" for s in per_rank_stats],
                            textposition='outside',
                        ))
                        fig_bubble.update_layout(
                            title='Bubble Ratio per Rank',
                            xaxis_title='PP Rank',
                            yaxis_title='Bubble Ratio (%)',
                            height=350,
                        )
                        st.plotly_chart(fig_bubble, use_container_width=True)

                    with col_right:
                        fig_compute = go.Figure()
                        fig_compute.add_trace(go.Bar(
                            x=[f"Rank {s['rank']}" for s in per_rank_stats],
                            y=[s['compute_ms'] for s in per_rank_stats],
                            name='Compute',
                            marker=dict(color='#3498db'),
                        ))
                        fig_compute.add_trace(go.Bar(
                            x=[f"Rank {s['rank']}" for s in per_rank_stats],
                            y=[s['bubble_ms'] for s in per_rank_stats],
                            name='Bubble',
                            marker=dict(color='#e0e0e0'),
                        ))
                        fig_compute.update_layout(
                            title='Compute vs Bubble Time',
                            xaxis_title='PP Rank',
                            yaxis_title='Time (ms)',
                            barmode='stack',
                            height=350,
                        )
                        st.plotly_chart(fig_compute, use_container_width=True)

        # Schedule table
        with st.expander("Schedule Table"):
            schedule_table = result.get('schedule_table', [])
            if schedule_table:
                st.markdown(f"**Total virtual microbatches**: {len(schedule_table)}")
                sched_df = pd.DataFrame(schedule_table, columns=['microbatch_id', 'model_chunk_id'])
                sched_df.index.name = 'virtual_mb_id'
                st.dataframe(sched_df, use_container_width=True, height=300)

    # ========================================================================
    # Tab 2: Module Performance
    # ========================================================================

    with tab2:
        st.header("Module Performance Overview")

        # Get list of modules with data
        available_modules = [
            name for name, data in module_result.items()
            if data.get('summary', {}).get('count', 0) > 0
        ]

        if not available_modules:
            st.warning("No module data available")
        else:
            # Module filter
            selected_modules = st.multiselect(
                "Select modules to display",
                options=available_modules,
                default=available_modules,
                help="Select which modules to include in the visualization"
            )

            # Duration comparison chart
            st.subheader("Module Duration Comparison")
            fig_duration = create_module_duration_chart(module_result, selected_modules)
            st.plotly_chart(fig_duration, width='stretch')

            # TFLOPS chart
            st.subheader("Module TFLOPS")
            fig_tflops = create_module_tflops_chart(module_result, selected_modules)
            st.plotly_chart(fig_tflops, width='stretch')

            # Bandwidth chart (only if data exists)
            fig_bw = create_module_bandwidth_chart(module_result, selected_modules)
            if fig_bw is not None:
                st.subheader("Module Bandwidth")
                st.plotly_chart(fig_bw, width='stretch')

            # Time distribution pie chart
            st.subheader("Time Distribution")
            col1, col2 = st.columns([1, 1])

            with col1:
                fig_pie = create_module_distribution(module_result)
                st.plotly_chart(fig_pie, width='stretch')

            with col2:
                # Display table with detailed stats
                st.markdown("**Detailed Statistics**")
                stats_data = []
                for module_name in selected_modules:
                    summary = module_result[module_name].get('summary', {})
                    row = {
                        'Module': module_name,
                        'Count': summary.get('count', 0),
                        'Mean (ms)': f"{summary.get('mean_duration_ms', 0):.2f}",
                        'Std (ms)': f"{summary.get('std_duration_ms', 0):.2f}",
                        'Total (ms)': f"{summary.get('mean_duration_ms', 0) * summary.get('count', 0):.2f}",
                    }
                    mean_tflops = summary.get('mean_tflops')
                    if mean_tflops is not None:
                        row['TFLOPS'] = f"{mean_tflops:.1f}"
                        row['FLOPs'] = f"{summary.get('flops', 0):.2e}"
                    mean_bw = summary.get('mean_bandwidth_gbs')
                    if mean_bw is not None:
                        row['BW (GB/s)'] = f"{mean_bw:.1f}"
                    stats_data.append(row)

                df_stats = pd.DataFrame(stats_data)

                # Ensure proper data types for Arrow compatibility
                if not df_stats.empty:
                    df_stats['Count'] = df_stats['Count'].astype(int)

                st.dataframe(df_stats, width='stretch', hide_index=True)

    # ========================================================================
    # Tab 3: Module Details
    # ========================================================================

    with tab3:
        st.header("Module Event Details")

        # Get list of modules with data
        available_modules = [
            name for name, data in module_result.items()
            if data.get('summary', {}).get('count', 0) > 0
        ]

        if not available_modules:
            st.warning("No module data available")
        else:
            # Module selection
            selected_module = st.selectbox(
                "Select module to analyze",
                options=available_modules,
                help="Choose a module to see detailed event timeline"
            )

            # Display module timeline
            st.subheader(f"Event Timeline - {selected_module}")
            fig_module_timeline = create_module_timeline(module_result, selected_module)
            st.plotly_chart(fig_module_timeline, width='stretch')

            # Display event table
            st.subheader("Event Details")
            events = module_result[selected_module].get('events', [])
            if events:
                df_events = pd.DataFrame(events)
                df_events['event_id'] = range(len(df_events))
                df_events = df_events[['event_id', 'start_ms', 'end_ms', 'duration_ms', 'launch_delay_ms']]

                # Ensure proper data types before formatting
                df_events['event_id'] = df_events['event_id'].astype(int)

                # Format columns to strings for display
                for col in ['start_ms', 'end_ms', 'duration_ms', 'launch_delay_ms']:
                    df_events[col] = df_events[col].apply(lambda x: f"{x:.2f}")

                st.dataframe(df_events, width='stretch', hide_index=True)

                # Download button
                csv = df_events.to_csv(index=False)
                st.download_button(
                    label="Download Event Data (CSV)",
                    data=csv,
                    file_name=f"{selected_module}_events.csv",
                    mime="text/csv",
                )
            else:
                st.info("No events found for this module")

    # ========================================================================
    # Tab 4: Optimizer
    # ========================================================================

    with tab4:
        st.header("Optimizer Phase Analysis")

        optimizer_data = result.get('optimizer', {})
        opt_summary = optimizer_data.get('summary', {})
        opt_per_phase = opt_summary.get('per_phase', [])
        opt_ranks = optimizer_data.get('ranks', {})
        pp_degree = result.get('pp_degree', 1)

        if opt_per_phase:
            # Top metrics
            phase_means = {row['phase']: row.get('mean_ms', 0.0) for row in opt_per_phase}
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Grad Sync", f"{phase_means.get('Grad Reduce-Scatter', 0):.1f} ms")
            with col2:
                st.metric("AllReduce", f"{phase_means.get('Allreduce', 0):.1f} ms")
            with col3:
                st.metric("Optimizer Step", f"{phase_means.get('Optimizer Step', 0):.1f} ms")
            with col4:
                st.metric("Training Log", f"{phase_means.get('Training Log', 0):.1f} ms")
            with col5:
                st.metric("Total Optimizer", f"{opt_summary.get('total_optimizer_ms', 0):.1f} ms")

            # Per-Phase Per-Rank table
            st.subheader("Per-Phase Breakdown")
            phase_df = pd.DataFrame(opt_per_phase)
            # Rename rank columns for display
            rename_map = {'phase': 'Phase', 'mean_ms': 'Mean (ms)'}
            for rank_idx in range(pp_degree):
                col_key = f'rank_{rank_idx}_ms'
                if col_key in phase_df.columns:
                    rename_map[col_key] = f'Rank {rank_idx} (ms)'
            phase_df = phase_df.rename(columns=rename_map)
            st.dataframe(phase_df, use_container_width=True, hide_index=True)

            # Grad Reduce-Scatter detail
            with st.expander("Grad Reduce-Scatter Detail"):
                grad_rs_data = []
                for rank_idx in sorted(opt_ranks.keys()):
                    events = opt_ranks[rank_idx].get('grad_reduce_scatter', [])
                    for evt in events:
                        grad_rs_data.append({
                            'Rank': rank_idx,
                            'Label': evt.get('label', ''),
                            'GPU Start (ms)': f"{evt.get('gpu_start_ms', 0):.2f}" if evt.get('gpu_start_ms') is not None else 'N/A',
                            'GPU End (ms)': f"{evt.get('gpu_end_ms', 0):.2f}" if evt.get('gpu_end_ms') is not None else 'N/A',
                            'GPU Duration (ms)': f"{evt.get('gpu_duration_ms', 0):.2f}",
                            'CPU Duration (ms)': f"{evt.get('cpu_duration_ms', 0):.2f}",
                        })
                if grad_rs_data:
                    st.dataframe(pd.DataFrame(grad_rs_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No Grad Reduce-Scatter events found")

            # Optimizer phase timeline (zoomed in)
            st.subheader("Optimizer Phase Timeline")
            ranks_result = result.get('ranks', {})
            opt_timeline_ops = {}
            for rank_idx in range(pp_degree):
                rd = ranks_result.get(rank_idx, {})
                opt_events = rd.get('optimizer_events', [])
                opt_timeline_ops[rank_idx] = sorted(
                    [e for e in opt_events if e.get('start_ms') is not None],
                    key=lambda x: x['start_ms']
                )

            # Build optimizer phase gantt chart
            opt_color_map = {
                'grad_reduce_scatter': '#9b59b6',
                'allreduce': '#1abc9c',
                'loss_postprocessing': '#f1c40f',
                'optimizer_step': '#e67e22',
                'training_log': '#7f8c8d',
            }
            opt_phase_display = {
                'grad_reduce_scatter': 'GradRS',
                'allreduce': 'AllReduce',
                'loss_postprocessing': 'LossPost',
                'optimizer_step': 'OptStep',
                'training_log': 'TrainLog',
            }

            fig_opt = go.Figure()
            for phase_key in ['grad_reduce_scatter', 'allreduce', 'loss_postprocessing',
                              'optimizer_step', 'training_log']:
                all_ops = []
                for rank_idx in range(pp_degree):
                    for evt in opt_timeline_ops.get(rank_idx, []):
                        if evt.get('phase') == phase_key:
                            all_ops.append({
                                'rank': f"PP Rank {rank_idx}",
                                'start_ms': evt['start_ms'],
                                'end_ms': evt['end_ms'],
                                'duration_ms': evt['duration_ms'],
                                'label': evt.get('label', ''),
                            })

                if not all_ops:
                    continue

                fig_opt.add_trace(go.Bar(
                    x=[op['duration_ms'] for op in all_ops],
                    y=[op['rank'] for op in all_ops],
                    base=[op['start_ms'] for op in all_ops],
                    orientation='h',
                    name=opt_phase_display[phase_key],
                    marker=dict(
                        color=opt_color_map[phase_key],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Phase: ' + opt_phase_display[phase_key] + '<br>'
                        'Rank: %{y}<br>'
                        'Start: %{base:.2f} ms<br>'
                        'End: %{customdata[1]:.2f} ms<br>'
                        'Duration: %{x:.2f} ms<br>'
                        '<extra></extra>'
                    ),
                    customdata=[[op['label'], op['end_ms']] for op in all_ops],
                    text=[op['label'] for op in all_ops],
                    textposition='inside',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                ))

            rank_labels = [f"PP Rank {i}" for i in range(pp_degree)]
            fig_opt.update_layout(
                title={'text': 'Optimizer Phase Timeline', 'font': {'size': 16, 'color': '#2c3e50'}},
                xaxis_title='Relative Time (ms)',
                yaxis_title='Pipeline Rank',
                barmode='overlay',
                height=max(350, pp_degree * 100 + 100),
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.2)', borderwidth=1
                ),
                yaxis=dict(
                    categoryorder='array', categoryarray=rank_labels[::-1],
                    gridcolor='rgba(200, 200, 200, 0.3)'
                ),
                xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
                plot_bgcolor='rgba(250, 250, 250, 0.5)',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_opt, use_container_width=True)

        else:
            st.info("No optimizer phase data available. Make sure trace contains optimizer-related events.")

    # ========================================================================
    # Tab 5: Summary Statistics
    # ========================================================================

    with tab5:
        st.header("Summary Statistics")

        # VPP Summary
        st.subheader("VPP Schedule Summary")
        vpp_summary = result.get('summary', {})
        per_rank_stats = vpp_summary.get('per_rank_stats', [])

        # Compute aggregated forward/backward stats across all ranks
        all_fwd_means = [s.get('avg_forward_ms', 0) for s in per_rank_stats if s.get('avg_forward_ms')]
        all_bwd_means = [s.get('avg_backward_ms', 0) for s in per_rank_stats if s.get('avg_backward_ms')]
        total_fwd = sum(s.get('num_forward', 0) for s in per_rank_stats)
        total_bwd = sum(s.get('num_backward', 0) for s in per_rank_stats)

        vpp_summary_df = pd.DataFrame([
            {'Metric': 'Pipeline Parallelism Degree', 'Value': str(result.get('pp_degree', 1))},
            {'Metric': 'Model Chunks (VPP)', 'Value': str(result.get('num_model_chunks', 1))},
            {'Metric': 'Microbatches', 'Value': str(result.get('num_microbatches', 'N/A'))},
            {'Metric': 'Total Forward Steps (all ranks)', 'Value': str(total_fwd)},
            {'Metric': 'Total Backward Steps (all ranks)', 'Value': str(total_bwd)},
            {'Metric': 'Avg Forward GPU Time (ms)', 'Value': f"{sum(all_fwd_means) / len(all_fwd_means):.2f}" if all_fwd_means else "N/A"},
            {'Metric': 'Avg Backward GPU Time (ms)', 'Value': f"{sum(all_bwd_means) / len(all_bwd_means):.2f}" if all_bwd_means else "N/A"},
            {'Metric': 'Total Time (ms)', 'Value': f"{vpp_summary.get('total_time_ms', 0):.1f}"},
            {'Metric': 'Bubble Ratio', 'Value': f"{vpp_summary.get('bubble_ratio', 0) * 100:.1f}%"},
        ])

        # Ensure all columns are string type for Arrow compatibility
        vpp_summary_df['Value'] = vpp_summary_df['Value'].astype(str)

        st.dataframe(vpp_summary_df, width='stretch', hide_index=True)

        # Module Summary
        st.subheader("Module Performance Summary")

        module_summary_data = []
        for module_name, data in module_result.items():
            summary = data.get('summary', {})
            if summary.get('count', 0) > 0:
                row = {
                    'Module': module_name,
                    'Count': summary.get('count', 0),
                    'Mean Duration (ms)': f"{summary.get('mean_duration_ms', 0):.2f}",
                    'Std Duration (ms)': f"{summary.get('std_duration_ms', 0):.2f}",
                    'Min Duration (ms)': f"{summary.get('min_duration_ms', 0):.2f}",
                    'Max Duration (ms)': f"{summary.get('max_duration_ms', 0):.2f}",
                    'Mean Launch Delay (ms)': f"{summary.get('mean_delay_ms', 0):.2f}",
                    'Total Time (ms)': f"{summary.get('mean_duration_ms', 0) * summary.get('count', 0):.2f}",
                }
                mean_tflops = summary.get('mean_tflops')
                if mean_tflops is not None:
                    row['FLOPs'] = f"{summary.get('flops', 0):.2e}"
                    row['Mean TFLOPS'] = f"{mean_tflops:.1f}"
                    min_tflops = summary.get('min_tflops')
                    max_tflops = summary.get('max_tflops')
                    row['Min TFLOPS'] = f"{min_tflops:.1f}" if min_tflops else "N/A"
                    row['Max TFLOPS'] = f"{max_tflops:.1f}" if max_tflops else "N/A"
                mean_bw = summary.get('mean_bandwidth_gbs')
                if mean_bw is not None:
                    row['Comm Volume (MB)'] = f"{(summary.get('comm_volume_bytes', 0) or 0) / 1e6:.1f}"
                    row['Mean BW (GB/s)'] = f"{mean_bw:.1f}"
                    min_bw = summary.get('min_bandwidth_gbs')
                    max_bw = summary.get('max_bandwidth_gbs')
                    row['Min BW (GB/s)'] = f"{min_bw:.1f}" if min_bw else "N/A"
                    row['Max BW (GB/s)'] = f"{max_bw:.1f}" if max_bw else "N/A"
                module_summary_data.append(row)

        if module_summary_data:
            module_summary_df = pd.DataFrame(module_summary_data)

            # Ensure proper data types for Arrow compatibility
            module_summary_df['Count'] = module_summary_df['Count'].astype(int)

            st.dataframe(module_summary_df, width='stretch', hide_index=True)

            # Download button
            csv = module_summary_df.to_csv(index=False)
            st.download_button(
                label="Download Module Summary (CSV)",
                data=csv,
                file_name="module_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("No module data available")

        st.markdown("---")
        st.subheader("Future Features")
        st.markdown("""
        - **Performance Bottleneck Detection**: Automatically identify performance bottlenecks
        - **Multi-trace Comparison**: Compare metrics across different traces
        - **Export Reports**: Generate comprehensive PDF/HTML reports
        """)


if __name__ == "__main__":
    main()
