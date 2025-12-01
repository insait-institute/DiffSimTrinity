import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

import waymax.visualization.color as color
from waymax.datatypes.observation import ObjectPose2D
from utils.env_and_state_manipulation import transform_direction, transform_points, transform_traj

"""
Example usage:

1) For creating animations of reactive behavior, first launch eval/main_eval_search.py with num_modes=1, num_actions_to_commit_to=1,
imagination_length=1, step_size=[0., 0.]. Then, in eval/evaluation_search.py right after jit_eval_scenario returns, 
call e.g. plot_animated(
        sim_states, 
        imag_traj=scenario_metrics['imag_traj'], 
        save='tmp.gif', 
        b_ix=0, # Which batch sample to plot 
        show_imag_traj_in_legend=False, 
        axis_margin=5).
This will create an animation.

2) For creating animations of search behavior, first launch eval/main_eval_search.py with num_modes=4, num_actions_to_commit_to=3,
imagination_length=15, step_size=[1e3., 1e-2.]. Then, in eval/evaluation_search.py right after jit_eval_scenario returns, 
call e.g. plot_animated(
        sim_states, 
        imag_traj=scenario_metrics['imag_traj'], 
        save='tmp.gif', 
        b_ix=0, # Which batch sample to plot 
        show_imag_traj_in_legend=True, 
        axis_margin=5).

3) For creating animations of AWM odometry behavior, first launch eval/main_eval_awm.py with use_mpc=1, planning_horizon=10,
num_imagined_rollouts=1, use_planner_for_eval=0. Then, in eval/evaluation_awm.py right after jit_eval_scenario returns, 
call e.g. plot_animated_awm(
        sim_states, 
        save='tmp.gif', 
        b_ix=0, # Which batch sample to plot 
        plot_planner=False, 
        plot_odometry=True, 
        plot_inverse_state=False, 
        odometry_anchor_steps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
        plot_odometry_length=5)
This will create an animation of the ego-vehicle's 5 next imagined timesteps, overlaid over its physical trajectory at times 0, 5, 10, 15, ...

4) For creating animations of AWM planner behavior, first launch eval/main_eval_awm.py with the right trained planner. 
Set use_mpc=1, planning_horizon=1, num_imagined_rollouts=1, use_planner_for_eval=1. This corresponds to reactive planner behavior.
Then, in eval/evaluation_awm.py, right after jit_eval_scenario returns, call e.g. plot_animated_awm(
        sim_states, 
        save='tmp.gif', 
        b_ix=0, # Which batch sample to plot 
        plot_planner=False, 
        plot_odometry=False, 
        plot_inverse_state=False)
This will create an animation of the ego-vehicle's reactive behavior, governed by the planner.
Note: for best results, the planner which we evaluate should have been used for action selection during training.
That is, you need to first train a network with use_planner_for_train=True (from the config), and then evaluate with use_planner_for_eval=True.
"""


def plot_animated(
    sim_states,
    imag_traj,
    save=None,
    b_ix=0,
    plot_traffic_lights=True,
    plot_realised_traj=True,
    plot_gt=True,
    axis_margin=15.0,
    fps=20,
    ego_box_color='C0',
    ego_box_alpha=1.0,
    ego_box_edgecolor='black',
    ego_box_edgewidth=1.5,
    ego_box_zorder=5,
    ego_marker_size=40,
    ego_marker_edgecolor='black',
    ego_marker_alpha=0.9,
    agent_box_color='wheat',
    agent_box_alpha=1.0,
    agent_box_edgecolor='black',
    agent_box_edgewidth=1.5,
    agent_box_zorder=4,
    agent_traj_color='gray',
    agent_traj_alpha=0.6,
    agent_traj_linewidth=1.0,
    agent_traj_zorder=2,
    agent_marker_size=15,
    agent_marker_zorder=3,
    agent_marker_edgecolor='black',
    agent_marker_alpha=0.7,
    roadgraph_cmap='Greys',
    roadgraph_size=0.5,
    roadgraph_linewidth=0.75,
    roadgraph_alpha=1.0,
    roadgraph_zorder=1,
    traffic_light_size=4,
    traffic_light_marker='o',
    traffic_light_zorder=6,
    ego_traj_color='C1',
    ego_traj_linewidth=2.0,
    ego_traj_alpha=1.0,
    ego_traj_label='Physical trajectory',
    ego_traj_zorder=4,
    gt_traj_color='#007fff',
    gt_traj_linewidth=2.0,
    gt_traj_alpha=0.6,
    gt_traj_label='Ground truth',
    gt_traj_zorder=2,
    imag_traj_color='m',
    imag_traj_linewidth=1.2,
    imag_traj_alpha=1.0,
    imag_traj_label='Imagined trajectory',
    imag_traj_linestyle='--',
    imag_traj_zorder=6,
    legend_loc='upper right',
    title_prefix='Timestep',
    show_imag_traj_in_legend=True,
):
    """Animate agent motion, imagined trajectories, and environment context."""
    plt.clf()
    fig, axs = plt.subplots()
    imag_traj = imag_traj[..., b_ix, :] # (80, K, T, 2)
    rg = sim_states.roadgraph_points.xy[b_ix] # (20K, 2)
    rg_val = sim_states.roadgraph_points.valid[b_ix] # (20K)
    log_traj = sim_states.log_trajectory.xy[b_ix] # (N, 91, 2)
    log_valid = sim_states.log_trajectory.valid[b_ix] # (N, 91)
    sim_traj = sim_states.sim_trajectory.xy[b_ix] # (N, 91, 2)
    is_sdc = sim_states.object_metadata.is_sdc[b_ix] # N = agents
    sdc_idx = jax.lax.top_k(is_sdc, 1)[1]
    ego_log_traj = log_traj[is_sdc][0] # (91, 2)
    ego_sim_traj = sim_traj[is_sdc][0] # (91, 2)
    rg_type = sim_states.roadgraph_points.types[b_ix] # (20K)
    tls_timestep = 0
    tls_valid = sim_states.log_traffic_light.valid[b_ix, :, tls_timestep] # (16,)
    tls_xy = sim_states.log_traffic_light.xy[b_ix, :, tls_timestep, :] # (16, 2)
    tls_state = sim_states.log_traffic_light.state[b_ix, :, tls_timestep] # (16,)
    log_box_corners = sim_states.log_trajectory.bbox_corners[b_ix, :, tls_timestep] # (128, 4, 2)
    # DONT CHANGE THE BLOCK ABOVE

    ax = axs
    imag_traj = np.asarray(imag_traj)
    rg_xy = np.asarray(rg)
    rg_valid_np = np.asarray(rg_val).astype(bool)
    rg_type_np = np.asarray(rg_type)
    log_traj_np = np.asarray(log_traj)
    log_valid_np = np.asarray(log_valid).astype(bool)
    sim_traj_np = np.asarray(sim_traj)
    try:
        sim_valid_np = np.asarray(sim_states.sim_trajectory.valid[b_ix]).astype(bool)
    except AttributeError:
        sim_valid_np = np.ones(sim_traj_np.shape[:2], dtype=bool)

    if sim_valid_np.ndim == 3 and sim_valid_np.shape[-1] == 1:
        sim_valid_np = sim_valid_np[..., 0]

    sim_xy = np.transpose(sim_traj_np, (1, 0, 2))  # (frames, agents, 2)
    sim_valid = np.transpose(sim_valid_np, (1, 0)) if sim_valid_np.ndim == 2 else np.ones(sim_xy.shape[:2], dtype=bool)

    sim_box_seq = None
    if hasattr(sim_states.sim_trajectory, 'bbox_corners'):
        sim_box_seq_np = np.asarray(sim_states.sim_trajectory.bbox_corners[b_ix])
        if sim_box_seq_np.ndim >= 4:
            sim_box_seq = np.transpose(sim_box_seq_np, (1, 0, 2, 3))

    is_sdc_mask = np.asarray(is_sdc).astype(bool)
    sdc_idx_arr = np.asarray(sdc_idx)
    ego_idx = int(sdc_idx_arr.reshape(-1)[0]) if sdc_idx_arr.size else int(np.argmax(is_sdc_mask))

    ego_log_traj_np = np.asarray(ego_log_traj)  # (91, 2)

    num_frames = sim_xy.shape[0]
    num_agents = sim_xy.shape[1]
    imag_length = imag_traj.shape[2] if imag_traj.ndim >= 3 else 0

    tls_xy_all = np.asarray(sim_states.log_traffic_light.xy[b_ix])
    tls_valid_all = np.asarray(sim_states.log_traffic_light.valid[b_ix]).astype(bool)
    tls_state_all = np.asarray(sim_states.log_traffic_light.state[b_ix])
    tls_num_frames = tls_xy_all.shape[1] if tls_xy_all.ndim >= 3 else 0

    # Pre-compute last imagination index available for each frame.
    if imag_length:
        imag_norm = np.linalg.norm(imag_traj, axis=-1)
        imag_valid_mask = imag_norm > 1e-3
        imag_valid_per_frame = np.any(imag_valid_mask, axis=(1, 2))
    else:
        imag_valid_mask = None
        imag_valid_per_frame = None

    history_steps = max(0, num_frames - imag_traj.shape[0]) if imag_length else 0
    last_imag_idx = np.full(num_frames, -1, dtype=int)
    last_seen = -1
    for frame in range(num_frames):
        imag_frame = frame - history_steps
        if imag_length and 0 <= imag_frame < imag_valid_per_frame.shape[0] and imag_valid_per_frame[imag_frame]:
            last_seen = imag_frame
        last_imag_idx[frame] = last_seen

    # Determine axis limits once so they remain fixed across frames.
    ego_mask = sim_valid[:, ego_idx]
    if ego_mask.any():
        ego_pts = sim_xy[:, ego_idx][ego_mask]
        x_min = ego_pts[:, 0].min() - axis_margin
        x_max = ego_pts[:, 0].max() + axis_margin
        y_min = ego_pts[:, 1].min() - axis_margin
        y_max = ego_pts[:, 1].max() + axis_margin
    else:
        points = []
        if rg_valid_np.any():
            points.append(rg_xy[rg_valid_np])

        flat_sim = sim_xy.reshape(-1, 2)
        flat_sim_mask = sim_valid.reshape(-1)
        if flat_sim_mask.any():
            points.append(flat_sim[flat_sim_mask])

        flat_log = log_traj_np.reshape(-1, 2)
        flat_log_mask = log_valid_np.reshape(-1)
        if flat_log_mask.any():
            points.append(flat_log[flat_log_mask])

        if imag_length:
            flat_imag = imag_traj.reshape(-1, 2)
            mask_imag = np.linalg.norm(flat_imag, axis=-1) > 1e-3
            if mask_imag.any():
                points.append(flat_imag[mask_imag])

        if not points:
            points.append(np.zeros((1, 2)))

        all_pts = np.concatenate(points, axis=0)
        x_min, y_min = all_pts.min(axis=0) - axis_margin
        x_max, y_max = all_pts.max(axis=0) + axis_margin

    span_x = x_max - x_min
    span_y = y_max - y_min
    min_span = 1e-3
    base_span = max(span_x, span_y, min_span)
    half_span = base_span / 2.0
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    x_min = center_x - half_span
    x_max = center_x + half_span
    y_min = center_y - half_span
    y_max = center_y + half_span

    x_center = center_x
    y_center = center_y

    def fmt_x(x, _):
        return f"{x - x_center:.1f}"

    def fmt_y(y, _):
        return f"{y - y_center:.1f}"

    legend_handles = []
    legend_labels = []
    if show_imag_traj_in_legend and imag_length:
        phantom_line = plt.Line2D(
            [],
            [],
            color=imag_traj_color,
            linestyle=imag_traj_linestyle,
            linewidth=imag_traj_linewidth,
            alpha=imag_traj_alpha,
            label=imag_traj_label,
        )
        legend_handles.append(phantom_line)
        legend_labels.append(imag_traj_label)

    def draw_frame(frame):
        ax.clear()

        if rg_valid_np.any():
            ax.scatter(
                rg_xy[rg_valid_np, 0],
                rg_xy[rg_valid_np, 1],
                c=rg_type_np[rg_valid_np],
                cmap=roadgraph_cmap,
                linewidth=roadgraph_linewidth,
                s=roadgraph_size,
                alpha=roadgraph_alpha,
                zorder=roadgraph_zorder,
            )

        if plot_gt and ego_log_traj_np.size:
            ax.plot(
                ego_log_traj_np[:, 0],
                ego_log_traj_np[:, 1],
                '--',
                c=gt_traj_color,
                linewidth=gt_traj_linewidth,
                alpha=gt_traj_alpha,
                label=gt_traj_label,
                zorder=gt_traj_zorder,
            )

        if plot_realised_traj:
            ego_hist_valid = sim_valid[:frame + 1, ego_idx]
            if ego_hist_valid.any():
                ego_hist = sim_xy[:frame + 1, ego_idx][ego_hist_valid]
                ax.plot(
                    ego_hist[:, 0],
                    ego_hist[:, 1],
                    c=ego_traj_color,
                    linewidth=ego_traj_linewidth,
                    alpha=ego_traj_alpha,
                    label=ego_traj_label,
                    zorder=ego_traj_zorder,
                )

        if frame < sim_xy.shape[0] and sim_valid[frame, ego_idx]:
            ego_pos = sim_xy[frame, ego_idx]
            if sim_box_seq is not None:
                ego_box_sim = sim_box_seq[min(frame, sim_box_seq.shape[0] - 1), ego_idx]
                if np.any(np.abs(ego_box_sim) > 0):
                    ax.add_patch(
                        Polygon(
                            ego_box_sim,
                            closed=True,
                            fill=True,
                            facecolor=ego_box_color,
                            edgecolor=ego_box_edgecolor,
                            linewidth=ego_box_edgewidth,
                            alpha=ego_box_alpha,
                            zorder=ego_box_zorder,
                        )
                    )
                else:
                    ax.scatter(
                        ego_pos[0],
                        ego_pos[1],
                        s=ego_marker_size,
                        c=ego_box_color,
                        edgecolors=ego_marker_edgecolor,
                        alpha=ego_marker_alpha,
                        zorder=ego_box_zorder,
                    )
            else:
                ax.scatter(
                    ego_pos[0],
                    ego_pos[1],
                    s=ego_marker_size,
                    c=ego_box_color,
                    edgecolors=ego_marker_edgecolor,
                    alpha=ego_marker_alpha,
                    zorder=ego_box_zorder,
                )

        for agent_idx in range(num_agents):
            if agent_idx == ego_idx:
                continue
            agent_hist_valid = sim_valid[:frame + 1, agent_idx]
            if not agent_hist_valid.any():
                continue
            agent_hist = sim_xy[:frame + 1, agent_idx][agent_hist_valid]
            ax.plot(
                agent_hist[:, 0],
                agent_hist[:, 1],
                c=agent_traj_color,
                alpha=agent_traj_alpha,
                linewidth=agent_traj_linewidth,
                zorder=agent_traj_zorder,
            )
            if sim_valid[frame, agent_idx]:
                if sim_box_seq is not None:
                    agent_box = sim_box_seq[min(frame, sim_box_seq.shape[0] - 1), agent_idx]
                    if np.any(np.abs(agent_box) > 0):
                        ax.add_patch(
                            Polygon(
                                agent_box,
                                closed=True,
                                fill=True,
                                facecolor=agent_box_color,
                                edgecolor=agent_box_edgecolor,
                                linewidth=agent_box_edgewidth,
                                alpha=agent_box_alpha,
                                zorder=agent_box_zorder,
                            )
                        )
                    else:
                        agent_pos = sim_xy[frame, agent_idx]
                        ax.scatter(
                            agent_pos[0],
                            agent_pos[1],
                            c=agent_box_color,
                            s=agent_marker_size,
                            edgecolors=agent_marker_edgecolor,
                            alpha=agent_marker_alpha,
                            zorder=agent_marker_zorder,
                        )
                else:
                    agent_pos = sim_xy[frame, agent_idx]
                    ax.scatter(
                        agent_pos[0],
                        agent_pos[1],
                        c=agent_box_color,
                        s=agent_marker_size,
                        edgecolors=agent_marker_edgecolor,
                        alpha=agent_marker_alpha,
                        zorder=agent_marker_zorder,
                    )

        if plot_traffic_lights and tls_num_frames:
            tls_frame = min(frame, tls_num_frames - 1)
            tls_mask = tls_valid_all[:, tls_frame]
            tls_xy_frame = tls_xy_all[:, tls_frame]
            tls_state_frame = tls_state_all[:, tls_frame]
            for xy, state in zip(tls_xy_frame[tls_mask], tls_state_frame[tls_mask]):
                ax.plot(
                    xy[0],
                    xy[1],
                    marker=traffic_light_marker,
                    color=color.TRAFFIC_LIGHT_COLORS[int(state)],
                    ms=traffic_light_size,
                    zorder=traffic_light_zorder,
                )

        imag_frame = frame - history_steps
        if show_imag_traj_in_legend:
            if imag_length and imag_frame >= 0:
                source_idx = last_imag_idx[frame]
                if source_idx >= 0 and source_idx < imag_traj.shape[0]:
                    offset = imag_frame - source_idx
                    if 0 <= offset < imag_length:
                        added_label = False
                        for mode in range(imag_traj.shape[1]):
                            traj = imag_traj[source_idx, mode]
                            valid_mask = imag_valid_mask[source_idx, mode]
                            if offset < traj.shape[0]:
                                seg = traj[offset:]
                                seg_valid = valid_mask[offset:]
                                seg = seg[seg_valid]
                                if seg.size:
                                    if sim_valid[frame, ego_idx]:
                                        ego_pos = sim_xy[frame, ego_idx]
                                        if np.linalg.norm(seg[0] - ego_pos) > 1e-6:
                                            seg = np.concatenate([ego_pos[None], seg], axis=0)
                                    label = imag_traj_label if not added_label else None
                                    ax.plot(
                                        seg[:, 0],
                                        seg[:, 1],
                                        c=imag_traj_color,
                                        linestyle=imag_traj_linestyle,
                                        linewidth=imag_traj_linewidth,
                                        alpha=imag_traj_alpha,
                                        label=None,
                                        zorder=imag_traj_zorder,
                                    )
                                    added_label = True

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_box_aspect(1.0)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel('Δx [m]')
        ax.set_ylabel('Δy [m]')
        ax.set_title(f'{title_prefix} {frame}')
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            handles = list(handles) + legend_handles
            labels = list(labels) + legend_labels
        if labels:
            ax.legend(handles, labels, loc=legend_loc)
        return []

    interval_ms = 1000.0 / max(fps, 1)
    anim = animation.FuncAnimation(fig, draw_frame, frames=num_frames, interval=interval_ms, blit=False)
    output_path = save or 'tmp_animation.gif'
    writer_fps = max(fps, 1)
    anim.save(output_path, writer=animation.PillowWriter(fps=writer_fps))
    plt.close(fig)
    return output_path


def plot_preds_awm(sim_states, save=None, b_ix=0, plot_traffic_lights=True, plot_planner=True, plot_odometry=True, 
        plot_realised_traj=True, plot_inverse_state=True, plot_gt=True, min_r=10, loc='upper left'):

    plt.clf()
    simulations = sim_states['simulations']
    sim_states = sim_states['sim_states']
    rg = sim_states.roadgraph_points.xy[0] # (B, 20K, 2)
    rg_val = sim_states.roadgraph_points.valid[0] # (B, 20K)
    log_traj = sim_states.log_trajectory # (80, B, N, 91)
    is_sdc = sim_states.object_metadata.is_sdc[0, b_ix] # N = agents
    rg_type = sim_states.roadgraph_points.types[0] # (B, 20K)
    log_box_corners = sim_states.log_trajectory.bbox_corners # (80, B, N, 91, 4, 2)
    ego_log_traj = log_traj.xy[0, b_ix, is_sdc][0] # (91, 2)

    # Plot traffic lights
    if plot_traffic_lights:
        tls = sim_states.log_traffic_light
        tls_timestep = 0
        tls_valid = tls.valid[0, b_ix, :, tls_timestep]
        if tls_valid.sum() == 0:
            pass
        tls_xy = tls.xy[0, b_ix, :, tls_timestep][tls_valid]
        tls_state = tls.state[0, b_ix, :, tls_timestep][tls_valid]
        for xy, state in zip(tls_xy, tls_state):
            tl_color = color.TRAFFIC_LIGHT_COLORS[int(state)]
            plt.gca().plot(xy[0], xy[1], marker='o', color=tl_color, ms=4)
    
    log_sdc_traj = sim_states.log_trajectory.xy[0, b_ix, is_sdc][0] # (91, 2)
    sim_sdc_traj = sim_states.sim_trajectory.xy[0, b_ix, is_sdc][0] # (91, 2)
    plt.scatter(rg[b_ix, rg_val[b_ix], 0], rg[b_ix, rg_val[b_ix], 1], c=rg_type[b_ix, rg_val[b_ix]], cmap='Greys', linewidth=0.75, s=0.5)
    if plot_gt:
        #plt.plot(log_sdc_traj[:, 0], log_sdc_traj[:, 1], '--', c='#007fff', label='Ground truth', linewidth=3, zorder=2)
        plt.plot(log_sdc_traj[:, 0], log_sdc_traj[:, 1], '--', c='k', label='Ground truth', linewidth=1, zorder=2)
        pts = log_box_corners[0, b_ix, is_sdc, 0][0] # (4, 2)
        box = Polygon(pts, closed=True, fill=True, facecolor='C0', edgecolor='black', linewidth=1.5)
        plt.gca().add_patch(box)

    #plt.scatter(log_sdc_traj[:, 0][0], log_sdc_traj[:, 1][0], zorder=3, edgecolors='k', c='#007fff')

    plt.xlim([log_sdc_traj[:, 0].min() - 10, log_sdc_traj[:, 0].max() + 10])
    plt.ylim([log_sdc_traj[:, 1].min() - 10, log_sdc_traj[:, 1].max() + 10])

    # Plot simulations
    sim_sdc_traj = sim_states.current_sim_trajectory.xy[:, b_ix, is_sdc, 0][:, 0] # (80, 2)
    if plot_realised_traj:
        plt.plot(log_sdc_traj[:10, 0], log_sdc_traj[:10, 1], '--', c='C1', alpha=1, linewidth=2.5)
        plt.plot(sim_sdc_traj[:, 0], sim_sdc_traj[:, 1], '--', c='C1', alpha=1, label='Realized', linewidth=2.5)
    
    # Plot planner trajectory from current step (T, 2)
    t_fixed = 0
    offset_loc = sim_sdc_traj[t_fixed]
    # planner_next_states = jnp.stack([s_t[0][-1][0]['planner_next_state'][0, b_ix, -2:] for s_t in simulations], axis=0) # (80, 2)
    planner_next_states = jnp.stack([jnp.stack(
        [x['planner_next_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 3)
    inv_opt_states = jnp.stack([jnp.stack(
        [x['inv_opt_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 2)
    odometries = jnp.stack([jnp.stack(
        [x['odometry_next_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 3)
    

    Nsim = planner_next_states.shape[0]
    Nhorizon = planner_next_states.shape[1]
    sim_xyyaw = sim_states.current_sim_trajectory.stack_fields(('x', 'y', 'yaw'))[..., 0, :][:, b_ix] # (80, N=128, 3)
    sim_xyyaw_sdc = sim_xyyaw[:, is_sdc][:, 0] # (80, 3)
    sim_sdc_state = sim_states.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y', 'speed'))[..., 0, :][:, b_ix][:, is_sdc][:, 0] # (80, 6)
    
    # Plot planner
    dt = 0.1
    planner_label_added = False
    if plot_planner:
        for t_ix, t_fixed in enumerate(range(80)):
            for s in range(Nsim):
                # Global state
                xy = sim_sdc_state[t_fixed, :2]
                yaw = sim_sdc_state[t_fixed, 2]
                speed = sim_sdc_state[t_fixed, -1]
                velxy = sim_sdc_state[t_fixed, 3:5]

                # Current values for the state in global coords
                true_coords = [xy]
                current_xy = xy
                current_yaw = yaw
                current_speed = speed
                current_velxy = velxy

                for h in range(1): # Nhorizon
                    
                    # Planned values
                    planned_vxy = planner_next_states[s, h, t_fixed, 1:]

                    # Pose
                    st_pose = ObjectPose2D.from_center_and_yaw(xy=current_xy, yaw=current_yaw)
                    st_pose_matrix = st_pose.matrix[None]
                    st_delta_yaw = st_pose.delta_yaw
                    st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)
                    
                    # Current values in local coord
                    current_velxy_local = transform_direction(pts_dir=current_velxy[None], pose_matrix=st_pose_matrix)[0]
                    
                    # Next state velocities in local
                    next_state_vxy = current_velxy_local + planned_vxy
                    
                    # Next state velocities in global
                    next_state_vxy = transform_direction(pts_dir=next_state_vxy[None], pose_matrix=st_pose_matrix_inv)[0]
                    
                    # Next state in global
                    next_state_speed = jnp.sqrt(next_state_vxy[0]**2 + next_state_vxy[1]**2)
                    accel = (next_state_speed - current_speed) / dt
                    steering = (jnp.arctan2(next_state_vxy[1], next_state_vxy[0]) - current_yaw) / (current_speed * dt + 0.5 * accel * dt**2)
                    next_x = current_xy[0] + current_velxy[0] * dt + 0.5 * jnp.cos(current_yaw) * accel * dt**2
                    next_y = current_xy[1] + current_velxy[1] * dt + 0.5 * jnp.sin(current_yaw) * accel * dt**2
                    next_yaw = current_yaw + steering * (current_speed * dt + 0.5 * accel * dt**2)
                    next_speed = current_speed + accel * dt
                    next_vx = next_speed * jnp.cos(next_yaw)
                    next_vy = next_speed * jnp.sin(next_yaw)
                    current_xy = jnp.stack([next_x, next_y], axis=-1)
                    current_yaw = next_yaw
                    current_speed = next_speed
                    current_velxy = jnp.stack([next_vx, next_vy], axis=-1)
                    true_coords.append(current_xy)
                
                planned_traj = jnp.stack(true_coords, axis=0)
                if planner_label_added == False:
                    plt.scatter(planned_traj[:, 0], planned_traj[:, 1], s=5, c='green', label='Planned traj', zorder=30)
                    planner_label_added = True
                else:
                    plt.scatter(planned_traj[:, 0], planned_traj[:, 1], s=5, c='green', zorder=30)
                break

	# Plot inv opt state norms
    if plot_inverse_state:
        assert plot_odometry == False
        inv_opt_state_norms = jnp.linalg.norm(inv_opt_states[:, 0], ord=2, axis=-1) # (Nsim, T, 1)
        inv_norm = plt.gca().scatter(sim_sdc_traj[:, 0], sim_sdc_traj[:, 1], c=jnp.log(inv_opt_state_norms[0]), zorder=4, alpha=0.3, cmap='jet')
        cbar = plt.colorbar(inv_norm, ax=plt.gca(), pad=0.01)
        cbar.set_label('Log of norm of inv. state displacement')  # optional label

	# Plot planner
	
    	
    # Plot odometry (imagined trajectories)
    if plot_odometry:
        odometry_label_added = False
        for t_ix, t_fixed in enumerate([0, 10, 20, 30, 40, 50, 60]):
        
            is_sdc_ = sim_states.object_metadata.is_sdc[:, b_ix]
            xy = sim_states.current_sim_trajectory.xy[:, b_ix][is_sdc_][t_fixed, 0]
            yaw = sim_states.current_sim_trajectory.yaw[:, b_ix][is_sdc_][t_fixed, 0]
            st_pose = ObjectPose2D.from_center_and_yaw(xy=xy, yaw=yaw)
            st_pose_matrix = st_pose.matrix[None]
            st_delta_yaw = st_pose.delta_yaw
            st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)
            offset_loc = sim_sdc_traj[t_fixed]
            
            for s_ix in range(Nsim):
                odometries_t_ix = odometries[s_ix, :, t_fixed, :2] # (Nhorizon, 2)
                odometries_t_ix_theta = odometries[s_ix, :, t_fixed, 2]
                
                true_coords = [offset_loc]
                current_xy = offset_loc
                current_yaw = sim_states.current_sim_trajectory.yaw[t_fixed, b_ix, is_sdc_[t_fixed]] # ()
                for k in range(Nhorizon):
                    # Pose from current location
                    p = ObjectPose2D.from_center_and_yaw(xy=current_xy[None], yaw=current_yaw[0])
                    p_pose_matrix = p.matrix[None]
                    p_delta_yaw = p.delta_yaw
                    p_pose_matrix_inv = jnp.linalg.inv(p_pose_matrix)

                    # Project odometry predictions to global
                    current_xy = transform_points(pts=odometries_t_ix[k][None], pose_matrix=p_pose_matrix_inv[0]) # (1, 2)
                    #current_yaw = (current_yaw + odometries_t_ix_theta[k])
                    current_yaw = -(odometries_t_ix_theta[k] + p_delta_yaw[None]) # VERIFY THIS FORMULA! IT IS RANDOM!!!!
                    true_coords.append(current_xy[0])
                    current_xy = current_xy[0]
                
                odometries_t_ix = jnp.stack(true_coords, axis=0)
                if odometry_label_added == False:
                    plt.scatter([], [], s=15, edgecolor='k', c='white', linewidth=0.5, alpha=1, label='Imagined future', zorder=10)
                    plt.scatter(odometries_t_ix[..., 0], odometries_t_ix[..., 1], s=15, edgecolor='k', linewidth=0.5, alpha=1, zorder=10)
                    odometry_label_added = True
                else:
                    plt.scatter(odometries_t_ix[..., 0], odometries_t_ix[..., 1], s=15, edgecolor='k', linewidth=0.5, alpha=1, zorder=4)
	
            
    # For all agents
    all_log_traj = sim_states.log_trajectory.xy # (T, B, N)
    all_log_val = sim_states.log_trajectory.valid # (T, B, N)
    for k in range(128):
        if is_sdc[k]:
            continue
        log_sdc_traj = all_log_traj[0, b_ix, k]
        val = all_log_val[0, b_ix, k]
        if jnp.any(val):
            plt.plot(log_sdc_traj[val, 0], log_sdc_traj[val, 1], c='gray', alpha=0.5)
            pts = log_box_corners[0, b_ix, k, 0] # (4, 2)
            box = Polygon(pts, closed=True, fill=True, facecolor='wheat', edgecolor='black', linewidth=1.5)
            plt.gca().add_patch(box)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.gca().set_box_aspect(1.0)   # same effect
    cx, cy = ego_log_traj[45, 0], ego_log_traj[45, 1]     
    r = jnp.max(jnp.stack([jnp.abs(ego_log_traj[:, 0] - cx).max() + min_r, jnp.abs(ego_log_traj[:, 1] - cy).max() + min_r]))
    plt.gca().set_xlim(cx - r, cx + r)
    plt.gca().set_ylim(cy - r, cy + r)
    if not plot_inverse_state:
        plt.legend(loc=loc)
    # plt.gca().set_aspect('equal', adjustable='box')

    if save:
        plt.savefig(save, dpi=250, bbox_inches='tight')


def plot_animated_awm(sim_states, save=None, b_ix=0, plot_traffic_lights=True, plot_planner=True, plot_odometry=True, 
        plot_realised_traj=True, plot_inverse_state=True, plot_gt=True, min_r=10, loc='upper left', fps=20, frame_stride=1,
        odometry_anchor_steps=(0, 10, 20, 30, 40, 50, 60), title_prefix='Timestep', roadgraph_cmap='Greys', roadgraph_size=0.5,
        roadgraph_linewidth=0.75, roadgraph_alpha=1.0, traffic_light_size=4, traffic_light_marker='o', realized_color='C1',
        realized_linewidth=2.5, realized_alpha=1.0, planner_color='green', odometry_facecolor='white', odometry_edgecolor='k',
        odometry_marker_size=15, odometry_alpha=1.0, plot_odometry_length=10):
    """Animate plot_preds_awm visuals over time and export as a GIF."""

    # Local import to avoid NameError if the module-level import is modified.
    from matplotlib import animation as mpl_animation

    plt.clf()
    simulations = sim_states['simulations']
    sim_states = sim_states['sim_states']
    rg = sim_states.roadgraph_points.xy[0] # (B, 20K, 2)
    rg_val = sim_states.roadgraph_points.valid[0] # (B, 20K)
    log_traj = sim_states.log_trajectory # (80, B, N, 91)
    is_sdc = sim_states.object_metadata.is_sdc[0, b_ix] # N = agents
    rg_type = sim_states.roadgraph_points.types[0] # (B, 20K)
    log_box_corners = sim_states.log_trajectory.bbox_corners # (80, B, N, 91, 4, 2)
    ego_log_traj = log_traj.xy[0, b_ix, is_sdc][0] # (91, 2)

    fig, ax = plt.subplots()
    frame_stride = max(1, int(frame_stride))
    odometry_anchor_steps = tuple(odometry_anchor_steps) if odometry_anchor_steps else ()

    rg_np = np.asarray(rg)
    rg_val_np = np.asarray(rg_val).astype(bool)
    rg_type_np = np.asarray(rg_type)
    log_box_corners_np = np.asarray(log_box_corners)
    ego_log_traj_np = np.asarray(ego_log_traj)
    is_sdc_mask = np.asarray(is_sdc).astype(bool)
    base_box_rel_corners = []
    base_box_base_yaw = []
    base_box_world = []
    num_box_agents = log_box_corners_np.shape[2] if log_box_corners_np.ndim >= 3 else 0
    for k in range(num_box_agents):
        try:
            base_corners = log_box_corners_np[0, b_ix, k, 0]  # (4, 2)
        except Exception:
            base_corners = None
        if base_corners is None or base_corners.shape != (4, 2) or not np.any(np.abs(base_corners) > 0):
            base_box_rel_corners.append(None)
            base_box_base_yaw.append(0.0)
            base_box_world.append(None)
            continue
        centroid = base_corners.mean(axis=0)
        v10 = base_corners[1] - base_corners[0]
        base_yaw = float(np.arctan2(v10[1], v10[0]))
        rel = base_corners - centroid
        base_box_rel_corners.append(rel)
        base_box_base_yaw.append(base_yaw)
        base_box_world.append(base_corners)

    sim_sdc_traj = sim_states.current_sim_trajectory.xy[:, b_ix, is_sdc, 0][:, 0] # (80, 2)
    sim_sdc_traj_np = np.asarray(sim_sdc_traj)
    sim_len = sim_sdc_traj_np.shape[0] if sim_sdc_traj_np.ndim else 0
    if sim_len == 0:
        sim_len = 1
        sim_sdc_traj_np = np.zeros((1, 2))

    sim_xy_all_np = np.asarray(sim_states.current_sim_trajectory.xy[:, b_ix])
    if sim_xy_all_np.ndim >= 4:
        sim_xy_all_np = sim_xy_all_np[..., 0, :]
    sim_valid_all_np = np.ones(sim_xy_all_np.shape[:2], dtype=bool)
    try:
        sim_valid_all_np = np.asarray(sim_states.current_sim_trajectory.valid[:, b_ix]).astype(bool)
        if sim_valid_all_np.ndim >= 3 and sim_valid_all_np.shape[-1] == 1:
            sim_valid_all_np = sim_valid_all_np[..., 0]
    except AttributeError:
        pass

    num_agents = sim_xy_all_np.shape[1] if sim_xy_all_np.ndim >= 2 else 0
    if sim_valid_all_np.ndim >= 2:
        num_agents = min(num_agents, sim_valid_all_np.shape[1])

    try:
        sim_yaw_np = np.asarray(sim_states.current_sim_trajectory.yaw[:, b_ix])
        if sim_yaw_np.ndim == 3 and sim_yaw_np.shape[-1] == 1:
            sim_yaw_np = sim_yaw_np[..., 0]
    except AttributeError:
        sim_yaw_np = None

    log_xy_all_np = np.asarray(sim_states.log_trajectory.xy[0, b_ix]) if hasattr(sim_states.log_trajectory, 'xy') else None
    log_valid_all_np = np.asarray(sim_states.log_trajectory.valid[0, b_ix]).astype(bool) if hasattr(sim_states.log_trajectory, 'valid') else None
    log_len_total = log_xy_all_np.shape[1] if log_xy_all_np is not None and log_xy_all_np.ndim >= 2 else 0
    log_horizon = min(10, log_len_total) if log_len_total else 0

    def _estimate_heading(traj_xy, traj_valid, agent_idx):
        headings = np.zeros(traj_xy.shape[0])
        for t in range(traj_xy.shape[0]):
            yaw_val = None
            if log_box_corners_np is not None and log_box_corners_np.size:
                try:
                    corners_t = log_box_corners_np[0, b_ix, agent_idx, t]
                    if corners_t.shape == (4, 2) and np.any(np.abs(corners_t) > 0):
                        v10_t = corners_t[1] - corners_t[0]
                        yaw_val = float(np.arctan2(v10_t[1], v10_t[0]))
                except Exception:
                    yaw_val = None
            if yaw_val is None:
                if not traj_valid[t]:
                    continue
                if t + 1 < traj_xy.shape[0] and traj_valid[t + 1]:
                    delta = traj_xy[t + 1] - traj_xy[t]
                elif t > 0 and traj_valid[t - 1]:
                    delta = traj_xy[t] - traj_xy[t - 1]
                else:
                    delta = np.array([1.0, 0.0])
                yaw_val = float(np.arctan2(delta[1], delta[0]))
            headings[t] = yaw_val
        return headings

    combined_len = log_horizon + sim_len
    combined_xy = np.zeros((combined_len, num_agents, 2))
    combined_valid = np.zeros((combined_len, num_agents), dtype=bool)
    combined_yaw = np.zeros((combined_len, num_agents))

    if log_horizon and log_xy_all_np is not None and log_valid_all_np is not None:
        for agent_idx in range(num_agents):
            agent_valid = log_valid_all_np[agent_idx, :log_horizon] if log_valid_all_np.ndim >= 2 else np.ones(log_horizon, dtype=bool)
            agent_xy = log_xy_all_np[agent_idx, :log_horizon]
            combined_xy[:log_horizon, agent_idx] = agent_xy
            combined_valid[:log_horizon, agent_idx] = agent_valid
            agent_heading = _estimate_heading(agent_xy, agent_valid, agent_idx)
            combined_yaw[:log_horizon, agent_idx] = agent_heading

    for t in range(sim_len):
        combined_idx = t + log_horizon
        combined_xy[combined_idx] = sim_xy_all_np[t] if t < sim_xy_all_np.shape[0] else 0.0
        if sim_valid_all_np.shape[0] > t:
            combined_valid[combined_idx] = sim_valid_all_np[t]
        else:
            combined_valid[combined_idx] = False
        if sim_yaw_np is not None and sim_yaw_np.shape[0] > t:
            combined_yaw[combined_idx] = sim_yaw_np[t]

    total_frames = combined_len if combined_len > 0 else 1

    frame_indices = list(range(0, total_frames, frame_stride))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    tls = sim_states.log_traffic_light
    tls_valid_np = np.asarray(tls.valid[0, b_ix])
    tls_xy_np = np.asarray(tls.xy[0, b_ix])
    tls_state_np = np.asarray(tls.state[0, b_ix])
    tls_frames = tls_valid_np.shape[-1] if tls_valid_np.ndim >= 2 else 0

    sim_xyyaw = sim_states.current_sim_trajectory.stack_fields(('x', 'y', 'yaw'))[..., 0, :][:, b_ix] # (80, N=128, 3)
    sim_sdc_state = sim_states.current_sim_trajectory.stack_fields(('x', 'y', 'yaw', 'vel_x', 'vel_y', 'speed'))[..., 0, :][:, b_ix][:, is_sdc][:, 0] # (80, 6)

    planner_next_states = jnp.stack([jnp.stack(
        [x['planner_next_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 3)
    inv_opt_states = jnp.stack([jnp.stack(
        [x['inv_opt_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 2)
    odometries = jnp.stack([jnp.stack(
        [x['odometry_next_state'] for x in sim[-1]], axis=0) 
        for sim in simulations], axis=0)[..., 0, b_ix, :] # (Nsim=4, Nhorizon=10, T=80, 3)
    Nsim = planner_next_states.shape[0]
    Nhorizon = planner_next_states.shape[1]

    inv_opt_state_norms = jnp.linalg.norm(inv_opt_states[:, 0], ord=2, axis=-1) # (Nsim, T, 1)
    inv_opt_state_norms_np = np.asarray(inv_opt_state_norms)

    all_log_traj = sim_states.log_trajectory.xy # (T, B, N)
    all_log_val = sim_states.log_trajectory.valid # (T, B, N)
    ego_indices = np.where(is_sdc_mask)[0]
    ego_idx = int(ego_indices[0]) if ego_indices.size else 0

    center_idx = min(45, ego_log_traj_np.shape[0] - 1)
    cx, cy = ego_log_traj_np[center_idx, 0], ego_log_traj_np[center_idx, 1]
    r = float(np.max(np.stack([
        np.abs(ego_log_traj_np[:, 0] - cx).max() + min_r,
        np.abs(ego_log_traj_np[:, 1] - cy).max() + min_r,
    ])))
    r = max(r, float(min_r))

    planner_label_added = False
    odometry_label_added = False
    realized_label_added = False
    inv_cbar = None
    combined_ego_traj = np.concatenate([ego_log_traj_np[:log_horizon], sim_sdc_traj_np], axis=0) if log_horizon else sim_sdc_traj_np

    def render_box(center, yaw, agent_idx, facecolor, edgecolor, alpha, zorder, linewidth, from_log_phase=False):
        rel = base_box_rel_corners[agent_idx] if agent_idx < len(base_box_rel_corners) else None
        if rel is None:
            ax.scatter(center[0], center[1], s=40, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
            return
        base_yaw = base_box_base_yaw[agent_idx] if agent_idx < len(base_box_base_yaw) else 0.0
        yaw_delta = yaw - base_yaw
        if not from_log_phase:
            yaw_delta = yaw_delta + np.pi / 2.0
        c, s = np.cos(yaw_delta), np.sin(yaw_delta)
        rot = np.array([[c, -s], [s, c]])
        corners = rel @ rot.T + center
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        )

    def draw_frame(frame_idx):
        nonlocal planner_label_added, odometry_label_added, realized_label_added, inv_cbar

        ax.clear()
        if inv_cbar is not None:
            inv_cbar.remove()
            inv_cbar = None
        t_lim = min(frame_idx, total_frames - 1)

        # Road graph
        ax.scatter(rg_np[b_ix, rg_val_np[b_ix], 0], rg_np[b_ix, rg_val_np[b_ix], 1], c=rg_type_np[b_ix, rg_val_np[b_ix]], cmap=roadgraph_cmap, linewidth=roadgraph_linewidth, s=roadgraph_size, alpha=roadgraph_alpha)

        # Plot traffic lights
        if plot_traffic_lights and tls_frames:
            tls_frame = min(t_lim, tls_frames - 1)
            tls_mask = tls_valid_np[:, tls_frame]
            if tls_mask.any():
                tls_xy_frame = tls_xy_np[:, tls_frame][tls_mask]
                tls_state_frame = tls_state_np[:, tls_frame][tls_mask]
                for xy, state in zip(tls_xy_frame, tls_state_frame):
                    tl_color = color.TRAFFIC_LIGHT_COLORS[int(state)]
                    ax.plot(xy[0], xy[1], marker=traffic_light_marker, color=tl_color, ms=traffic_light_size)

        if plot_gt:
            ax.plot(
                ego_log_traj_np[:, 0],
                ego_log_traj_np[:, 1],
                '--',
                c='#007fff',
                linewidth=2.0,
                alpha=0.6,
                label='Ground truth',
                zorder=2,
            )

        if plot_realised_traj:
            ax.plot(
                combined_ego_traj[:t_lim + 1, 0],
                combined_ego_traj[:t_lim + 1, 1],
                c='C1',
                alpha=1.0,
                label='Physical trajectory',
                linewidth=2.0,
                linestyle='-',
                zorder=4,
            )
            realized_label_added = True

        if combined_xy.size and num_agents:
            t_sim_idx = min(t_lim, combined_xy.shape[0] - 1)
            log_phase = t_sim_idx < log_horizon
            if combined_valid.shape[0] > t_sim_idx and combined_valid.shape[1] > ego_idx and combined_valid[t_sim_idx, ego_idx]:
                ego_pos = combined_xy[t_sim_idx, ego_idx]
                yaw_val = float(combined_yaw[t_sim_idx, ego_idx]) if combined_yaw.shape[0] > t_sim_idx and combined_yaw.shape[1] > ego_idx else 0.0
                render_box(ego_pos, yaw_val, ego_idx, facecolor='C0', edgecolor='black', alpha=1.0, zorder=5, linewidth=1.5, from_log_phase=log_phase)

            for agent_idx in range(num_agents):
                if agent_idx == ego_idx:
                    continue
                if combined_valid.shape[0] > t_sim_idx and combined_valid.shape[1] > agent_idx:
                    hist_mask = combined_valid[:t_sim_idx + 1, agent_idx]
                    if np.any(hist_mask):
                        agent_hist = combined_xy[:t_sim_idx + 1, agent_idx][hist_mask]
                        ax.plot(agent_hist[:, 0], agent_hist[:, 1], c='gray', alpha=0.5, linewidth=1.0, zorder=2)
                    if combined_valid[t_sim_idx, agent_idx]:
                        agent_pos = combined_xy[t_sim_idx, agent_idx]
                        yaw_val = float(combined_yaw[t_sim_idx, agent_idx]) if combined_yaw.shape[0] > t_sim_idx and combined_yaw.shape[1] > agent_idx else 0.0
                        render_box(agent_pos, yaw_val, agent_idx, facecolor='wheat', edgecolor='black', alpha=0.9, zorder=4, linewidth=1.5, from_log_phase=log_phase)

        dt = 0.1
        if plot_planner and t_lim >= log_horizon:
            t_fixed = int(np.clip(t_lim - log_horizon, 0, sim_len - 1))
            for s in range(Nsim):
                xy = sim_sdc_state[t_fixed, :2]
                yaw = sim_sdc_state[t_fixed, 2]
                speed = sim_sdc_state[t_fixed, -1]
                velxy = sim_sdc_state[t_fixed, 3:5]

                true_coords = [xy]
                current_xy = xy
                current_yaw = yaw
                current_speed = speed
                current_velxy = velxy

                for h in range(1): # Nhorizon
                    planned_vxy = planner_next_states[s, h, t_fixed, 1:]
                    st_pose = ObjectPose2D.from_center_and_yaw(xy=current_xy, yaw=current_yaw)
                    st_pose_matrix = st_pose.matrix[None]
                    st_delta_yaw = st_pose.delta_yaw
                    st_pose_matrix_inv = jnp.linalg.inv(st_pose_matrix)
                    current_velxy_local = transform_direction(pts_dir=current_velxy[None], pose_matrix=st_pose_matrix)[0]
                    next_state_vxy = current_velxy_local + planned_vxy
                    next_state_vxy = transform_direction(pts_dir=next_state_vxy[None], pose_matrix=st_pose_matrix_inv)[0]
                    next_state_speed = jnp.sqrt(next_state_vxy[0]**2 + next_state_vxy[1]**2)
                    accel = (next_state_speed - current_speed) / dt
                    steering = (jnp.arctan2(next_state_vxy[1], next_state_vxy[0]) - current_yaw) / (current_speed * dt + 0.5 * accel * dt**2)
                    next_x = current_xy[0] + current_velxy[0] * dt + 0.5 * jnp.cos(current_yaw) * accel * dt**2
                    next_y = current_xy[1] + current_velxy[1] * dt + 0.5 * jnp.sin(current_yaw) * accel * dt**2
                    next_yaw = current_yaw + steering * (current_speed * dt + 0.5 * accel * dt**2)
                    next_speed = current_speed + accel * dt
                    next_vx = next_speed * jnp.cos(next_yaw)
                    next_vy = next_speed * jnp.sin(next_yaw)
                    current_xy = jnp.stack([next_x, next_y], axis=-1)
                    current_yaw = next_yaw
                    current_speed = next_speed
                    current_velxy = jnp.stack([next_vx, next_vy], axis=-1)
                    true_coords.append(current_xy)
                
                planned_traj = jnp.stack(true_coords, axis=0)
                if planner_label_added == False:
                    ax.scatter(planned_traj[:, 0], planned_traj[:, 1], s=5, c=planner_color, label='Planned traj', zorder=30)
                    planner_label_added = True
                else:
                    ax.scatter(planned_traj[:, 0], planned_traj[:, 1], s=5, c=planner_color, zorder=30)
                break

        if plot_inverse_state and t_lim >= log_horizon:
            t_sim_lim = int(np.clip(t_lim - log_horizon, 0, sim_len - 1))
            inv_vals = inv_opt_state_norms_np[0, :t_sim_lim + 1]
            inv_norm = ax.scatter(sim_sdc_traj_np[:t_sim_lim + 1, 0], sim_sdc_traj_np[:t_sim_lim + 1, 1], c=np.log(inv_vals), zorder=4, alpha=0.3, cmap='jet')
            inv_cbar = plt.colorbar(inv_norm, ax=ax, pad=0.01)
            inv_cbar.set_label('Log of norm of inv. state displacement')

        if plot_odometry and t_lim >= log_horizon:
            anchor_steps = set(step for step in odometry_anchor_steps if step <= t_lim)
            anchor_steps.update(range(0, t_lim + 1, max(1, int(plot_odometry_length))))
            anchor_steps.add(t_lim)
            for t_fixed in sorted(anchor_steps):
                if t_fixed < log_horizon:
                    continue
                sim_idx = t_fixed - log_horizon
                if sim_idx < 0 or sim_idx >= sim_xy_all_np.shape[0]:
                    continue
                if sim_valid_all_np.shape[0] <= sim_idx or sim_valid_all_np.shape[1] <= ego_idx or not sim_valid_all_np[sim_idx, ego_idx]:
                    continue

                offset_loc = sim_xy_all_np[sim_idx, ego_idx]
                current_yaw = 0.0
                if sim_yaw_np is not None and sim_yaw_np.shape[0] > sim_idx and sim_yaw_np.shape[1] > ego_idx:
                    current_yaw = float(sim_yaw_np[sim_idx, ego_idx])
                
                for s_ix in range(Nsim):
                    odometries_t_ix = np.asarray(odometries[s_ix, :, sim_idx, :2], dtype=np.float32) # (Nhorizon, 2)
                    odometries_t_ix_theta = np.asarray(odometries[s_ix, :, sim_idx, 2], dtype=np.float32)
                    
                    true_coords = [offset_loc]
                    current_xy = offset_loc
                    cyaw = current_yaw
                    for k in range(Nhorizon):
                        p = ObjectPose2D.from_center_and_yaw(xy=np.asarray(current_xy[None], dtype=np.float32), yaw=np.asarray([cyaw], dtype=np.float32))
                        p_pose_matrix = p.matrix[None]
                        p_delta_yaw = p.delta_yaw
                        p_pose_matrix_inv = jnp.linalg.inv(p_pose_matrix)

                        current_xy = transform_points(pts=odometries_t_ix[k][None], pose_matrix=p_pose_matrix_inv[0]) # (1, 2)
                        cyaw = float(-(odometries_t_ix_theta[k] + p_delta_yaw[0]))
                        true_coords.append(current_xy[0])
                        current_xy = current_xy[0]
                    
                    odometries_t_ix = jnp.stack(true_coords, axis=0)
                    if odometry_label_added == False:
                        assert odometries_t_ix.shape[0] >= 5
                        ax.scatter([], [], s=odometry_marker_size, edgecolor='black', facecolors='none', linewidth=0.8, alpha=1.0, zorder=10)
                        ax.scatter(odometries_t_ix[:plot_odometry_length, 0], odometries_t_ix[:plot_odometry_length, 1], s=odometry_marker_size, edgecolor='black', facecolors='none', linewidth=0.8, alpha=1.0, zorder=8)
                        odometry_label_added = True
                    else:
                        ax.scatter(odometries_t_ix[:plot_odometry_length, 0], odometries_t_ix[:plot_odometry_length, 1], s=odometry_marker_size, edgecolor='black', facecolors='none', linewidth=0.8, alpha=1.0, zorder=8)

        # Fallback static rendering if no sim trajectory to animate agents.
        if not (sim_xy_all_np.size and num_agents):
            t_idx = min(t_lim, all_log_traj.shape[0] - 1)
            for k in range(128):
                if is_sdc_mask[k]:
                    continue
                log_sdc_traj_k = all_log_traj[t_idx, b_ix, k]
                val = all_log_val[t_idx, b_ix, k]
                if jnp.any(val):
                    ax.plot(log_sdc_traj_k[val, 0], log_sdc_traj_k[val, 1], c='gray', alpha=0.5)
                    pts = log_box_corners_np[min(t_idx, log_box_corners_np.shape[0] - 1), b_ix, k, 0] # (4, 2)
                    box = Polygon(pts, closed=True, fill=True, facecolor='wheat', edgecolor='black', linewidth=1.5)
                    ax.add_patch(box)


        ax.set_xlabel('Δx [m]')
        ax.set_ylabel('Δy [m]')
        ax.set_box_aspect(1.0)
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        def fmt_x(x, _):
            return f"{x - cx:.1f}"
        def fmt_y(y, _):
            return f"{y - cy:.1f}"
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        handles, labels = ax.get_legend_handles_labels()

        # Proxy for a transparent, black-edge scatter circle
        odometry_proxy = Line2D(
            [], [],                         # no data
            marker='o',
            markersize=8,
            markerfacecolor='none',         # transparent
            markeredgecolor='black',
            linestyle='none',
            label='Odometry'
        )

        if plot_odometry:
            handles.append(odometry_proxy)
            labels.append('Odometry')

        if handles:
            ax.legend(handles, labels, loc=loc)
        

        ax.set_title(f'{title_prefix} {frame_idx}')
        return []

    interval_ms = 1000.0 / max(fps, 1)
    anim = mpl_animation.FuncAnimation(fig, draw_frame, frames=frame_indices, interval=interval_ms, blit=False)
    output_path = save or 'tmp_animation.gif'
    writer_fps = max(fps, 1)
    anim.save(output_path, writer=mpl_animation.PillowWriter(fps=writer_fps))
    plt.close(fig)
    return output_path

# plot_animated_awm(sim_states, save='tmp3.gif', b_ix=4, plot_planner=False, plot_odometry=True, plot_inverse_state=False, odometry_anchor_steps=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
