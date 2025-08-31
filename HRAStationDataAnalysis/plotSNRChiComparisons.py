import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import os
import glob
import gc
import h5py
import pickle
from icecream import ic
import configparser
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from HRASimulation.HRAEventObject import HRAevent 

# --- Utility Functions ---

def loadHRAfromH5(filename):
    """
    Loads a list of HRAevent objects from an HDF5 file.
    """
    ic(f"Loading HRA event list from: {filename}")
    eventList = []
    with h5py.File(filename, 'r') as hf:
        for key in hf.keys():
            obj_bytes = hf[key][0]
            event = pickle.loads(obj_bytes)
            eventList.append(event)
    ic(f"Successfully loaded {len(eventList)} events.")
    return eventList

def load_station_data(folder, date, station_id, data_name):
    """
    Loads and concatenates data files for a specific station and data type.
    """
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        ic(f"Warning: No files found for {data_name} with pattern: {file_pattern}")
        return np.array([])
    
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
        
    return np.concatenate(data_arrays, axis=0)

def get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNR, Chi, and weights from the HRAevent list, separating them
    into direct and reflected triggers using separate weight names.
    """
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}

    for event in HRAeventList:
        direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
        if not np.isnan(direct_weight) and direct_weight > 0:
            triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
            if triggered_direct:
                split_weight = direct_weight / len(triggered_direct)
                for st_id in triggered_direct:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        direct_data['snr'].append(snr)
                        direct_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        direct_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        direct_data['weights'].append(split_weight)

        reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
        if not np.isnan(reflected_weight) and reflected_weight > 0:
            triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
            if triggered_reflected:
                split_weight = reflected_weight / len(triggered_reflected)
                for st_id in triggered_reflected:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        reflected_data['snr'].append(snr)
                        reflected_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        reflected_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        reflected_data['weights'].append(split_weight)

    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])
            
    return direct_data, reflected_data

def get_all_cut_masks(data_dict, cuts, cut_type='rcr'):
    """
    Applies all cut combinations and returns a dictionary of boolean masks.
    cut_type: 'rcr' for RCR events, 'backlobe' for backlobe events
    """
    snr = data_dict['snr']
    chircr = data_dict['ChiRCR']
    chi2016 = data_dict['Chi2016']
    
    if cut_type == 'rcr':
        # Original RCR cuts
        chi_rcr_snr_cut_values = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
        chi_diff = chircr - chi2016
        
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chircr > chi_rcr_snr_cut_values
        masks['chi_diff_cut'] = chi_diff > cuts['chi_diff_threshold']
        
        masks['snr_and_snr_line'] = masks['snr_cut'] & masks['snr_line_cut']
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
        
    elif cut_type == 'backlobe':
        # Backlobe cuts (reversed logic)
        chi_2016_snr_cut_values = np.interp(snr, cuts['chi_2016_line_snr'], cuts['chi_2016_line_chi'])
        chi_diff = chircr - chi2016
        
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chi2016 > chi_2016_snr_cut_values
        masks['chi_diff_cut'] = chi_diff < cuts['chi_diff_threshold']
        
        masks['snr_and_snr_line'] = masks['snr_cut'] & masks['snr_line_cut']
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
    
    return masks

def calculate_cut_stats_table(data_dict, cuts, is_sim, title, pre_mask_count=None, cut_type='rcr'):
    """Calculates the percentage of events/weight passing each cut individually and all together."""
    lines = [f"{title}:"]
    
    if pre_mask_count is not None:
        lines.append(f"- {'All Data (pre-mask)':<20}: {pre_mask_count}")

    if is_sim:
        total_val = np.sum(data_dict['weights'])
        if total_val == 0: return f"{title}\nNo weight in dataset."
    else:
        total_val = len(data_dict['snr'])
        if total_val == 0: return f"{title}\nNo events in dataset."
        lines.append(f"- {'All Data (post-mask)':<20}: {total_val}")
    
    weights = data_dict.get('weights', np.ones_like(data_dict['snr']))
    masks = get_all_cut_masks(data_dict, cuts, cut_type=cut_type)
    
    cut_masks_to_report = {
        f"SNR < {cuts['snr_max']}": masks['snr_cut'],
        "ChiRCR > SNR Line": masks['snr_line_cut'],
        f"ChiRCR - Chi2016 > {cuts['chi_diff_threshold']}": masks['chi_diff_cut'],
        "All Cuts": masks['all_cuts']
    }
    
    for name, mask in cut_masks_to_report.items():
        passing_val = np.sum(weights[mask])
        if is_sim:
            lines.append(f"- {name:<20}: {passing_val/total_val:>7.2%}")
        else:
            lines.append(f"- {name:<20}: {int(passing_val)} ({passing_val/total_val:.2%})")
            
    return "\n".join(lines)


# --- Plotting Functions ---

def set_plot_labels(ax, xlabel, ylabel, title, xlim, ylim, xscale='linear', yscale='linear'):
    """Set common labels and properties for a subplot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(visible=True, which='both', axis='both')

def draw_cut_visuals(ax, plot_key, cuts_dict, cut_type='rcr'):
    """Draws lines and shaded regions for cuts on a given subplot."""
    snr_max = cuts_dict['snr_max']
    chi_diff_threshold = cuts_dict['chi_diff_threshold']

    if 'snr' in plot_key:
        ax.axvline(x=snr_max, color='m', linestyle='--', linewidth=1.5)
        # Shade the GOOD region (SNR < threshold) with normal shading
        ax.fill_betweenx(ax.get_ylim(), ax.get_xlim()[0], snr_max, color='m', alpha=0.2)
        # Shade the BAD region (SNR > threshold) with dashed pattern
        ax.fill_betweenx(ax.get_ylim(), snr_max, ax.get_xlim()[1], color='m', alpha=0.1, hatch='///')

    if plot_key == 'snr_vs_chircr':
        if cut_type == 'rcr':
            snr_line_snr = cuts_dict['chi_rcr_line_snr']
            snr_line_chi = cuts_dict['chi_rcr_line_chi']
            ax.plot(snr_line_snr, snr_line_chi, color='purple', linestyle='--', linewidth=1.5)
            # Shade the GOOD region (above the line) with normal shading
            ax.fill_between(snr_line_snr, snr_line_chi, 1, color='purple', alpha=0.2, interpolate=True)
            # Shade the BAD region (below the line) with dashed pattern
            ax.fill_between(snr_line_snr, 0, snr_line_chi, color='purple', alpha=0.1, hatch='///')
        elif cut_type == 'backlobe':
            snr_line_snr = cuts_dict['chi_2016_line_snr']
            snr_line_chi = cuts_dict['chi_2016_line_chi']
            ax.plot(snr_line_snr, snr_line_chi, color='orange', linestyle='--', linewidth=1.5)
            # Shade the GOOD region (above the line) with normal shading
            ax.fill_between(snr_line_snr, snr_line_chi, 1, color='orange', alpha=0.2, interpolate=True)
            # Shade the BAD region (below the line) with dashed pattern
            ax.fill_between(snr_line_snr, 0, snr_line_chi, color='orange', alpha=0.1, hatch='///')
    
    elif plot_key == 'snr_vs_chi2016':
        if cut_type == 'backlobe':
            snr_line_snr = cuts_dict['chi_2016_line_snr']
            snr_line_chi = cuts_dict['chi_2016_line_chi']
            ax.plot(snr_line_snr, snr_line_chi, color='orange', linestyle='--', linewidth=1.5)
            # Shade the GOOD region (above the line) with normal shading
            ax.fill_between(snr_line_snr, snr_line_chi, 1, color='orange', alpha=0.2, interpolate=True)
            # Shade the BAD region (below the line) with dashed pattern
            ax.fill_between(snr_line_snr, 0, snr_line_chi, color='orange', alpha=0.1, hatch='///')
    
    elif plot_key == 'chi_vs_chi':
        if cut_type == 'rcr':
            # Draw Chi difference cut line: ChiRCR = Chi2016 + threshold
            x_vals = np.linspace(0, 1, 100)
            y_vals = x_vals + chi_diff_threshold
            # Only show the part where y_vals <= 1
            valid_mask = y_vals <= 1
            if np.any(valid_mask):
                ax.plot(x_vals[valid_mask], y_vals[valid_mask], color='darkgreen', linestyle='--', linewidth=1.5)
                # Shade the GOOD region (above the line) with normal shading
                ax.fill_between(x_vals[valid_mask], y_vals[valid_mask], 1, color='darkgreen', alpha=0.2)
                # Shade the BAD region (below the line) with dashed pattern
                ax.fill_between(x_vals[valid_mask], 0, y_vals[valid_mask], color='darkgreen', alpha=0.1, hatch='///')
        elif cut_type == 'backlobe':
            # Draw Chi difference cut line: ChiRCR = Chi2016 + threshold (but reversed)
            x_vals = np.linspace(0, 1, 100)
            y_vals = x_vals + chi_diff_threshold
            # Only show the part where y_vals <= 1
            valid_mask = y_vals <= 1
            if np.any(valid_mask):
                ax.plot(x_vals[valid_mask], y_vals[valid_mask], color='darkorange', linestyle='--', linewidth=1.5)
                # Shade the GOOD region (below the line) with normal shading
                ax.fill_between(x_vals[valid_mask], 0, y_vals[valid_mask], color='darkorange', alpha=0.2)
                # Shade the BAD region (above the line) with dashed pattern
                ax.fill_between(x_vals[valid_mask], y_vals[valid_mask], 1, color='darkorange', alpha=0.1, hatch='///')
    
    elif plot_key == 'snr_vs_chidiff':
        if cut_type == 'rcr':
            # Draw horizontal line for Chi difference cut
            ax.axhline(y=chi_diff_threshold, color='darkgreen', linestyle='--', linewidth=1.5)
            # Shade the GOOD region (above the line) with normal shading
            ax.fill_betweenx([chi_diff_threshold, ax.get_ylim()[1]], ax.get_xlim()[0], ax.get_xlim()[1], color='darkgreen', alpha=0.2)
            # Shade the BAD region (below the line) with dashed pattern
            ax.fill_betweenx([ax.get_ylim()[0], chi_diff_threshold], ax.get_xlim()[0], ax.get_xlim()[1], color='darkgreen', alpha=0.1, hatch='///')
        elif cut_type == 'backlobe':
            # Draw horizontal line for Chi difference cut (reversed)
            ax.axhline(y=chi_diff_threshold, color='darkorange', linestyle='--', linewidth=1.5)
            # Shade the GOOD region (below the line) with normal shading
            ax.fill_betweenx([ax.get_ylim()[0], chi_diff_threshold], ax.get_xlim()[0], ax.get_xlim()[1], color='darkorange', alpha=0.2)
            # Shade the BAD region (above the line) with dashed pattern
            ax.fill_betweenx([chi_diff_threshold, ax.get_ylim()[1]], ax.get_xlim()[0], ax.get_xlim()[1], color='darkorange', alpha=0.1, hatch='///')


def plot_2x2_grid(fig, axs, base_data_config, cuts_dict, overlays=None, hist_bins_dict=None):
    """
    Master function to generate a 2x2 grid. Plots a base layer (scatter or hist)
    and then adds any number of overlay layers on top.
    """
    im = None
    base_data = base_data_config['data']
    base_plot_type = base_data_config['type']

    plot_configs = {
        'snr_vs_chi2016': {'xlabel': 'SNR', 'ylabel': 'Chi2016', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'snr_vs_chircr': {'xlabel': 'SNR', 'ylabel': 'ChiRCR', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'chi_vs_chi': {'xlabel': 'Chi2016', 'ylabel': 'ChiRCR', 'xlim': (0, 1), 'ylim': (0, 1)},
        'snr_vs_chidiff': {'xlabel': 'SNR', 'ylabel': 'ChiRCR - Chi2016', 'xlim': (3, 100), 'ylim': (-0.4, 0.4), 'xscale': 'log'}
    }
    
    for i, (key, p) in enumerate(plot_configs.items()):
        ax = axs.flatten()[i]
        set_plot_labels(ax, p['xlabel'], p['ylabel'], f"{p['ylabel']} vs {p['xlabel']}", p.get('xlim'), p.get('ylim'), p.get('xscale', 'linear'))
        
        # Plot Base Layer
        base_map = {
            'snr_vs_chi2016': {'x': base_data['snr'], 'y': base_data['Chi2016']},
            'snr_vs_chircr': {'x': base_data['snr'], 'y': base_data['ChiRCR']},
            'chi_vs_chi': {'x': base_data['Chi2016'], 'y': base_data['ChiRCR']},
            'snr_vs_chidiff': {'x': base_data['snr'], 'y': base_data['ChiRCR'] - base_data['Chi2016']}
        }
        if base_plot_type == 'scatter':
            ax.scatter(base_map[key]['x'], base_map[key]['y'], **base_data_config['style'])
        elif base_plot_type == 'hist' and np.sum(base_data.get('weights', 0)) > 0:
            h, xedges, yedges, im_temp = ax.hist2d(base_map[key]['x'], base_map[key]['y'], bins=hist_bins_dict[key], weights=base_data['weights'], norm=colors.LogNorm())
            if im is None: im = im_temp

        # Plot Overlay Layers
        if overlays:
            for overlay in overlays:
                overlay_data = overlay['data']
                overlay_map = {
                    'snr_vs_chi2016': {'x': overlay_data['snr'], 'y': overlay_data['Chi2016']},
                    'snr_vs_chircr': {'x': overlay_data['snr'], 'y': overlay_data['ChiRCR']},
                    'chi_vs_chi': {'x': overlay_data['Chi2016'], 'y': overlay_data['ChiRCR']},
                    'snr_vs_chidiff': {'x': overlay_data['snr'], 'y': overlay_data['ChiRCR'] - overlay_data['Chi2016']}
                }
                x_data, y_data = overlay_map[key]['x'], overlay_map[key]['y']
                
                if overlay['style'].get('color_by_weight'):
                    weights = overlay_data['weights']
                    sort_indices = np.argsort(weights)
                    x_data, y_data, weights = x_data[sort_indices], y_data[sort_indices], weights[sort_indices]
                    
                    # Enhanced weight visualization with better color mapping and size scaling
                    norm = colors.LogNorm(vmin=weights.min(), vmax=weights.max())
                    scatter = ax.scatter(x_data, y_data, c=weights, cmap='viridis', norm=norm, 
                                       alpha=overlay['style']['alpha'], s=overlay['style']['s'])
                    
                    # Add colorbar for weight visualization if this is the first overlay
                    if overlay == overlays[0]:  # Only add colorbar for first overlay to avoid duplicates
                        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
                        cbar.set_label('Event Weight', fontsize=8)
                        cbar.ax.tick_params(labelsize=7)
                else:
                    ax.scatter(x_data, y_data, **overlay['style'])

        if cuts_dict:
            draw_cut_visuals(ax, key, cuts_dict, cut_type='rcr')  # For now, only show RCR cuts on main plots
        if key == 'chi_vs_chi':
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1)
            
            legend_elements = []
            if base_data_config['label']:
                if base_plot_type == 'hist':
                    legend_elements.append(plt.Rectangle((0,0),1,1,fc="lightblue", label=base_data_config['label']))
                else:
                    legend_elements.append(Line2D([0], [0], marker=base_data_config['style'].get('marker', 'o'), color='w', label=base_data_config['label'], markerfacecolor=base_data_config['style']['c'], markersize=8))

            if overlays:
                for overlay in overlays:
                    if overlay['label']:
                        if overlay['style'].get('color_by_weight'):
                            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=overlay['label'], markerfacecolor='blue', markersize=8, alpha=0.5))
                        else:
                            legend_elements.append(Line2D([0], [0], marker=overlay['style'].get('marker', 'o'), color='w', label=overlay['label'], markerfacecolor=overlay['style']['c'], markersize=8))
            ax.legend(handles=legend_elements, loc='upper left')

    return im

def run_analysis_for_station(station_id, station_data, event_ids, unique_indices, pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date):
    """
    Runs the full plotting and saving pipeline for a given station ID and its data.
    """
    ic(f"--- Running analysis for Station {station_id} ---")

    # --- Get Masks and Save Passing Events ---
    masks_rcr = get_all_cut_masks(station_data, cuts, cut_type='rcr')
    masks_backlobe = get_all_cut_masks(station_data, cuts, cut_type='backlobe')
    
    passing_events_to_save = {}
    
    # Note: The keys here must be valid Python identifiers for np.savez
    passing_events_to_save['snr_cut_only'] = np.zeros(np.sum(masks_rcr['snr_cut']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['snr_and_snr_line'] = np.zeros(np.sum(masks_rcr['snrand_snr_line']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['all_cuts'] = np.zeros(np.sum(masks_rcr['all_cuts']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_cut_only'] = np.zeros(np.sum(masks_backlobe['snr_cut']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_and_snr_line'] = np.zeros(np.sum(masks_backlobe['snrand_snr_line']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_all_cuts'] = np.zeros(np.sum(masks_backlobe['all_cuts']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])

    passing_events_to_save['snr_cut_only']['event_id'] = event_ids[masks_rcr['snr_cut']]
    passing_events_to_save['snr_cut_only']['unique_index'] = unique_indices[masks_rcr['snr_cut']]
    passing_events_to_save['snr_and_snr_line']['event_id'] = event_ids[masks_rcr['snrand_snr_line']]
    passing_events_to_save['snr_and_snr_line']['unique_index'] = unique_indices[masks_rcr['snrand_snr_line']]
    passing_events_to_save['all_cuts']['event_id'] = event_ids[masks_rcr['all_cuts']]
    passing_events_to_save['all_cuts']['unique_index'] = unique_indices[masks_rcr['all_cuts']]
    passing_events_to_save['backlobe_cut_only']['event_id'] = event_ids[masks_backlobe['snr_cut']]
    passing_events_to_save['backlobe_cut_only']['unique_index'] = unique_indices[masks_backlobe['snr_cut']]
    passing_events_to_save['backlobe_and_snr_line']['event_id'] = event_ids[masks_backlobe['snrand_snr_line']]
    passing_events_to_save['backlobe_and_snr_line']['unique_index'] = unique_indices[masks_backlobe['snrand_snr_line']]
    passing_events_to_save['backlobe_all_cuts']['event_id'] = event_ids[masks_backlobe['all_cuts']]
    passing_events_to_save['backlobe_all_cuts']['unique_index'] = unique_indices[masks_backlobe['all_cuts']]
    
    savename = f'{plot_folder}PassingEvents_Station{station_id}_{date}.npz'
    np.savez(savename, **passing_events_to_save)
    ic(f"Saved passing event combinations for Station {station_id} to {savename}")

    # --- Plotting ---
    # Plot 1: Data Only with layered cuts
    ic("Generating layered scatter plot for data...")
    
    base_data_config = {
        'data': station_data,
        'type': 'scatter',
        'label': 'Data',
        'style': {'marker': '.', 's': 5, 'alpha': 0.4, 'c': 'gray'}
    }
    
    # Create data subsets for overlay
    data_snr_snr_line = {key: station_data[key][masks_rcr['snrand_snr_line']] for key in station_data}
    data_all_cuts = {key: station_data[key][masks_rcr['all_cuts']] for key in station_data}
    
    count_snr_snr_line = len(data_snr_snr_line['snr'])
    count_all_cuts = len(data_all_cuts['snr'])

    data_overlays = [
        {
            'data': data_snr_snr_line,
            'label': f'Pass SNR+SNR Line (N={count_snr_snr_line})',
            'style': {'marker': 'x', 's': 15, 'alpha': 0.8, 'c': 'darkturquoise'}
        },
        {
            'data': data_all_cuts,
            'label': f'Pass All Cuts (N={count_all_cuts})',
            'style': {'marker': '*', 's': 25, 'alpha': 0.9, 'c': 'magenta'}
        }
    ]

    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 14))
    fig1.suptitle(f'Data: Chi Comparison for Station {station_id} on {date}\n{rcr_cut_string}', fontsize=14)
    plot_2x2_grid(fig1, axs1, base_data_config, cuts, overlays=data_overlays)
    rcr_stats_str = calculate_cut_stats_table(station_data, cuts, is_sim=False, title="Data Stats (RCR Cuts)", pre_mask_count=pre_mask_count, cut_type='rcr')
    backlobe_stats_str = calculate_cut_stats_table(station_data, cuts, is_sim=False, title="Data Stats (Backlobe Cuts)", pre_mask_count=pre_mask_count, cut_type='backlobe')
    stats_str = f"{rcr_stats_str}\n\n{backlobe_stats_str}"
    fig1.text(0.5, 0.01, stats_str, ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    fig1.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(f'{plot_folder}Data_SNR_Chi_2x2_WithCuts_Station{station_id}_{date}.png')
    plt.close(fig1)

    # --- Other Plots (Sim vs Data, etc.) ---
    sim_base_config = {'data': sim_direct, 'type': 'hist', 'label': 'Direct Sim (Hist)'}
    data_overlay_config = {'data': station_data, 'label': 'Data', 'style': {'marker': '.', 's': 5, 'alpha': 0.8, 'c': 'orangered'}}
    reflected_overlay_config = {'data': sim_reflected, 'label': 'Reflected Sim (Weighted)', 'style': {'s': 12, 'alpha': 0.6, 'color_by_weight': True}}

    # Data over Composite Sim Plot
    ic("Generating data over composite simulation plot...")
    fig_all, axs_all = plt.subplots(2, 2, figsize=(13, 15))
    fig_all.suptitle(f'Data vs Composite Simulation - Station {station_id}\n{rcr_cut_string}', fontsize=14)
    im_all = plot_2x2_grid(fig_all, axs_all, sim_base_config, cuts, overlays=[reflected_overlay_config, data_overlay_config], hist_bins_dict=hist_bins)
    
    direct_stats = calculate_cut_stats_table(sim_direct, cuts, True, "Direct Sim", cut_type='rcr')
    reflected_stats = calculate_cut_stats_table(sim_reflected, cuts, True, "Reflected Sim", cut_type='rcr')
    fig_all.text(0.5, 0.01, f"{direct_stats}\n\n{reflected_stats}", ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    
    if im_all:
        fig_all.tight_layout(rect=[0, 0.18, 0.9, 0.95])
        cbar_ax = fig_all.add_axes([0.91, 0.2, 0.02, 0.7])
        fig_all.colorbar(im_all, cax=cbar_ax, label='Direct Weighted Counts (Evts/Yr)')
    else:
        fig_all.tight_layout(rect=[0, 0.18, 1, 0.95])
    plt.savefig(f'{plot_folder}Data_over_Sim_Composite_SNR_Chi_2x2_Station{station_id}_{date}.png')
    plt.close(fig_all)


if __name__ == "__main__":
    # --- Configuration and Setup ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    
    sim_file = config['SIMULATION']['sim_file']
    direct_weight_name = config['SIMULATION']['direct_weight_name']
    reflected_weight_name = config['SIMULATION']['reflected_weight_name']
    sim_sigma = float(config['SIMULATION']['sigma'])

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)
    
    ic.configureOutput(prefix='Chi-SNR Analysis | ')
    
    # --- Define Stations and Load Sim Data ---
    station_ids_to_process = [13, 14, 15, 17, 18, 19, 30]
    HRAeventList = loadHRAfromH5(sim_file)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    sim_direct, sim_reflected = get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=sim_sigma)

    # --- Define Cuts & Bins ---
    cuts = {
        'snr_max': 33,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.65, 0.65, 0.7, 0.76, 0.77, 0.81, 0.83]),  # More aggressive cut
        'chi_diff_threshold': 0.08,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.65, 0.65, 0.7, 0.76, 0.77, 0.81, 0.83]) # More aggressive cut
    }
    rcr_cut_string = f"RCR Cuts: SNR < {cuts['snr_max']} & ChiRCR > SNR Line & ChiRCR - Chi2016 > {cuts['chi_diff_threshold']}"
    backlobe_cut_string = f"Backlobe Cuts: SNR < {cuts['snr_max']} & Chi2016 > SNR Line & ChiRCR - Chi2016 < {cuts['chi_diff_threshold']}"
    
    log_bins = np.logspace(np.log10(3), np.log10(100), 31)
    linear_bins = np.linspace(0, 1, 31)
    diff_bins = np.linspace(-0.4, 0.4, 31)
    hist_bins = {
        'snr_vs_chi2016': [log_bins, linear_bins], 'snr_vs_chircr': [log_bins, linear_bins],
        'chi_vs_chi': [linear_bins, linear_bins], 'snr_vs_chidiff': [log_bins, diff_bins]
    }
    
    # --- Main Loop for Individual and Summed Stations ---
    all_stations_data = {key: [] for key in ['snr', 'Chi2016', 'ChiRCR']}
    all_stations_event_ids = []
    all_stations_unique_indices = []
    total_pre_mask_count = 0

    for station_id in station_ids_to_process:
        ic(f"Loading data for Station {station_id}...")
        times = load_station_data(station_data_folder, date, station_id, 'Times')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventID')
        pre_mask_count = len(times)
        total_pre_mask_count += pre_mask_count
        
        if times.size == 0: 
            ic(f"Skipping Station {station_id} due to missing Times/EventID data.")
            continue

        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        
        if Chi2016_array.size == 0 or ChiRCR_array.size == 0:
            ic(f"Skipping Station {station_id} due to missing Chi data.")
            continue

        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)
        
        station_data = {
            'snr': snr_array[initial_mask][unique_indices],
            'Chi2016': Chi2016_array[initial_mask][unique_indices],
            'ChiRCR': ChiRCR_array[initial_mask][unique_indices]
        }
        station_event_ids = event_ids_raw[initial_mask][unique_indices]
        
        ic(f"Station {station_id} has {len(station_data['snr'])} events after masking.")
        
        for key in all_stations_data:
            all_stations_data[key].append(station_data[key])
        all_stations_event_ids.append(station_event_ids)
        all_stations_unique_indices.append(unique_indices)
        
        run_analysis_for_station(station_id, station_data, station_event_ids, unique_indices, pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date)

    # --- Run Analysis for Summed Stations ---
    if len(all_stations_data['snr']) > 1:
        summed_station_data = {key: np.concatenate(all_stations_data[key]) for key in all_stations_data}
        summed_event_ids = np.concatenate(all_stations_event_ids)
        summed_unique_indices = np.concatenate(all_stations_unique_indices)
        summed_station_id = '+'.join(map(str, station_ids_to_process))
        run_analysis_for_station(summed_station_id, summed_station_data, summed_event_ids, summed_unique_indices, total_pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date)
    else:
        ic("Not enough station data to perform a summed analysis.")

    ic("Processing complete.")
