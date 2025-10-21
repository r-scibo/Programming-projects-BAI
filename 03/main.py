#%%
from pulp import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import time
from pulp import PULP_CBC_CMD

def sec_to_time(seconds):
    """Convert seconds from midnight to HH:MM format."""
    return time.strftime("%H:%M", time.gmtime(seconds))

def bakery():
    # Input file is called './bakery.txt'
    input_filename = './bakery.txt'
    data_np = np.loadtxt(input_filename)
    
    leng = data_np.shape[0]
    
    # Initialize LP problem 
    prob = LpProblem("Bakery Schedule", LpMinimize)
    
    # Create variables 
    s_times = [f"s_{i}" for i in range(leng)]
    start_times = LpVariable.dicts("s", s_times, 0, 86400, LpInteger)
    
    end = LpVariable("end", 0, None, LpInteger)
    start = LpVariable("start", 0, None, LpInteger)
    
    v = {f"{i}_{j}": LpVariable(f"u_{i}_{j}", 0, 1, LpInteger)
         for i in range(leng) for j in range(leng) if i != j}
    
    # Objective and constraints
    prob += end - start, "Total time of baking schedule"
    
    for i in range(leng):
        prob += start_times[f"s_{i}"] >= data_np[i][1], f"Bake after ready {i}"
        prob += start_times[f"s_{i}"] + data_np[i][3] <= data_np[i][2], f"Take out before delivery {i}"
        prob += start_times[f"s_{i}"] + data_np[i][3] - end <= 0, f"Lower bound on end {i}"
        prob += start_times[f"s_{i}"] - start >= 0, f"Upper bound on start {i}"
    
    M = data_np[:, 2].max() - data_np[:, 1].min()
    for i in range(leng):
        for j in range(leng):
            if i != j:
                prob += start_times[f"s_{i}"] + data_np[i][3] <= start_times[f"s_{j}"] + M*(1 - v[f"{i}_{j}"]), f"Non overlapping {i}_{j} constraint 1"
                prob += start_times[f"s_{j}"] + data_np[j][3] <= start_times[f"s_{i}"] + M*v[f"{i}_{j}"], f"Non overlapping {i}_{j} constraint 2"
    
    # Solve 
    prob.writeLP("BakerySchedule.lp")
    prob.solve(PULP_CBC_CMD(timeLimit=150))
    
    # Extract solved start times from the LP solution.
    solved_start_times = []
    for i in range(leng):
        varname = f"s_{i}"
        solved_start_times.append(start_times[varname].varValue)
    solved_start_times = np.array(solved_start_times)
    
    finish_times = solved_start_times + data_np[:, 3]
    
    # Build list of items for plotting
    items = []
    for i in range(leng):
        items.append({
            'ID': int(data_np[i][0]),
            'start': solved_start_times[i],
            'baking': data_np[i][3],
            'finish': finish_times[i],
            'ready': data_np[i][2]
        })
    
    # Sort items 
    items.sort(key=lambda x: x['start'])
    
    # Determine color for each
    for i in range(len(items) - 1):
        gap = items[i+1]['start'] - items[i]['finish']
        items[i]['color'] = 'red' if gap <= 120 else 'blue'
    items[-1]['color'] = 'blue'
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.7)
    
    # Main chart 
    ax = fig.add_subplot(gs[0, 0])
    y_positions = range(len(items))
    for idx, item in enumerate(items):
        # Plot the colored baking interval rectangle
        ax.broken_barh([(item['start'], item['baking'])],
                       (idx - 0.4, 0.8),
                       facecolors=item['color'],
                       edgecolor='black')
        # Plot the orange marker for customer arrival (ready time)
        ax.plot(item['ready'], idx, 'o', color='orange', markersize=8)
        # Add a dashed line connecting the end of the rectangle to the orange dot
        ax.plot([item['finish'], item['ready']], [idx, idx],
                linestyle='--', color='gray', linewidth=1)
    
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"Item {it['ID']}" for it in items])
    ax.set_ylabel("Items")
    ax.set_title("Bakery Scheduling Infographic")
    
    xmin_plot = min(it['start'] for it in items) - 300
    xmax_plot = max(it['finish'] for it in items) + 300
    ax.set_xlim(xmin_plot, xmax_plot)
    
    # Include start, finish, and customer arrival times on the bottom x-axis.
    all_times = sorted(set([it['start'] for it in items] + 
                           [it['finish'] for it in items] +
                           [it['ready'] for it in items]))
    ax.set_xticks(all_times)
    ax.set_xticklabels([sec_to_time(t) for t in all_times], rotation=45)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Baking Time (HH:MM)")
    
    # Legend panel 
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.set_facecolor("lightgrey")
    legend_ax.set_axis_off()
    
    # A) Sample baking period bar (blue)
    bar_x, bar_y = 0.1, 0.55  
    bar_w, bar_h = 0.3, 0.25
    legend_bar = mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                    facecolor='blue', edgecolor='black',
                                    transform=legend_ax.transAxes)
    legend_ax.add_patch(legend_bar)
    legend_ax.text(bar_x + bar_w/2, bar_y + bar_h/2,
                   "Baking period",
                   transform=legend_ax.transAxes,
                   ha='center', va='center', fontsize=10, color='white')
    
    # B) Arrow for "Time to put in the oven"
    legend_ax.annotate("Time to put in the oven",
                       xy=(bar_x, bar_y + bar_h/2),
                       xycoords='axes fraction',
                       xytext=(bar_x, bar_y + bar_h + 0.35),
                       textcoords='axes fraction',
                       arrowprops=dict(arrowstyle="->", color="black"),
                       fontsize=10, va='center', ha='right')
    
    # C) Arrow for "Time to take out of the oven"
    legend_ax.annotate("Time to take out of the oven",
                       xy=(bar_x + bar_w, bar_y + bar_h/2),
                       xycoords='axes fraction',
                       xytext=(bar_x + bar_w + 0.15, bar_y + bar_h + 0.35),
                       textcoords='axes fraction',
                       arrowprops=dict(arrowstyle="->", color="black"),
                       fontsize=10, va='center', ha='left')
    
    # D) Orange marker for "Expected arrival of customer"
    marker_x, marker_y = 0.55, 0.5
    legend_ax.plot(marker_x, marker_y, 'o', color='orange', markersize=10,
                   transform=legend_ax.transAxes)
    legend_ax.annotate("Expected arrival of customer",
                       xy=(marker_x, marker_y),
                       xycoords='axes fraction',
                       xytext=(marker_x + 0.12, marker_y - 0.25),
                       textcoords='axes fraction',
                       arrowprops=dict(arrowstyle="->", color="black"),
                       fontsize=10, va='center', ha='left')
    
    # E) Red rectangle for "Critical item"
    crit_x, crit_y = 0.75, 0.55
    crit_w, crit_h = 0.15, 0.25
    critical_patch = mpatches.Rectangle((crit_x, crit_y), crit_w, crit_h,
                                        facecolor='red', edgecolor='black',
                                        transform=legend_ax.transAxes)
    legend_ax.add_patch(critical_patch)
    legend_ax.text(crit_x + crit_w/2, crit_y + crit_h/2,
                   "Critical item",
                   transform=legend_ax.transAxes,
                   ha='center', va='center', fontsize=10, color='white')
    
    # F) Legend title
    legend_ax.text(0.5, 1, "Legend", transform=legend_ax.transAxes,
                   fontsize=12, fontweight='bold', ha='center')
    
    # Save the visualization as PNG and display
    plt.savefig('./visualization.png', dpi=300)
    plt.show()
    
    # Return a dictionary with the start times for each pastry
    retval = {f"s_{i}": solved_start_times[i] for i in range(leng)}
    return retval

bakery()

# %%
