# Original X post: https://x.com/yuruyurau/status/1942231466446057727
# Wolfram community post: https://community.wolfram.com/groups/-/m/t/3516580

'''𝐓𝐡𝐞 𝐁𝐢𝐧𝐝𝐢𝐧𝐠 𝐏𝐫𝐨𝐛𝐥𝐞𝐦
Neuroscientist Francis Crick and philosopher David Chalmers identified the binding problem: how does the
brain integrate distributed neural activity into unified conscious experiences?
This mathematical "creature" exemplifies the same puzzle computationally.

Thousands of individual coordinate points become bound into a single perceived object - a creature -
despite no central coordinator in the algorithm. The binding occurs in our visual system, not the mathematics,
yet the mathematics seems pre-structured to enable this binding. Could this suggest either that mathematical
relationships naturally support perceptual integration, or that our binding mechanisms are tuned to recognize
mathematical harmonies?

Only functions used: sin(x), cos(x), sqrt(x), x^2, and addition.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb

LIVE_DISPLAY = False  # Set to False to save as GIF instead

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)  # Higher DPI for better quality
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(70, 330)
ax.set_ylim(30, 350) # ax.set_ylim(350, 30)  # Inverted y-axis to match original
ax.set_aspect('equal')
ax.axis('off')

# Dark background (matches Wolfram's GrayLevel[9/255])
bg_color = to_rgb('#000000')
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Precompute all points for animation (memory efficient)
num_frames = 150
num_points = 10000
t_values = np.linspace(0, 2*np.pi, num_frames)

# Precompute base values
i_vals = np.arange(num_points)
x = i_vals
y = i_vals / 235.0

# Pre-allocate arrays
xp_all = np.zeros((num_frames, num_points))
yp_all = np.zeros((num_frames, num_points))

# Compute all frames
print("Precomputing frames...")
for frame_idx, t in enumerate(t_values):
    # Intermediate calculations
    k = (4 + np.sin(x/11 + 8*t)) * np.cos(x/14)
    e = y/8 - 19
    d = np.sqrt(k**2 + e**2) + np.sin(y/9 + 2*t)
    q = 2*np.sin(2*k) + np.sin(y/17)*k*(9 + 2*np.sin(y - 3*d))
    c = d**2/49 - t

    # Final coordinates
    xp = q + 50*np.cos(c) + 200
    yp = q*np.sin(c) + 39*d - 440

    # Store results with y-inversion
    xp_all[frame_idx] = xp
    yp_all[frame_idx] = 400 - yp

# Create initial scatter plot
scatter = ax.scatter(xp_all[0], yp_all[0], s=0.1,
                     color='white', alpha=0.9, rasterized=True)
# Animation update function
def update(frame):
    scatter.set_offsets(np.column_stack((xp_all[frame], yp_all[frame])))
    return [scatter]

# Create animation
print("Creating animation...")
ani = FuncAnimation(fig, update, frames=num_frames,
                    blit=True, interval=33)

# Let's save the animation as a GIF or live display
if LIVE_DISPLAY:
    print("Displaying animation live...")
    plt.show()  # Uncomment to display the animation live (not recommended for large frames)
else:
    # Save as GIF
    print("Saving GIF...")
    # ani.save('creature.gif', writer=PillowWriter(fps=30),
    #          dpi=100, savefig_kwargs={'facecolor': bg_color})

    ani.save('math_creature.gif',
            writer='pillow',  # Use pillow directly
            fps=30,
            dpi=150,
            savefig_kwargs={'facecolor': bg_color}
            )
    print("Animation saved as 'math_creature.gif'")