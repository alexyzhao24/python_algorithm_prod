'''

Tour the emg2pose package for Human-Robot Interface with EMG data

https://github.com/facebookresearch/emg2pose

'''

import pandas as pd
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

home_dir = os.environ.get('HOME')

# Download the EMG2Pose metadata CSV file from the specified URL
subprocess.run(
    ["curl", "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv", "-o", "emg2pose_metadata.csv"],
    check=True
)
# curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv -o emg2pose_metadata.csv

metadata_df = pd.read_csv(f"./emg2pose_metadata.csv")
metadata_df.head(5)

# Assuming the mini-dataset is alreay downloaded and extracted

# Now the fun starts
# Load in all datasets
sessions = sorted(glob.glob(f"{home_dir}/projects/Repos_HumanRobotInterface/emg2pose/emg2pose_dataset_mini/*.hdf5"))
print("Number of sessions found:", len(sessions))
sessions

# Now Let's look at a Dataset 15
from emg2pose.data import Emg2PoseSessionData

session = sessions[15]
data = Emg2PoseSessionData(hdf5_path=session)

print(f"Session Info:")
print(f"1) Filename: {data.metadata['filename']}")
print(f"2) {data.fields}:")

print(f"{'emg shape: ':<20} {data['emg'].shape}")
print(f"{'joint_angles shape: ':<20} {data['joint_angles'].shape}")
print(f"{'time shape: ':<20} {data['time'].shape}")

metadata_df[metadata_df["filename"] == data.metadata["filename"]]

# visualize the data
import emg2pose.visualization as visualization

print("Visualizing the EMG data of chosen session")
ax = visualization.ik_failure_plot(data)
plt.show() # display the figure and ax

# downsample joint angles to 30Hz
from emg2pose.utils import downsample
joint_angles = data["joint_angles"]
joint_angles_30hz = downsample(joint_angles, native_fs=2000, target_fs=30)

assert not np.any(np.isnan(joint_angles_30hz))

# Visualize the hand meshs: works in Jupyter not showing up for python script
print(f"Visualizing downsampled joint angles shape: {joint_angles_30hz.shape}")
fig1 = visualization.plot_hand_mesh(joint_angles_30hz[100], auto_range=False)
fig1.show()  # this is needed to display the figure in regular Python scripts, not needed in Jupyter

# Generate 30 hz animated hand meshs: works in Jupyter not showing up for python script
print(f"Animated joint angles shape: {joint_angles_30hz.shape}")
fig2 = visualization.get_plotly_animation_for_joint_angles(joint_angles_30hz[0:250])
fig2.show()  # this is needed to display the figure in regular Python scripts, not needed in Jupyter

''' Crashing issue for now to generate a video with mediapy even in Jupyter
# Render the Plotly Animation to Video Frames
import mediapy

frames = visualization.joint_angles_to_frames_parallel(joint_angles_30hz[0:250])
frames = visualization.remove_alpha_channel(frames)
mediapy.show_video(frames, width=800, fps=30, downsample=True)
'''

## Let's Load a Checkpoint and Generate some Predictions
from emg2pose.utils import generate_hydra_config_from_overrides

config = generate_hydra_config_from_overrides(
    overrides=[
        "experiment=tracking_vemg2pose",
        f"checkpoint={home_dir}/projects/Repos_HumanRobotInterface/emg2pose/emg2pose_model_checkpoints/regression_vemg2pose.ckpt"
    ]
)

from emg2pose.lightning import Emg2PoseModule

module = Emg2PoseModule.load_from_checkpoint(
    config.checkpoint,
    network=config.network,
    optimizer=config.optimizer,
    lr_scheduler=config.lr_scheduler,
)
print(f"Loaded model from {config.checkpoint} onto device {module.device}")

session = data
start_idx = 0
stop_idx = 10_000

import torch

session_window = session[start_idx:stop_idx]

# no_ik_failure is not a field so we slice separately
no_ik_failure_window = session.no_ik_failure[start_idx:stop_idx]

batch = {
    "emg": torch.Tensor([session_window["emg"].T]),  # BCT
    "joint_angles": torch.Tensor([session_window["joint_angles"].T]),  # BCT
    "no_ik_failure": torch.Tensor([no_ik_failure_window]),  # BT
}

## forward prediction: make sure that both model and input are on the same device (GPU or CPU)!
# For each value v, if it has a to method (i.e., it is a tensor), it is moved to the target device (typically GPU).
# Non-tensor values are left unchanged.
batch = {k: v.to(module.device) if hasattr(v, 'to') else v for k, v in batch.items()}
preds, joint_angles, no_ik_failure = module.forward(batch)

# Algorithms that use the initial state for ground truth will do poorly
# when the first joint angles are missing!
if (joint_angles[:, 0] == 0).all():
    print(
        "Warning! Ground truth not available at first time step!"
    )

# BCT --> TC (as numpy): please remove the graidents using detach().cpu() if you want to use the numpy array
if module.device.type == 'cuda':
    preds = preds[0].T.detach().cpu().numpy()
    joint_angles = joint_angles[0].T.detach().cpu().numpy()
else:
    preds = preds[0].T.detach().numpy()
    joint_angles = joint_angles[0].T.detach().numpy()

preds.shape
joint_angles.shape

joint_angles_30hz = downsample(joint_angles, native_fs = 2000, target_fs = 30)
visualization.get_plotly_animation_for_joint_angles(joint_angles_30hz[0:250], color="gray")

preds_30hz = downsample(preds, native_fs=2000, target_fs=30)
visualization.get_plotly_animation_for_joint_angles(preds_30hz[0:250], color="lightpink")

''' Currently have issue of generating videos
## Compare the Ground Truth and Predictions Side-by-Side
gt_frames = visualization.joint_angles_to_frames_parallel(joint_angles_30hz[0:250], color="gray")
pred_frames = visualization.joint_angles_to_frames_parallel(preds_30hz[0:250], color="lightpink")

gt_frames = visualization.remove_alpha_channel(gt_frames)
pred_frames = visualization.remove_alpha_channel(pred_frames)

mediapy.show_videos(dict(gt=gt_frames, pred=pred_frames), width=400, fps=30, downsample=True)
'''

N_COLS = 2
N_ROWS = 10

fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(4*N_COLS, 2*N_ROWS))

axs_flattened = axs.flatten()
for i, ax in enumerate(axs_flattened):
    ax.set_title(f"Joint Angle {i}")
    ax.plot(joint_angles_30hz[:, i], label="gt")
    ax.plot(preds_30hz[:, i], label="pred")

    ax.legend()

fig.suptitle("Predicted vs. Ground Truth Joint Angles")

plt.tight_layout()
fig.subplots_adjust(top=0.95)

plt.show()
