import numpy as np
from vispy import scene, app
from vispy.scene import visuals
from vispy.color import ColorArray
from vispy.geometry import create_box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import os
import cv2


# Example data generation: Uncomment to use for testing
# phis = [torch.randint(0,10,(2,16,24,24)) for _ in range(10)]

def create_phi_batch_list(phis):
    """ Create a list of batches from the provided data. """
    phi_batch_list = [[] for _ in range(phis[0].shape[0])]
    for phi in phis:
        for b in range(phi.shape[0]):
            phi_batch_list[b].append(phi[b])
    return phi_batch_list

class InteractiveTensorVisualization:
    def __init__(self, data, batch_idx=0):
        """ Initialize the interactive tensor visualization. 

        args: 
            data: list of Phi values (tensors)

        """
        # Convert data to batch list
        data = create_phi_batch_list(data)
        # Extract the specific batch index
        self.data = [e.detach().cpu().numpy() for e in data[batch_idx]]
        self.data = self.pool_data(self.data)  # Pool the data to fit into an 8x8x8 grid
        self.normalize_data()
        self.z, self.y, self.x = self.data[0].shape
        self.timesteps = len(self.data)
        self.current_step = 0

        # Setup the VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 600))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 50

        self.meshes = []
        self.init_scene()

        # Setup timer for animation
        self.timer = app.Timer(interval=0.5, connect=self.update)
        self.timer.start()
        app.run()

    def normalize_data(self):
        """ Normalize the data to ensure color values are between 0 and 1. """
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        self.data = (self.data - min_val) / (max_val - min_val)

    def pool_data(self, data):
        """ Pools the data to fit into an 8x8x8 grid. """
        pooled_data = []
        for frame in data:
            pooled_frame = self.pool_3d(frame, 8, 8, 8)
            pooled_data.append(pooled_frame)
        return np.array(pooled_data)

    def pool_3d(self, data, pool_z, pool_y, pool_x):
        """ Max pool the 3D data to the specified shape. """
        z, y, x = data.shape
        stride_z = z // pool_z
        stride_y = y // pool_y
        stride_x = x // pool_x
        pooled_data = np.zeros((pool_z, pool_y, pool_x))
        for i in range(pool_z):
            for j in range(pool_y):
                for k in range(pool_x):
                    pooled_data[i, j, k] = np.average(data[i*stride_z:(i+1)*stride_z, j*stride_y:(j+1)*stride_y, k*stride_x:(k+1)*stride_x])
        return pooled_data

    def init_scene(self):
        """ Initialize the 3D scene with cubes representing tensor data. """
        # Create a box mesh and extract vertices and faces
        meshdata = create_box(width=1, height=1, depth=1, width_segments=1, height_segments=1, depth_segments=1)
        vertices = meshdata[0]['position']  # Extract the position field
        faces = meshdata[1]

        # Create cubes based on tensor data
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    color_val = self.data[0][k, j, i]
                    color = (color_val, 0, 1 - color_val, max(0.1, color_val))
                    translated_vertices = vertices + np.array([i * 2, j * 2, k * 2])
                    mesh = visuals.Mesh(vertices=translated_vertices, faces=faces, color=color)
                    self.view.add(mesh)
                    self.meshes.append(mesh)

    def update(self, event):
        """ Update the scene for the next frame of animation. """
        if self.current_step + 1 < self.timesteps:
            self.current_step += 1
        else:
            self.current_step = 0
        data = self.data[self.current_step]#[0, 0]
        idx = 0
        # Update colors of the cubes based on current tensor data
        for k in range(self.z):
            for j in range(self.y):
                for i in range(self.x):
                    color_val = data[k, j, i]
                    color = (color_val, 0, 1 - color_val, max(0.1, color_val))
                    self.meshes[idx].color = ColorArray(color).rgba
                    idx += 1
        self.canvas.update()

    def on_key_press(self, event):
        """ Handle key press events for pausing and resuming the animation. """
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()


# Example usage: Uncomment to generate and use data
# tensor_data = [torch.rand(5,5,5) for i in range(10)]  # Large tensor data for visualization
# vis = InteractiveTensorVisualization(phis, 0)


class VideoAnimationTensor:
    def __init__(self, data, output_file='animation.mp4', fps=2, batch_idx=0):
        """ Initialize the video animation tensor visualization. """
        # Convert data to batch list
        data = create_phi_batch_list(data)
        # Extract the specific batch index
        self.data = [e.detach().cpu().numpy() for e in data[batch_idx]]
        
        # Number of timesteps
        self.timesteps = len(self.data)

        # Pool the data to fit into an 8x8x8 grid
        self.data = self.pool_data(self.data)
        self.z, self.y, self.x = self.data[0].shape

        # Output file and frame rate
        self.output_file = output_file
        self.fps = fps

        # Setup Matplotlib figure and axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.x)
        self.ax.set_ylim(0, self.y)
        self.ax.set_zlim(0, self.z)
        
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, frames=self.timesteps, repeat=False)
        
        # Save the video
        self.save_video()

    def pool_data(self, data):
        """ Pools the data to fit into an 8x8x8 grid. """
        pooled_data = []
        for frame in data:
            pooled_frame = self.pool_3d(frame, 8, 8, 8)
            pooled_data.append(pooled_frame)
        return np.array(pooled_data)

    def pool_3d(self, data, pool_z, pool_y, pool_x):
        """ Max pool the 3D data to the specified shape. """
        z, y, x = data.shape
        stride_z = z // pool_z
        stride_y = y // pool_y
        stride_x = x // pool_x
        pooled_data = np.zeros((pool_z, pool_y, pool_x))
        for i in range(pool_z):
            for j in range(pool_y):
                for k in range(pool_x):
                    pooled_data[i, j, k] = np.average(data[i*stride_z:(i+1)*stride_z, j*stride_y:(j+1)*stride_y, k*stride_x:(k+1)*stride_x])
        return pooled_data

    def update(self, frame):
        """ Update the frame for the animation. """
        self.ax.clear()
        self.ax.set_xlim(0, self.x)
        self.ax.set_ylim(0, self.y)
        self.ax.set_zlim(0, self.z)
        data = self.data[frame]
        max_val = np.max(data)
        min_val = np.min(data)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        # Update colors of the bars based on current tensor data
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    color_val = data[k, j, i]
                    normalized_color_val = norm(color_val)
                    rgba_color = (normalized_color_val, 0, 1 - normalized_color_val, max(0.1, normalized_color_val))
                    self.ax.bar3d(i, j, k, 1, 1, 1, color=rgba_color, alpha=normalized_color_val)
        print(f"Frame {frame} Done")

    def save_video(self):
        """ Save the animation as a video. """
        self.ani.save(self.output_file, writer='ffmpeg', fps=self.fps)
        print(f"Video saved as {self.output_file}")

# Example usage
# Generate random data for testing
# tensor_data = [torch.rand(1, 64, 64, 64) for _ in range(60)]
# Save video
# video = VideoAnimationTensor(phis)

def save_video_from_phi_list(phi_list, file_location):
    # simple black and white for now
    frame_list_np = []
    for frame in phi_list:
        frame_list_np.append(frame[[2,1,0]].transpose(0,1).transpose(1,2).detach().cpu().numpy())
    writer = cv2.VideoWriter(file_location, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_list_np[0].shape[0], frame_list_np[1].shape[0]), True)
    for frame in frame_list_np:
        writer.write((frame*255).astype('uint8'))
    writer.release()












# Example usage
# a = save_video_from_phi_list(create_phi_batch_list(phis)[0], "dog.mp4")
