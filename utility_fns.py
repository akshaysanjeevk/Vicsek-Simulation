import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  
# from moviepy.editor import VideoFileClip  




def VicsekSimulation(r0, th0, r, v, dt, ts):
    N = r0.shape[0]
    r_series = np.zeros((N, 2, ts))
    th_series = np.zeros((N, ts))
    r_series[:, :, 0] = r0
    th_series[:, 0] = th0
    for j in range(1, ts):
        r_series[:, :, j], th_series[:, j] = VicsekUpdation(
            N,r_series[:, :,j-1],
            th_series[:,j-1],
            r,v,dt
            )
    return r_series, th_series



import numpy as np

def VicsekUpdation(N, r_old, th_old, r, v, dt):
    x_old, y_old = r_old[:, 0], r_old[:, 1]
    x_new = x_old + np.cos(th_old) * v * dt
    y_new = y_old + np.sin(th_old) * v * dt
    x_new %= 1.0
    y_new %= 1.0
    r_new = np.stack((x_new, y_new), axis=1)
    th_new = np.zeros(N)

    for i in range(N):
        # Compute periodic distance
        dx = np.abs(r_old[:, 0] - r_old[i, 0])
        dy = np.abs(r_old[:, 1] - r_old[i, 1])
        dx = np.minimum(dx, 1.0 - dx)  # periodic wrap
        dy = np.minimum(dy, 1.0 - dy)

        # Euclidean distance with periodic boundaries
        dist = np.sqrt(dx**2 + dy**2)

        # Find neighbors within radius r
        neighbors = np.where(dist < r)[0]

        # Average angle of neighbors + noise
        mean_angle = np.arctan2(np.mean(np.sin(th_old[neighbors])),
                                np.mean(np.cos(th_old[neighbors])))
        th_new[i] = mean_angle + np.random.normal(0, 0.01)

    return r_new, th_new



# def generate_particle_video(x_data, y_data, angle_data, output_filename='particles.mp4', fps=30):
#     """
#     Generates a video of particles moving from time series arrays of x, y coordinates, and angles.

#     Parameters:
#     x_data (numpy.ndarray): A 2D array where each row represents the x-coordinates of particles at a time step.
#     y_data (numpy.ndarray): A 2D array where each row represents the y-coordinates of particles at a time step.
#     angle_data (numpy.ndarray): A 2D array where each row represents the angles of the particles at a time step.
#     output_filename (str): The filename for the output video.
#     fps (int): Frames per second for the output video.
#     """
#     num_frames, num_particles = x_data.shape
    
#     fig, ax = plt.subplots()
#     ax.set_xlim(np.min(x_data), np.max(x_data))
#     ax.set_ylim(np.min(y_data), np.max(y_data))
    
#     arrows = [ax.arrow(x_data[0, i], y_data[0, i], np.cos(angle_data[0, i]) * 0.1, np.sin(angle_data[0, i]) * 0.1,
#                         head_width=0.05, head_length=0.1, fc='blue', ec='blue') for i in range(num_particles)]
    
#     def update(frame):
#         for i, arrow in enumerate(arrows):
#             arrow.remove()
#             arrows[i] = ax.arrow(x_data[frame, i], y_data[frame, i], 
#                                  np.cos(angle_data[frame, i]) * 0.1, np.sin(angle_data[frame, i]) * 0.1,
#                                  head_width=0.05, head_length=0.1, fc='blue', ec='blue')
#         return arrows
    
#     ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)
#     ani.save(output_filename, writer='ffmpeg', fps=fps)
#     plt.close(fig)
    
#     print(f"Video saved as {output_filename}")