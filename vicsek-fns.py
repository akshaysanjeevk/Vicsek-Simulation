import numpy as np
import matplotlib.pyplot as plt

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



def VicsekUpdation(N, r_old, th_old,  r, v, dt):
    x_old, y_old = r_old[:, 0], r_old[:, 1]
    r_new = r_old + v*dt
    x_new = x_old + np.cos(th_old)*dt
    y_new = y_old + np.cos(th_old)*dt
    r_new = np.array((x_new,y_new)).T
    for i in range(N):
        neighbours = np.where((r_old > r_old[2]-r) &   (r_old < r_old[2]+r))
        th_new = np.mean(th_old[neighbours[0]])+np.random.normal(0, .01)
    
    return r_new, th_new




def PhaseMatrixVideo(phase_arr, theta0, r, time, cmap, fname):
    cmap = firefly
    norm = plt.Normalize(0, 2 * np.pi)
    colors = [cmap(norm(phase)) for phase in theta0]
    N = phase_arr.shape[0]
    grid_size = np.sqrt(N)
    circle_radius = 0.4
    circle_positions = [(i % grid_size, i // grid_size) for i in range(N)]

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('black')


    circles = [plt.Circle((pos[0], pos[1]), 
                          circle_radius, 
                          color=cmap(norm(theta0[i])), lw=2) 
               for i, pos in enumerate(circle_positions)]
    for circle in circles:
        ax.add_artist(circle)

    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')

    def init():
        for circle in circles:

            circle.set_color(cmap(norm(0)))
        return circles

    def animate(i):
        phases = phase_arr[i]
        ax.set_title(f'Kuramoto Oscillators \n $r={ round(r[i], 3)}, t={round(time[i],1)}$', color = 'white', fontsize=24)
        for j, circle in enumerate(circle):
            color = cmap(norm(phases[j]))
            circle.set_color(color)
        return circles

    anim = FuncAnimation(fig, animate , init_func=init, frames=T, interval=100, blit=True)
    
    anim.save(f'{fname}.mp4', writer='ffmpeg', fps=15, dpi=450, bitrate=2000)       