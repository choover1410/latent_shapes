import numpy as np
import pylab as plt
import skfmm
import matplotlib.patches as patches


def burn_grain(X, Y, phi, N, dx):
    phi_init = phi[::-1, :]
    d = skfmm.distance(phi, dx=dx)

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10, 4)
    CS = axs[0].contour(X, Y, d, N)
    axs[0].set_aspect('equal')

    circle_radius = 1.0
    circle_center_x = 0.0
    circle_center_y = 0.0

    circle = patches.Circle((circle_center_x, circle_center_y), circle_radius, 
                    edgecolor='black', facecolor='none', linewidth=4)
    axs[0].add_artist(circle)

    all_line_ints = []
    for i in range(0, len(CS.allsegs)-1):
        line_int = 0
        for polygon in range(len(CS.allsegs[i])):
            dat= CS.allsegs[i][polygon]
            for pnt in range(len(dat[:,0])):
                radius = np.sqrt(dat[pnt,0]**2 + dat[pnt,1]**2)
                if radius > 1.0:
                    continue
                if pnt > 0:
                    x2 = (dat[pnt,0] - dat[pnt-1,0])**2
                    y2 = (dat[pnt,1] - dat[pnt-1,1])**2
                    dist = np.sqrt(x2 + y2)
                    line_int += dist
        all_line_ints.append(line_int)

    final_line_ints = [i / (2*np.pi) for i in all_line_ints if i>0]
    final_line_ints.append(0)
    x_ints = [i for i in range(len(final_line_ints))]
    max_cont = x_ints[-1]
    x_ints_norm = [i/max_cont for i in x_ints]
    axs[1].plot(x_ints_norm, final_line_ints)

    #########################
    # Comment to view results
    #plt.show()
    plt.close(fig)
    #########################

    return final_line_ints, phi

