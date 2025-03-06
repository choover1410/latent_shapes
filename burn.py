import numpy as np
import pylab as plt
import skfmm


def burn_grain(X, Y, phi, N, dx):
    phi_init = phi[::-1, :]
    d = skfmm.distance(phi, dx=dx)

    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(14, 4)
    axs[0].imshow(phi_init, cmap='copper')
    axs[0].grid()
    CS = axs[1].contour(X, Y, d, N)

    all_line_ints = []
    for i in range(0, len(CS.allsegs)-1):
        line_int = 0
        for polygon in range(len(CS.allsegs[i])):
            dat= CS.allsegs[i][polygon]
            for pnt in range(len(dat[:,0])):
                if pnt > 0:
                    x2 = (dat[pnt,0] - dat[pnt-1,0])**2
                    y2 = (dat[pnt,1] - dat[pnt-1,1])**2
                    dist = np.sqrt(x2 + y2)
                    line_int += dist
        all_line_ints.append(line_int)
    all_line_ints.append(0)
    x_ints = [i for i in range(len(all_line_ints))]
    max_cont = x_ints[-1]
    x_ints_norm = [i/max_cont for i in x_ints]
    axs[2].plot(x_ints_norm, all_line_ints)
    plt.show()

    return all_line_ints, d

