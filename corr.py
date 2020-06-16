## For calculating the 3D correlation function (Slosar 2011)
import numpy as np
import h5py
from itertools import combinations
import array 


def corr_spec(spec, sep_dist_size , sm_l, num_pairs = 10**4, rand_seed=13,savedir = './',savefile= None, box_size = [63, 51, 2415]):
    """Calculates the 3D correlation function using the pixels along spectra
    implementing the equation 4.3 slosar et al. 2011 arXiv:1104.5244

    Argumentss :
    spec = an array with shape (number of spectra*size of each spectra= total number of pixels  , 5). 
           The second dimention contains (x, y, z, deltaF, sigma_deltaF) for each pixel.
           
    sep_dist_size : a list with size of 3 containing the maximum size of x, y, z components of the 3D separation distance. 
                  So, they could be from 0 to the box size
    sm_l : The distance between pixels will be coarse with that size. 
    num_pairs = total number of pairs you want to take the summation over
    savefile : Pass a file name to get the output in an hdf5 file
    box_size : a list of size 3 containing the x, y, z length of the simulation/obseration box

    returns :

    if savefile = None : it returns a tuple containing (the numerator of the equation,  weights for each term in the sum, counts for
    each term in the sum)

    Otherwise saves the same info on an hdf5 file

    """
    [xdim, ydim, zdim] = [sep_dist_size[0], sep_dist_size[1], sep_dist_size[2] ]
    ## Binning the separation distance space
    zdim = np.around(zdim/sm_l)
    tdim = np.around(np.sqrt(xdim**2 + ydim**2)/sm_l)

    # A 3D array containing value of correlattion function in the x,y,z bins.
    corr_numerator = np.zeros(shape=(1+int(tdim), 1+int(zdim)))
    weights = np.zeros(shape=(1+int(tdim), 1+int(zdim)))
    # An array to store numebr of pairs contributed to that point in seperation space
    stat_pairs = np.zeros(shape=(1+int(tdim), 1+int(zdim)))

    first_attempt = True
    found_pairs = 0

    np.random.seed(rand_seed)

    while found_pairs < num_pairs:
        L_Box=[0,0,0]
        new_pairs = np.random.randint(0, np.shape(spec)[0], size=(10**5,2))
        [L_Box[0], L_Box[1], L_Box[2]] = [box_size[0]*np.ones(shape=(10**5,)), box_size[1]*np.ones(shape=(10**5,)),box_size[2]*np.ones(shape=(10**5,))]
        xp = np.abs(spec[new_pairs[:,0], 0] - spec[new_pairs[:,1], 0])
        # The simulation box is periodic, take the minimum separation coordinates
        xp = np.minimum(xp, L_Box[0] - xp)
        yp = np.abs(spec[new_pairs[:,0], 1] - spec[new_pairs[:,1], 1])
        yp = np.minimum(yp, L_Box[1] - yp)
        zp = np.abs(spec[new_pairs[:,0], 2] - spec[new_pairs[:,1], 2])
        zp = np.minimum(zp, L_Box[2] - zp)
        tp = np.around(np.sqrt(xp*xp +yp*yp)/sm_l)
        zp = np.around(zp/sm_l)

        ## delte the pairs with separartion larger than desired limits
        ind = np.where(tp > tdim)[0]
        [tp, zp, new_pairs] = [np.delete(tp,ind), np.delete(zp,ind), np.delete(new_pairs, ind, axis=0)]
        ind = np.where(zp > zdim)[0]
        [tp, zp, new_pairs] = [np.delete(tp,ind), np.delete(zp,ind), np.delete(new_pairs, ind, axis=0)]

        if first_attempt :
            [t, z, pairs] = [tp, zp, new_pairs]
            found_pairs = np.shape(pairs)[0]
            first_attempt = False
        else :
            [t, z, pairs] = [np.append(t, tp), np.append(z, zp), np.append(pairs, new_pairs, axis=0)]
            found_pairs = np.shape(pairs)[0]

    
    # intrinsic fluctuation of IGM, from the appendix in LATIS paper 0.19. It contributes to the variance in each pixel
    sigma_IGM = 0.19
    spec[:,3] = np.sqrt(spec[:,3]**2 + sigma_IGM**2)

    for i in range(np.shape(pairs)[0]):
        # Total inverse variance
        w = (1/spec[pairs[i,0], 3]**2)*(1/spec[pairs[i,1], 3]**2)
        stat_pairs[int(t[i]), int(z[i])] += 1
        weights[int(t[i]),int(z[i])] += w
        corr_numerator[int(t[i]),int(z[i])] += w*spec[pairs[i,0], 4]*spec[pairs[i,1], 4]
    
    ## avoid devision by zero. Find the unexplored points in correlation function domain
    ind = np.where(stat_pairs == 0)
    weights[ind[0], ind[1]] = 1
    stat_pairs[ind[0], ind[1]] = 1

    if savefile is not None :
        f = h5py.File(savedir+str(rand_seed)+savefile, 'w')
        # nummerator of the correlation equation
        f['corr_numerator'] = corr_numerator
        # weights on each gird of separation distance space
        f['weights'] = weights
        # number of counts in each grid
        f['stat_pairs'] = stat_pair
        f['smothing_length'] = sm_l
        # dimentions of the separation distance space
        f['dimenstions'] = np.array([xdim, ydim, zdim])
        # rand seed being used
        f['rand_seed'] = rand_seed
        f.close()

    else :
        print('Run = ', rand_seed, 'finished !!!')
        # the first term is numerator of correlation
        return (corr_numerator, weights, stat_pairs)








def find_neighbors(x_0,y_0,z_0,r, z_bound,delta_r=1):
    """ It is not being used here. Has been written for other purposes.
    find all nearby voxels to (x_0,y_0,z_0) at a radius of between r-delta_r and r+delta_r 

    returns a  list of neighbor's coordinates
    """
    # store all neighbor's coordinates points[i][0] is the x coordinate
    # of neighbore i and ...
    points = []
    # I constrain coordinates one by one
    # NEEDS TO GET MORE EFFICIENT
    x = np.arange(-(r+delta_r), r+delta_r+0.1, 1)
    x = x[ np.where((x+x_0 <= 59)*(x+x_0>=34))[0]]
    for x_s in x:
        s = np.sqrt((r+delta_r)**2 - x_s**2)
        y = np.arange(-1.0*s, s+0.1, 1)
        y = y[np.where((y+y_0<=47)*(y+y_0>=28))[0]]

        for y_s in y:
            z=np.array([])
            bound_z_1 = (r-delta_r)**2 - x_s**2 - y_s**2
            bound_z_2 = (r+delta_r)**2 - x_s**2 - y_s**2

            if bound_z_2 <= 0 :
                bound_z_1 = 0
                bound_z_2 = 0
            if bound_z_1 < 0:
                bound_z_1 = 0
            
            z_p = np.arange(np.sqrt(bound_z_1), np.sqrt(bound_z_2)+0.1, 1)
            z = np.append(z, z_p)
            # if z is 0, we should not duplicate it in z
            if np.any(z_p) != 0:
                z_p = -1.0*z_p
                z = np.append(z, z_p)
            
            z = z[np.where((z+z_0 <= z_bound[1])*(z+z_0 >= z_bound[0]))[0]]
            for z_s in z:
                points.append([int(np.around(x_s+x_0)),int(np.around(y_s+y_0)),int(np.around(z_s+z_0))])


    return np.array(points)

            


        


    

