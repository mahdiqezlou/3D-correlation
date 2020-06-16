import h5py
import numpy as np
import corr
from latis import Latis
from mock_latis import Mock_latis

def more_corr(spec_obj, rand_seed_rank , rand_seed, comm, MPI, box_size =120  ,xdim= 30,ydim=30, zdim=30, sm_l=1.5,file_name = 'As_2.24'):
    """ Calling for more pairs since memory limit does not let to draw too many pairs at once """

    #m = Mock_latis(spec_file='./spectra/pix_'+file_name+'.dat')
    m = spec_obj

    zdim_orig = zdim
    zdim = np.around(zdim_orig/sm_l)
    tdim = np.around(np.sqrt(xdim**2 + ydim**2)/sm_l)
    corr_numerator = np.zeros(shape=(1+int(tdim), 1+int(zdim)))
    weights = np.zeros(shape=(1+int(tdim), 1+int(zdim)))
    stat_pairs = np.zeros(shape=(1+int(tdim), 1+int(zdim)))
    # call corr() for a single rand seed and add the results to the containers defined above
    for i in rand_seed_rank :
        one_run = corr.corr_spec(m.spec, xdim=xdim, ydim=ydim, zdim=zdim_orig, sm_l=sm_l,num_pairs=10**6, rand_seed = i,savefile=None, box_size=box_size)
        corr_numerator += one_run[0]
        weights += one_run[1]
        stat_pairs += one_run[2]
    # make sure array elements are contiguous
    corr_numerator = np.ascontiguousarray(corr_numerator, type(corr_numerator[1,1]))
    weights = np.ascontiguousarray(weights, type(weights[1,1]))
    stat_pairs = np.ascontiguousarray(stat_pairs, type(stat_pairs[1,1]))
    # variables for comm.Reduce()
    corr_numerator_total = np.zeros_like(corr_numerator)
    weights_total = np.zeros_like(weights)
    stat_pairs_total = np.zeros_like(stat_pairs)
    ### Call manager rank here
    comm.Reduce(corr_numerator, corr_numerator_total, op=MPI.SUM, root=0)
    comm.Reduce(weights, weights_total, op=MPI.SUM, root = 0)
    comm.Reduce(stat_pairs, stat_pairs_total, root=0)
    rank = comm.Get_rank()
    if rank ==0 :
        write_on_file(corr_numerator=corr_numerator_total, weights=weights_total, stat_pairs=stat_pairs_total, sm_l=sm_l, dim=np.array([xdim, ydim, zdim]), rand_seed=rand_seed, file_name=file_name)


def write_on_file(corr_numerator, weights, stat_pairs, sm_l, dim, rand_seed, file_name='As_1.29'):
    """ Write the on an hdf5 file """

    f = h5py.File('mock_30Mpc_'+file_name+'.hdf5','w')
    f['dimenstions'] = dim
    f['smoothing_length'] = sm_l
    f['rand_seed'] = rand_seed
    f['corr_numerator'] = corr_numerator
    f['weights'] = weights
    f['stat_pairs'] = stat_pairs
    f.close()





def add_corr(file_name):

    R = [np.arange(0,300, 3), np.arange(303,704, 4), np.arange(706,1113, 5)]
    rand_seed = np.append(R[0],R[1])
    rand_seed = np.append(rand_seed, R[2])
    f = h5py.File('mock_30Mpc_'+file_name+'.hdf5','w')
    ft = h5py.File('./dir_corr/mock/9mock.hdf5','r')
    corr_numerator = np.zeros_like(ft['corr_numerator'])
    weights = np.zeros_like(ft['weights'])
    stat_pairs = np.zeros_like(ft['stat_pairs'])
    f['dimenstions'] = np.array([30., 30., 30.])
    f['smoothing_length'] = 2
    f['rand_seed'] = rand_seed 
    ft.close()

    for i in rand_seed :
        ft = h5py.File('./dir_corr/mock/'+str(i)+'mock.hdf5','r')
        corr_numerator +=ft['corr_numerator'][:]
        weights += ft['weights']
        stat_pairs += ft['stat_pairs']

    f['corr_numerator'] = corr_numerator
    f['weights'] = weights
    f['stat_pairs'] = stat_pairs

    ft.close()
    f.close()


