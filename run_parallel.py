from parallel_corr import more_corr
from mpi4py import MPI
import time
import numpy as np
from latis import Latis
from mock_latis import Mock_latis
#latis = Latis()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 

tss = time.asctime()
print('Rank =', rank, 'started!', tss, flush=True)

# random seeds for pairs selection processes in corr.py
#R = [np.arange(0,300, 3), np.arange(303,704, 4), np.arange(706,1113, 5), np.arange(1114, 2114, 3), np.arange(2116, 3200, 7), np.arange(3211, 4300, 4)]
rand_seed = np.loadtxt('rand_seed.gz').astype(int)[0]
# rand seeds for each rank
t = int(np.size(rand_seed)/size)
rand_seed_rank = rand_seed[rank*t: rank*t + t]

file_name = 'LCDM_n10'
m = Mock_latis(spec_file='./spectra/pix_'+file_name+'.dat')
#file_name = file_name + 'e9_pairs'
spec_obj=m

#file_name = 'LATIS_e9_pairs'
#spec_obj = latis
more_corr(spec_obj=spec_obj, rand_seed_rank=rand_seed_rank, rand_seed=rand_seed,comm=comm, MPI=MPI, xdim= 30, ydim=30, zdim=30, sm_l=1.5, file_name=file_name, box_size=[63,51,2415])

