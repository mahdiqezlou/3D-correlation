from parallel_corr import more_corr
from mpi4py import MPI
import time
import numpy as np


def run_parallel(spec, xdim, ydim, zdim, sm_l, outfile, boxsize)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size() 

    tss = time.asctime()
    print('Rank =', rank, 'started!', tss, flush=True)

    # rand_seed for each rank
    rand_seed = np.loadtxt('rand_seed.gz').astype(int)[0]

    t = int(np.size(rand_seed)/size)
    rand_seed_rank = rand_seed[rank*t: rank*t + t]



    more_corr(spec= spec, rand_seed_rank=rand_seed_rank, rand_seed=rand_seed,comm=comm, MPI=MPI, xdim= xdim, ydim=ydim, zdim=zdim, sm_l=sm_l, output_file=outfile, box_size= boxsize)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=str, default="spectra.hdf5", help='look at corr.py for description', required=True)
    parser.add_argument('--sepsize', type=list, default=[30., 30., 30.], help='the same as  sep_dist_size in corr.py', required=False)
    parser.add_argument('--sml', type=float, default=1, help='smoothing scale in h^-1 cMpc', required=True)
    parser.add_argument('--boxsize', type=list , default=[60., 60., 60.], help='box size in h^-1 cMpc', required = True)
    parser.add_argument('--outfile', type=str, default="corr.dat", help='Type of reionization history', required=True, choices=["linear", "quasar"])
    args = parser.parse_args()


    run_parallel(spec = args.spec, xdim=, z_f= args.z_f, Emax=args.Emax, alpha_q = args.alphaq, clumping_fac = args.cf)
