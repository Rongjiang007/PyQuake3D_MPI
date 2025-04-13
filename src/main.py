import readmsh
import numpy as np
import sys
import matplotlib.pyplot as plt
import QDsim
from math import *
import time
import argparse
import os
import psutil
from datetime import datetime
from mpi4py import MPI

file_name = sys.argv[0]
print(file_name)

from mpi_config import comm, rank, size



if __name__ == "__main__":
    #eleVec,xg=None,None
    #nodelst,xg=None,None
    sim0=None
    jud_coredir=None
    blocks_to_process=[]
    if(rank==0):
        try:
            #start_time = time.time()
            parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
            parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
            parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

            args = parser.parse_args()

            fnamegeo = args.inputgeo
            fnamePara = args.inputpara
        
        except:
            #fnamegeo='examples/Heto/planar10W.msh'
            #fnamePara='examples/Heto/parameter.txt'
            #fnamegeo='examples/bp5t/bp5t.msh'
            #fnamePara='examples/bp5t/parameter.txt'
            #fnamegeo='examples/turkey/turkey_cut.msh'
            #fnamePara='examples/turkey/parameter.txt'
            #fnamegeo='examples/WMF/WMF3.msh'
            #fnamePara='examples/WMF/parameter.txt'
            fnamegeo='examples/cascadia/cascadia35km_ele4.msh'
            fnamePara='examples/cascadia/parameter.txt'
        print('Input msh geometry file:',fnamegeo, flush=True)
        print('Input parameter file:',fnamePara, flush=True)
        
        
        f=open('state.txt','w')
        f1=open('curve.txt','w')
        f.write('Program start time: %s\n'%str(datetime.now()))
        f.write('Input msh geometry file:%s\n'%fnamegeo)
        f.write('Input parameter file:%s\n'%fnamePara)
        
        #fname='bp5t.msh'
        
        nodelst,elelst=readmsh.read_mshV2(fnamegeo)
        print('Number of Node',nodelst.shape, flush=True)
        print('Number of Element',elelst.shape, flush=True)

        sim0=QDsim.QDsim(elelst,nodelst,fnamePara)
        jud_coredir,blocks_to_process=sim0.get_block_core(comm, rank, size)

        # x=np.ones(len(elelst))
        # y=sim0.tree_block.blocks_process_MVM(x,blocks_to_process,'A1s')
        # k1=1714
        # print(jud_coredir,np.min(y),k1,y[k1-10:k1+100])

        fname='test.vtk'
        sim0.ouputVTK(fname)



    print('rank:',rank)
    #xg = comm.bcast(xg, root=0)
    #nodelst = comm.bcast(nodelst, root=0)
    #elelst = comm.bcast(elelst, root=0)
    sim0 = comm.bcast(sim0, root=0)
    jud_coredir = comm.bcast(jud_coredir, root=0)

    #print(jud_coredir)
    #print(sim0.tree_block,sim0.tree_block.thread_svd)
    start_time = MPI.Wtime()
    #if(rank==0):
        #sim0.tree_block.parallel_traverse_SVD(comm, rank, size)
    
    if(jud_coredir==False):
        #sim0.local_blocks=sim0.tree_block.parallel_traverse_SVD(sim0.Para0['Corefunc directory'],plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
        if(rank==0):
            sim0.tree_block.master(sim0.Para0['Corefunc directory'],blocks_to_process,size-1)
        else:
            sim0.tree_block.worker()
        sim0.local_blocks=sim0.tree_block.parallel_block_scatter_send(sim0.tree_block.blocks_to_process,plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
    else:
        sim0.local_blocks=sim0.tree_block.parallel_block_scatter_send(blocks_to_process,plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
        
        #local_index=sim0.tree_block.parallel_vector_scatter(comm,rank,size,sim0.xg)
        #sim0.test1()
        #print('local_blocks',rank)

        #if(rank==0):
    
    if(rank==0):
        SLIP=[]
        SLIPV=[]
        Tt=[]
        f=open('state.txt','w')
        f.write('Program start time: %s\n'%str(datetime.now()))
        f.write('Input msh geometry file:%s\n'%fnamegeo)
        f.write('Input parameter file:%s\n'%fnamePara)
        f.write('iteration time_step(s) maximum_slip_rate(m/s) time(s) time(h) Relerrormax1 Relerrormax2\n')


    totaloutputsteps=int(sim0.Para0['totaloutputsteps'])
    for i in range(totaloutputsteps):

        if(i==0):
            dttry=sim0.htry
        else:
            dttry=dtnext
        dttry,dtnext=sim0.simu_forward(dttry)
        #sim0.simu_forward(dttry)
        if(rank==0):
            year=sim0.time/3600/24/365
            if(i%20==0):
                print('iteration:',i, flush=True)
                print('dt:',dttry,' max_vel:',np.max(np.abs(sim0.slipv)),' Seconds:',sim0.time,'  Days:',sim0.time/3600/24,
                'year',year, flush=True)
            f.write('%d %f %f %.16e %f %f\n' %(i,dttry,np.max(np.abs(sim0.slipv1)),np.max(np.abs(sim0.slipv2)),sim0.time,sim0.time/3600.0/24.0))
            
            #f1.write('%d %f %f %f %.6e %.16e\n'%(i,dttry,sim0.time,sim0.time/3600.0/24.0,sim0.Tt[index1_],sim0.slipv[index1_]))
            SLIP.append(sim0.slip)
            SLIPV.append(sim0.slipv)
            Tt.append(sim0.Tt)
            
            # if(sim0.time>60):
            #     break
            outsteps=int(sim0.Para0['outsteps'])
            directory='out'
            if not os.path.exists(directory):
                os.mkdir(directory)
            if(i%outsteps==0):
                SLIP=np.array(SLIP)
                SLIPV=np.array(SLIPV)
                Tt=np.array(Tt)
                np.save('examples/cascadia/slipv/slipv_%d'%i,SLIPV)
                np.save('examples/cascadia/slip/slip_%d'%i,SLIP)
                np.save('examples/cascadia/Tt/Tt_%d'%i,Tt)

                SLIP=[]
                SLIPV=[]
                Tt=[]

                if(sim0.Para0['outputvtk']=='True'):
                    fname=directory+'/step'+str(i)+'.vtk'
                    sim0.ouputVTK(fname)
                if(sim0.Para0['outputmatrix']=='True'):
                    fname='step'+str(i)
                    sim0.outputtxt(fname)
                #if(year>1200 or np.max(np.abs(sim0.slipv))<0.01):
                #if(year>280):
                #    break
            



    end_time = MPI.Wtime()

    if rank == 0:
        
        print(f"Program run time: {end_time - start_time:.6f} sec")
        timetake=end_time - start_time
        f.write('Program end time: %s\n'%str(datetime.now()))
        f.write("Time taken: %.2f seconds\n"%timetake)
        f.close()

        



