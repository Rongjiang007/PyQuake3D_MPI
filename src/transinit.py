import numpy as np
import readmsh


def findvtk(fname,tar):
    f=open(fname,'r')
    info=False
    for line in f:
        #print(line)
        if(info==True and line!='LOOKUP_TABLE default\n'):
            tem=np.array(line.split()).astype(float)
            #print(tem)
            tarvalue=tem
            info=False
        if(line==tar):
            print('!!!!!!!!!!!!!!!',line)
            info=True
    return tarvalue
        


fnamegeo='examples/bp5t/bp5t.msh'
nodelst,elelst=readmsh.read_mshV2(fnamegeo)

a=findvtk('out/step250.vtk','SCALARS a float\n')
b=findvtk('out/step250.vtk','SCALARS b float\n')
dc=findvtk('out/step250.vtk','SCALARS dc float\n')
rake=findvtk('out/step250.vtk','SCALARS rake[Degree] float\n')
rake=rake*np.pi/180.0
fric=findvtk('out/step250.vtk','SCALARS fric float\n')
state=findvtk('out/step250.vtk','SCALARS state float\n')
Tn=findvtk('out/step250.vtk','CELL_DATA 9250 SCALARS Normal_[MPa] float\n')


Tt=np.load('examples/bp5t/Tt/Tt_250.npy')
slipv=np.load('examples/bp5t/slipv/slipv_250.npy')
slip=np.load('examples/bp5t/slip/slip_250.npy')

print(np.max(slip[-1]))

f=open('bp5tparam.dat','w')
for i in range(len(Tt[0])):
    #print(fric[i],Tt[i])
    #f.write('%f %f %f %f %f %f 0 0' %(rake[i],a[i],b[i],dc[i],fric[i],Tt[i]))
    f.write('%f %f %f %f %f %f %f %f %.30f %f 0 0\n' %(rake[i],a[i],b[i],dc[i],fric[i],state[i],Tt[-1,i],Tn[i],slipv[-1,i],slip[-1,i]))
f.close()
