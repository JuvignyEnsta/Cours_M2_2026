# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


globCom = MPI.COMM_WORLD.Dup()
rank    = globCom.rank
nbp     = globCom.size

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=2.)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

nb_rows_per_task = 1
def task( irow : int ):
    convergence = np.empty((nb_rows_per_task, width), dtype=np.double)
    # Calcul de l'ensemble de mandelbrot :
    for ir in range(nb_rows_per_task):
        y = ir + irow
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            convergence[ir,x] = mandelbrot_set.convergence(c, smooth=True)
    return convergence

deb = time()
if rank==0: # Maître
    convergence = np.empty((height, width), dtype=np.double)
    iTask : int = 0
    buffer = np.empty((nb_rows_per_task,width), dtype=np.double)      
    for i in range(1,nbp):
        globCom.send(iTask, dest=i)
        iTask += nb_rows_per_task
    status = MPI.Status()
    while iTask < height:
        globCom.Recv(buffer,status=status)
        sender = status.Get_source()
        irow   = status.Get_tag()
        req = globCom.isend(iTask, dest=sender)
        iTask += nb_rows_per_task
        convergence[irow:irow+nb_rows_per_task,:] = buffer[:,:]
        req.wait()
    # Finalisation des tâches :
    iTask = -1
    for p in range(1,nbp):
        globCom.Recv(buffer,status=status)
        sender = status.Get_source()
        irow   = status.Get_tag()
        req = globCom.isend(iTask, dest=sender)
        convergence[irow:irow+nb_rows_per_task,:] = buffer[:,:]
        req.wait()
else:
    iTask = 0
    while iTask >=0:
        iTask = globCom.recv(source=0)
        if iTask >=0:
            res = task(iTask)
            globCom.Send(res, 0, iTask)
fin = time() 
print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")

if rank==0:
    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()
