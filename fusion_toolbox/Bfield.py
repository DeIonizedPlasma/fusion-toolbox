import numpy as np
import matplotlib.pyplot as plt

class Coil:
    def __init__(self,pts,I):
        """
        Coil Class.

        Parameters
        ----------
        pts : np.ndarray of shape (N,3)
            List of x,y,z triplets which define a closed current loop. The last entry should be
            equal to the first.
        I : float
            Coil current, in Amps
        """
        if not np.allclose(pts[0],pts[-1]): #If loop is not properly closed, connect first/last pts
            pts = np.append(pts,pts[0][None,:],axis=0)
        self.pts = pts
        self.I = I

    def B(self,xyz):
        """
        Returns the magnetic field 

        Parameters
        ----------
        pts : np.ndarray with last dimension of size 3

        Returns
        -------
        np.ndarray of 3D vectors at each point of xyz. Shape is (*xyz.shape,3)
        """
        B_out = np.zeros((len(xyz),3)) #Return array of B vectors in same shape as input

        drs = self.pts[1:]-self.pts[:-1]

        for i,pt in enumerate(self.pts[:-1]): #Skip last point because it repeats
            B_out += self.BGreen(xyz,pt,drs[i])

        return B_out*self.I

    def BGreen(self, xyz_samples, xyz_center, dl):
        """Evaluates the B field at all sample locations due to a wire segment. Uses the formula:
        dB = mu_0/4pi * I * dl x (r-r_0)/|r-r_0|^3
        The current I 

        Parameters
        ----------
        xyz_samples : np.ndarray of shape (N,3)
            Locations to evaluate B at

        xyz_center : np.ndarray of shape (3)
            Location of wire segment

        dl : np.ndarray of shape (3)
            Vector for wire segment. Should have magnitude equal to length of wire segment.

        """
        _r = xyz_samples - xyz_center[None,:]
        Bvecs = np.cross(dl,_r)/np.linalg.norm(_r,axis=1)[:,None]**3
        return 1e-7 * Bvecs #mu_0/4pi = 1e-7 H/m

    def plot_contour(self):
        """
        Displays a 3D plot of the coil
        """
        ax = plt.figure().add_subplot(projection='3d')
        x,y,z = self.pts.T
        ax.plot(x,y,z)
        bbox_val = np.max( (np.max(np.abs(self.pts.T[0])),
                            np.max(np.abs(self.pts.T[1])),
                            np.max(np.abs(self.pts.T[2])))
                         )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-bbox_val,bbox_val)
        ax.set_ylim(-bbox_val,bbox_val)
        ax.set_zlim(-bbox_val,bbox_val)
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

        plt.show()
    def plot_B_slice(self,axis,r,w1,w2,n1,n2):
        ax = plt.figure().add_subplot()
        #ax = plt.figure().add_subplot(projection='3d')
        X,Y,Z = gen_plane_pts(axis,r,w1,w2,n1,n2)
        if axis==0:
            C1,C2 = Y,Z
        elif axis==1:
            C1,C2 = X,Z
        elif axis==2:
            C1,C2 = X,Y
        Bpts = np.vstack((X.ravel(),Y.ravel(),Z.ravel())).T
        B_samples = self.B(Bpts).T
        B_clean = np.delete(B_samples,axis,axis=0).T
        B_2D = np.zeros((n2,n1,2))
        for i,B in enumerate(B_clean):
            B_2D[(i//n1)%n2,i%n1] = B
        ax.streamplot(C1,C2,B_2D[:,:,0],B_2D[:,:,1],density=1.5)
        ax.set_aspect(1)
        plt.show()

class PFCoil(Coil):
    def __init__(self,R,Z0,I,npts=100):
        thetas = np.linspace(0,2*np.pi,npts)
        X = R*np.cos(thetas)
        Y = R*np.sin(thetas)
        Z = Z0*np.ones(npts)
        self.pts = np.vstack((X,Y,Z)).T
        self.I = I

class TFCoil(Coil):
    def __init__(self,R,r,phi,I,npts=100):
        thetas = np.linspace(0,2*np.pi,npts)
        X = np.cos(phi)*(R+r*np.cos(thetas))
        Y = np.sin(phi)*(R+r*np.cos(thetas))
        Z = r*np.sin(thetas)
        self.pts = np.vstack((X,Y,Z)).T
        self.I = I

class Tokamak:
    def __init__(self,R,a,coils=[]):
        """
        Tokamak class.

        Parameters
        ----------
        R : float
            Major radius of tokamak
        a : float
            Minor radius of tokamak
        coils : list of Coil objects
            
        """
        self.coils = []
        self.R = R
        self.a = a

    def get_B_from_coils(self,pts):
        """
        Returns the total magnetic field due to all coils in the tokamak object.
        """
        tok_B = np.zeros_like(pts)
        for coil in self.coils:
            tok_B += coil.B(pts)
        return tok_B
    
    def make_PFset(self,R,Z,I):
        """
        Adds in a set of PF coils given a list of their R/Z coordinates and currents.
        
        Parameters
        ----------
        R : np.ndarray of shape (N)
            Vector of R positions for N coils
        Z : np.ndarray of shape (N)
            Vector of Z positions for N coils
        I : np.ndarray of shape (N)
            Vector of currents I for N coils
        """
        #Still under construction
        N = len(R)
        for i in range(N):
            self.coils.append(PFCoil(R[i],Z[i],I[i]))
        

    def make_TFset(self,phi,I):
        """
        Adds in a set of TF coils given a list of their toroidal angle phi and currents.
        
        Parameters
        ----------
        phi : np.ndarray of shape (N)
            Vector of toroidal angles for N coils
        I : np.ndarray of shape (N)
            Vector of currents I for N coils
        """
        #Still under construction
        N = len(phi)
        for i in range(N):
            self.coils.append(TFCoil(self.R,self.a,phi[i],I[i]))

'''
thetas = np.linspace(0,2*np.pi,100)
X = 1*np.cos(thetas)
Y = 1*np.sin(thetas)
Z = 0*X
XYZ = np.vstack((X,Y,Z)).T

a = Coil(XYZ,1)
field = a.B(np.array([[0,0,0],[0,1,2]]))
print(field)
'''

a = PFCoil(1,0,1)
field = a.B(np.array([[0,0,0],[0,1,2]]))
print(field)

#CMod = Tokamak(4,1)

#coils = CMod.make_PFset(4,1,1)
#coils = CMod.make_PFset(4,-1,1)
def gen_plane_pts(axis,r,w1,w2,n1,n2):
    """
    Generates a 2D grid of points on a plane in 3D which is normal to one of the three axes.

    Parameters
    ----------
    axis : int
        Which axis the plane should be normal to. Valid values are 0, 1, or 2 for x, y, or z.
    r : np.ndarray of shape (3)
        Displacement from origin of the center of the plane
    w1 : float
        Physical width of first dimension of grid
    w2 : float
        Physical width of second dimension of grid
    n1 : int
        Number of grid points along side 1 of the grid.
    n2 : int
        Number of grid points along side 2 of the grid.
    """
    if axis==0:
        yspan = np.linspace(r[1]-w1/2,r[1]+w1/2,n1)
        zspan = np.linspace(r[2]-w2/2,r[2]+w2/2,n2)
        Y,Z = np.meshgrid(yspan,zspan)
        X = np.zeros_like(Y)
    elif axis==1:
        xspan = np.linspace(r[1]-w1/2,r[1]+w1/2,n1)
        zspan = np.linspace(r[2]-w2/2,r[2]+w2/2,n2)
        X,Z = np.meshgrid(xspan,zspan)
        Y = np.zeros_like(Z)
    elif axis==2:
        xspan = np.linspace(r[1]-w1/2,r[1]+w1/2,n1)
        yspan = np.linspace(r[2]-w2/2,r[2]+w2/2,n2)
        X,Y = np.meshgrid(xspan,yspan)
        Z = np.zeros_like(X)
    return X,Y,Z
