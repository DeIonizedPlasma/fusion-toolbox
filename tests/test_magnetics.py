import numpy as np
import numpy.testing as npt
from math import pi
from fusion_toolbox.Bfield import *
import pytest

MU_0 = 4*pi*1E-7 # T * m / A

def generate_pf_coil_xyz(R, N = 100, z = 0): 
    '''
    Get xyz coordinates of points along a pf coil 
    R: radius [m]
    N: number of points 
    z: pf coil height [m]
    Returns: xyz - np.ndarray [N,3]
    '''
    thetas =  np.linspace(0, 2*pi, N)
    xyz = np.zeros((N,3))
    xyz[:,0] = R * np.cos(thetas)
    xyz[:,1] = R * np.sin(thetas)
    xyz[:,2] = z
    return xyz

def analytic_B_center_of_pf_coil(I, R): 
    return np.array([0,0,MU_0 * I / (2 * R)])
def analytic_B_axis_of_pf_coil(I,R,z):
    return np.array([0,0,MU_0*2*pi*R**2*I/(4*pi*(z**2+R**2)**(3/2))])

@pytest.mark.parametrize(
    "I, R",
    [[1E3,0.3], [1E5, 0.02], [1E7, 0.004], [1E2, 1]
    ])
def test_B_center_of_pf_circular_coil(I, R): 
    '''
    I: current [A]
    R: coil radius [m]
    Checks the calculated field at the center of a circular coil against the analytic solution
    '''

    # Generate test coil and calculate the field at the center
    xyz = generate_pf_coil_xyz(R, N = int(1E4))
    test_coil = Coil(xyz, I)
    B_center = test_coil.B([np.array([0,0,0])])
    
    # Compare to analytic calculation
    B_analytic = [analytic_B_center_of_pf_coil(I,R)]
    npt.assert_almost_equal(B_center, B_analytic, decimal = 4)

@pytest.mark.parametrize(
    "R1, Z1, I1, R2, Z2, I2",
    [[1,1,1,1,-1,1],[1,1,1,1,-1,-1]
    ])
def test_tokamak_B_total_center(R1,Z1,I1,R2,Z2,I2):
    '''
    R1: Radius of first loop [m]
    Z1: Z offset of first loop [m]
    I1: Current of first loop [A]
    R2: Radius of second loop [m]
    Z2: Z offset of second loop [m]
    I2: Current of second loop [A]
    Checks the field exactly centered between two parallel planar coil loops against analytic sol.
    '''
    #Generate tokamak with test coil pair (major and minor radius are arbitrary)
    tok = Tokamak(1,1)
    tok.make_PFset([R1,R2],[Z1,Z2],[I1,I2])
    B = tok.get_B_from_coils(np.array([[0.,0,0]]))
    B_center_analytic = analytic_B_axis_of_pf_coil(I1,R1,Z1-(Z1+Z2)/2)\
                       +analytic_B_axis_of_pf_coil(I2,R2,-Z1+(Z1+Z2)/2)
    npt.assert_almost_equal(B[0],B_center_analytic, decimal = 9)
