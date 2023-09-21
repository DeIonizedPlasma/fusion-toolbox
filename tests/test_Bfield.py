''' Module to test Bfield.py calculations'''
from math import pi
import numpy as np
import numpy.testing as npt
import pytest
from fusion_toolbox.Bfield import Coil


MU_0 = 4*pi*1E-7 # T * m / A

def generate_pf_coil_xyz(coil_radius, num_coil_pts = 100, coil_height = 0):
    '''
    Get xyz coordinates of points along a pf coil 
    coil radius: [m]
    num_coil_points: Number of points used to define the coil (int)
    coil_height: pf coil height [m]
    Returns: xyz - np.ndarray [N,3]
    '''
    thetas =  np.linspace(0, 2*pi, num_coil_pts)
    xyz = np.zeros((num_coil_pts,3))
    xyz[:,0] = coil_radius * np.cos(thetas)
    xyz[:,1] = coil_radius * np.sin(thetas)
    xyz[:,2] = coil_height
    return xyz

def analytic_B_center_of_pf_coil(current, coil_radius):
    '''
    Analytic calculation of magnetic field at the center of a circular pf coil 
    current: [A]
    coil radius: [m]
    returns: B-field [T]
    '''
    return np.array([0,0,MU_0 * current / (2 * coil_radius)])

@pytest.mark.parametrize(
    "current, coil_radius",
    [[1E3,0.3], [1E5, 0.02], [1E7, 0.004], [1E2, 1]
    ])
def test_B_center_of_pf_circular_coil(current, coil_radius):
    '''
    current: [A]
    coil radius: [m]
    Checks the calculated field at the center of a circular coil against the analytic solution
    '''

    # Generate test coil and calculate the field at the center
    xyz = generate_pf_coil_xyz(coil_radius, num_coil_pts= int(1E4))
    test_coil = Coil(xyz, current)
    B_center = test_coil.B([np.array([0,0,0])])

    # Compare to analytic calculation
    B_analytic = [analytic_B_center_of_pf_coil(current,coil_radius)]
    npt.assert_almost_equal(B_center, B_analytic, decimal = 4)
