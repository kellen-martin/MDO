"""
import subprocess

# Path to GMAT executable
gmat_exe = "C:\\Users\\18475\\Desktop\\Projects\GMAT\\bin\\GMAT.exe"  # Adjust this path

# Path to GMAT script
gmat_script = "C:\\Users\\18475\\Desktop\\Projects\\GMAT\\samples\\Ex_DoubleLunarSwingby.script"

# Run GMAT
subprocess.run([gmat_exe, gmat_script])

subprocess.run([gmat_exe, gmat_script])
"""
import numpy as np
import sys
sys.path.append(r"C:\\Users\\18475\\Desktop\\Projects\\GMAT\\api")
from load_gmat import gmat
gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\samples\\Ex_MarsBPlane.script')
gmat.ShowObjects()
gmat.RunScript()
gmat.ShowObjects()

v1 = gmat.GetObject('dVx')
v2 = gmat.GetObject('MOI')
v1.Help()
print(v1)
"""
element1 = float(v1.GetField('Element1'))
element2 = float(v1.GetField('Element2'))
element3 = float(v1.GetField('Element3'))

delta_v1 = np.array([float(v1.GetField('Element1')), float(v1.GetField('Element2')), float(v1.GetField('Element3'))])
delta_v2 = np.array([float(v2.GetField('Element1')), float(v2.GetField('Element2')), float(v2.GetField('Element3'))])
print(delta_v1)
print(delta_v2)

delta_v1_mag = np.linalg.norm(delta_v1)
delta_v2_mag = np.linalg.norm(delta_v2)

delta_v = delta_v1_mag + delta_v2_mag
print(delta_v)"
"""