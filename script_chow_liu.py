import numpy as np

total_count = 36 + 4 + 2 + 8 + 9 + 1 + 8 + 32

data = {('T', 'T', 'T'): 36,
        ('T', 'T', 'F'): 4,
        ('T', 'F', 'T'): 2,
        ('T', 'F', 'F'): 8,
        ('F', 'T', 'T'): 9,
        ('F', 'T', 'F'): 1,
        ('F', 'F', 'T'): 8,
        ('F', 'F', 'F'): 32,}

# X
pxt = (data[('T', 'T', 'T')] + data[('T', 'T', 'F')] + data[('T', 'F', 'T')] + data[('T', 'F', 'F')]) / total_count
pxf = (data[('F', 'T', 'T')] + data[('F', 'T', 'F')] + data[('F', 'F', 'T')] + data[('F', 'F', 'F')]) / total_count
# Z
pzt = (data[('T', 'T', 'T')] + data[('T', 'F', 'T')] + data[('F', 'T', 'T')] + data[('F', 'F', 'T')]) / total_count
pzf = (data[('T', 'T', 'F')] + data[('T', 'F', 'F')] + data[('F', 'T', 'F')] + data[('F', 'F', 'F')]) / total_count
# Y
pyt = (data[('T', 'T', 'T')] + data[('T', 'T', 'F')] + data[('F', 'T', 'T')] + data[('F', 'T', 'F')]) / total_count
pyf = (data[('T', 'F', 'T')] + data[('T', 'F', 'F')] + data[('F', 'F', 'T')] + data[('F', 'F', 'F')]) / total_count

# X, Y
pxtyt = (data[('T', 'T', 'T')] + data[('T', 'T', 'F')]) / total_count
pxtyf = (data[('T', 'F', 'T')] + data[('T', 'F', 'F')]) / total_count
pxfyt = (data[('F', 'T', 'T')] + data[('F', 'T', 'F')]) / total_count
pxfyf = (data[('F', 'F', 'T')] + data[('F', 'F', 'F')]) / total_count

# X, Z
pxtzt = (data[('T', 'T', 'T')] + data[('T', 'F', 'T')]) / total_count
pxtzf = (data[('T', 'T', 'F')] + data[('T', 'F', 'F')]) / total_count
pxfzt = (data[('F', 'T', 'T')] + data[('F', 'F', 'T')]) / total_count
pxfzf = (data[('F', 'T', 'F')] + data[('F', 'F', 'F')]) / total_count

# Z, Y
pytzt = (data[('T', 'T', 'T')] + data[('F', 'T', 'T')]) / total_count
pytzf = (data[('T', 'T', 'F')] + data[('F', 'T', 'F')]) / total_count
pyfzt = (data[('T', 'F', 'T')] + data[('F', 'F', 'T')]) / total_count
pyfzf = (data[('T', 'F', 'F')] + data[('F', 'F', 'F')]) / total_count

# Calculate mutual information x y
ixy = pxtyt*np.log2(pxtyt / (pxt * pyt))\
      + pxtyf*np.log2(pxtyf / (pxt * pyf)) \
      + pxfyt*np.log2(pxfyt / (pxf * pyt)) \
      + pxfyf*np.log2(pxfyf / (pxf * pyf))

# Calculate mutual information x z
ixz = pxtzt * np.log2(pxtzt / (pxt * pzt)) \
      + pxfzt * np.log2(pxfzt / (pxf * pzt))\
      + pxtzf * np.log2(pxtzf / (pxt * pzf))\
      + pxfzf * np.log2(pxfzf / (pxf * pzf))

# Calculate mutual information z y
izy = pytzt * np.log2(pytzt / (pyt * pzt)) \
      + pyfzt * np.log2(pyfzt / (pyf * pzt))\
      + pytzf * np.log2(pytzf / (pyt * pzf))\
      + pyfzf * np.log2(pyfzf / (pyf * pzf))

print(f"ixy = {ixy}")
print(f"ixz = {ixz}")
print(f"iyz = {izy}")








