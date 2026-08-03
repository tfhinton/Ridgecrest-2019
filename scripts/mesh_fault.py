#%%
import config
from codes.helpers import fault_from_cfm, remesh_fault
from codes import FaultTriangles


faults = [fault_from_cfm(fp) for fp in config.FAULT_FILEPATHS]

faults[0] = remesh_fault(faults[0], [0,1500,3000,5500,8500,11500,15000], n_top=40, n_bottom=8, project_below=10000., save=(config.FAULT_DIR / "mainshock_fault"))
faults[0].name = "Mainshock Fault"

faults[1] = remesh_fault(faults[1], [0,1500,3000,5500,8500,11500,15000], n_top=20, n_bottom=4, project_below=10000., save=(config.FAULT_DIR / "foreshock_fault"))
faults[1].name = "Foreshock Fault"

fault = FaultTriangles.merge(faults)
fault.name = "Ridgecrest Faults"
fault.save( config.FAULT_PICKLE.with_suffix("") )

# import cmcrameri.cm as cmc
# import matplotlib.pyplot as plt
# fault.plot_fault3d(color_by="layer", cmap=cmc.tokyo)
# plt.show()
# fault.plot_slip_2d(cmap=cmc.batlow)
# plt.show()
