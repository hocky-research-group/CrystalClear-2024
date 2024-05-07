from ovito.data import *
import numpy as np

def modify(frame: int, data: DataCollection):
    
    # This user-defined modifier function gets automatically called by OVITO whenever the data pipeline is newly computed.
    # It receives two arguments from the pipeline system:
    # 
    #    frame - The current animation frame number at which the pipeline is being evaluated.
    #    data  - The DataCollection passed in from the pipeline system. 
    #            The function may modify the data stored in this DataCollection as needed.
    # 
    # What follows is an example code snippet doing nothing aside from printing the current 
    # list of particle properties to the log window. Use it as a starting point for developing 
    # your own data modification or analysis functions. 
    
    if data.particles != None:
        print("There are %i particles with the following properties:" % data.particles.count)
        for property_name in data.particles.keys():
            print("  '%s'" % property_name)
    data.particles_.create_property('Transparency')
    stypes = data.particles.structure_types[...]
    liq_index = np.where(stypes == 0)[0]
    data.particles_['Transparency'][liq_index] = 0.9
