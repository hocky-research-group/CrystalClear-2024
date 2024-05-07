import hoomd
import hoomd.md
import numpy as np
import os
import sys
import argparse
import math
from copy import copy

def sample_spherical(npoints, ndim=3):
#simple points on a sphere from here https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.reshape((-1,3))

def fibonacci_sphere(samples):

    points = []
    phi = np.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = 1. - (i / float(samples - 1.)) * 2
        radius = np.sqrt(1. - y* y)
        theta = phi*i
        x = math.cos(theta)*radius
        z = math.sin(theta)*radius
        points.append((x,y,z))
    return points

def read_xyz_file(seed_file, particle_type_list):
    fh = open(seed_file,'r')
    line = fh.readline()
    config = []
    type_dict = {}
    type_list = []
    n_atoms = int(line)
    #skip a blank line
    line = fh.readline()
    line = fh.readline()
    while line:
        type, x, y, z = line.split()
        #if not type in type_dict:
        #    type_dict[type] = len(type_dict)
        #type_list.append(type_dict[type])
        type_list.append(particle_type_list.index(type))
        config.append(np.array((x,y,z)).astype(float))
        line = fh.readline()
    config = np.array(config)
    assert len(config) == n_atoms, "xyz reader- Number of atoms doesn't match the number of lines read in"
    return config, type_list
    

def screened_potential_shifted_force(r,rmin,rmax,radius_sum,steric_prefactor,electrostatic_prefactor,H,d):
    #a and b are set so that the force and energy goes to zero at rmax
    #make assumption that rmax > H

    #separate distances for electrostatic and hard sphere
    l = r - radius_sum
    lcut = rmax - radius_sum

    a = electrostatic_prefactor/d*np.exp(-lcut/d)
    b = -electrostatic_prefactor*np.exp(-lcut/d) - lcut*a

    p_term1 = electrostatic_prefactor*np.exp(-l/d) + a*l + b
    p_term2 = steric_prefactor*(28*((H/l)**.25-1) + 20./11*(1-(l/H)**2.75)+ 12*(l/H-1))
    if type(p_term2) == np.ndarray: p_term2[np.isnan(p_term2)]=0

    potential = p_term1 + p_term2*(l<H)

    # F = -dU/dr
    f_term1 = electrostatic_prefactor/d*np.exp(-l/d) - a
    f_term2 = -steric_prefactor/H*(12-7*(H/l)**(1.25)-5*(l/H)**(1.75))
    if type(f_term2) == np.ndarray: f_term2[np.isnan(f_term2)]=0
    force = f_term1 + (f_term2)*(l<H)
    return potential, force

mV_to_kBT = 25.7
joule_to_kBT = 4.11e-21

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--gpu",default=False,action="store_true")
    parser.add_argument("--zwall",default=True,help="Turn on wall perp to z (default: False)",action="store_true")
    parser.add_argument("--gravity",default=0,help="Turn on downward gravity constant force in z by setting to a positive number (default: 0)",type=float)
    parser.add_argument("--chargewall",default=0,help="Turn on wall perp to z attracting -(>0) or +(<0) particles (default: False)",type=float)
    parser.add_argument("-RN","--radiusN",default=105.0,help="Radius of colloids (default: %(default)s)",type=float)
    parser.add_argument("-RP","--radiusP",default=85.0,help="Radius of colloids (default: %(default)s)",type=float)
    parser.add_argument("-B","--brush_length",help="Brush length (default: %(default)s)",default=10,type=float)
    parser.add_argument("-s","--brush_density",help="Brush density (default: %(default)s)",default=0.09,type=float)
    parser.add_argument("-d","--debye_length",help="Debye length (default: %(default)s)",default=5.5,type=float)
    parser.add_argument("--surface_potentialN",help="Surface potential N in mV (default: %(default)s)",default=-50,type=float)
    parser.add_argument("--surface_potentialP",help="Surface potential P in mV (default: %(default)s)",default=30,type=float)
    parser.add_argument("--dielectric_constant",help="Dielectric constant (default: %(default)s)",default=68,type=float)
    parser.add_argument("--gamma",help="Drag coefficient (default: %(default)s)",default=0.001,type=float)
    parser.add_argument("--massP",help="Set the mass of the positive particles (default: %(default)s)",default=1.0,type=float)
    parser.add_argument("--massN",help="Set the mass of the negative particles (default: set relative to P)",default=None,type=float)
    parser.add_argument("--seed",help="Random seed (default: %(default)s)",default=1,type=int)
    parser.add_argument("--fraction_positive",help="Fraction positive charge (default: %(default)s)",default=0.5,type=float)
    parser.add_argument("-a","--lattice_spacing",help="Lattice spacing in terms of P diameter (default: %(default)s)",default=8,type=float)
    parser.add_argument("--lattice_repeats",default=8,type=int,help="times to repliacte the system in each direction (default: %(default)s)") #no of times we want to replicate in one direction
    parser.add_argument("--lattice_type",default="sc",help="Lattice type (bcc, sc, fcc) (default: %(default)s)")
    parser.add_argument("--orbit_factor",default=1.3,type=float,help="Factor beyond some of radii and brush at which to start N particles (default: %(default)s)")
    parser.add_argument("--dt",help="Simulation time step (default: %(default)s)",type=float,default=0.05)
    parser.add_argument("--dump_frequency",help="Dump frequency (default: %(default)s)",type=int,default=500000)
    parser.add_argument("--seed_file",help="xyz file specifying some seed coordinates",default=None)
    parser.add_argument("-i","--inputfile",default=None,help="Input file (gsd file, optional)",type=str)
    parser.add_argument("--scale_by",default="max_diameter",help="Scale xyz by this (default: %(default)s), options: max_diameter, max_radius")
    parser.add_argument("-o","--outputprefix",help="Output prefix (required)",type=str,required=True)
    parser.add_argument("-n","--nsteps",help="Number of steps to run",type=int,required=True)
    parser.add_argument("-T","--temperature",help="Temperature at which to run, or list of tuples specifying annealing schedule",default="1.0")
    parser.add_argument("-m","--mode",help="Integrator/dynamics scheme. Allowed: Minimize, Langevin, NVT (default: %(default)s)",default="Langevin")
    args = parser.parse_args()
    
    locals().update(vars(args))
    if gpu:
        hoomd.context.initialize("--mode=gpu")
        print("Running on the GPU")
    else:
        hoomd.context.initialize("--mode=cpu")
    np.random.seed(seed)
    

    #use harmonic average from Derjaguin approximation: https://en.wikipedia.org/wiki/Derjaguin_approximation
    ionic_radius = 2./(1./radiusN + 1./radiusP)
    repulsion_radius = (radiusN+radiusP)/2.
    max_radius = np.max(( radiusN, radiusP))

    radius_list = [radiusP, radiusN, ionic_radius]
    print("Particle radii and radii (Derjaguin, avg)",radius_list)
    
    brush2 = brush_length*2
    steric_prefactor = np.pi*16*repulsion_radius*(brush2**2)*(brush_density**(3./2))/35.
    print("Steric prefactor is: %f"%steric_prefactor)
    
    permitivity = 8.85e-12 #Farad/M
    ionic_radius_in_m = 1e-9*ionic_radius
    radiusN_in_m = 1e-9*radiusN
    radiusP_in_m = 1e-9*radiusP
    surface_potentialN_in_V = surface_potentialN/1000
    surface_potentialP_in_V = surface_potentialP/1000
    electrostatic_prefactors = [ 2*np.pi*dielectric_constant*permitivity*radiusP_in_m/joule_to_kBT*surface_potentialP_in_V*surface_potentialP_in_V, 2*np.pi*dielectric_constant*permitivity*radiusN_in_m/joule_to_kBT*surface_potentialN_in_V*surface_potentialN_in_V, 2*np.pi*dielectric_constant*permitivity*ionic_radius_in_m/joule_to_kBT*surface_potentialN_in_V*surface_potentialP_in_V ]
    print("Electrostatic prefactors are: ",electrostatic_prefactors)
    
    irun = 0
    if inputfile is not None:
        if not os.path.exists(inputfile):
            print("Error: inputfile %s does not exist"%inputfile)
            sys.exit(1)
        system=hoomd.init.read_gsd(inputfile,frame=-1)
        snapshot = system.take_snapshot()

        particle_type_list = system.particles.types
        print("Particle types:",particle_type_list)

        if zwall is True:
            max_z = snapshot.box.Lz/2. + radiusP/2.
            
        
    else:
        
        if lattice_type == "fcc":
            lattice = hoomd.lattice.fcc
        elif lattice_type == "bcc":
            lattice = hoomd.lattice.bcc
        elif lattice_type == "sc":
            lattice = hoomd.lattice.sc
        else:
            print("Lattice type %s is not supported (only fcc, bcc, sc)"%lattice_type)
            sys.exit(1)
  
        irun = 0    
        if hoomd.comm.get_rank() == 0:
            orbit_distance = (radiusP+radiusN+brush2)*orbit_factor
            print("Putting N particles at a distance of %f from P particles"%orbit_distance)
            num_positive = lattice_repeats**3
            num_negative_per_center = int( (1-fraction_positive)/fraction_positive )
            N = ((1+num_negative_per_center)*lattice_repeats)**3
            num_negative = N - num_positive
            P_unit_cell = lattice(a=lattice_spacing*2*radiusP )

            P_in_uc = P_unit_cell.N
            N_in_uc = P_in_uc*num_negative_per_center
            total_in_uc = P_in_uc + N_in_uc

            #scale mass if not set
            if massN is None:
                massN = massP/(radiusP/radiusN)**3
                print("Setting mass N: %f (mass P: %f)"%(massN,massP))
  
            mass_dict = {}
            mass_dict['N'] = massN
            mass_dict['P'] = massP

            particle_positions = P_unit_cell.position
            all_positions = []
            all_masses = []
            all_types = []
            all_diameters = []
            for i in range(P_in_uc):
                xyz = particle_positions[i]
                all_positions.append(xyz)
                all_masses.append(massP)
                all_types.append('P')
                all_diameters.append(2*radiusP)

                sattelite_positions = sample_spherical(num_negative_per_center)*orbit_distance
                for xyz_s in sattelite_positions:
                    all_positions.append(xyz_s+xyz)
                    all_masses.append(massN)
                    all_types.append('N')
                    all_diameters.append(2*radiusN)
            
            print(all_positions)
            uc = hoomd.lattice.unitcell(N = total_in_uc,
                            a1 = P_unit_cell.a1,
                            a2 = P_unit_cell.a2,
                            a3 = P_unit_cell.a3,
                            dimensions = P_unit_cell.dimensions,
                            position = all_positions,
                            type_name = all_types,
                            mass = all_masses,
                            diameter = np.array(all_diameters),
            )
            snapshot = uc.get_snapshot()
            snapshot.replicate(lattice_repeats,lattice_repeats,lattice_repeats)

            print("Box size: ",snapshot.box)

            #half of the box lengths in x, y, z directions
            max_z = snapshot.box.Lz/2.
            max_x = snapshot.box.Lx/2.
            max_y = snapshot.box.Ly/2.


            
            #add seed
            if seed_file is not None and os.path.exists(seed_file):
                seed_xyz, seed_particle_types = read_xyz_file(seed_file,particle_type_list)
                print(seed_particle_types)
                # divided by max diameter, have to scale back up
                if scale_by == "max_diameter":
                    seed_positions = seed_xyz * 2*max_radius
                    print("Warning: scaling xyz by max diameter")
                elif scale_by == "max_radius":
                    seed_positions = seed_xyz * max_radius
                    print("Warning: scaling xyz by max radius")
                else:
                    print("Warning: not scaling xyz by anything")
            else:
                seed_positions = None


        system=hoomd.init.read_snapshot(snapshot)
        print(system.box)
        hoomd.update.box_resize(Lx = 2.*max_x+8*radiusN, Ly = 2.*max_y+8*radiusN, Lz=2.*max_z+8*radiusN, period=None, scale_particles=False)
        print(system.box)
        mod_maxX = system.box.Lx/2.
        mod_maxY = system.box.Ly/2.
        mod_maxZ = system.box.Lz/2.

       # if zwall is True:
       #     tags_to_remove = []
       #     cutoff = 6*radiusN
       #     for p in system.particles:
       #         dx1 = abs(p.position[0] - max_x)
       #         dy1 = abs(p.position[1] - max_y)
       #         dz1 = abs(p.position[2] - max_z)
       #         dx2 = abs(p.position[0] - (-max_x))
       #         dy2 = abs(p.position[1] - (-max_y))
       #         dz2 = abs(p.position[2] - (-max_z))
       #         if dx1 < cutoff:
       #             tags_to_remove.append(p.tag)
       #         elif dy1 < cutoff:
       #             tags_to_remove.append(p.tag)
       #         elif dz1 < cutoff:
       #             tags_to_remove.append(p.tag)
       #         elif dx2 < cutoff:
       #             tags_to_remove.append(p.tag)
       #         elif dy2 < cutoff:
       #             tags_to_remove.append(p.tag)
       #         elif dz2 < cutoff:
       #             tags_to_remove.append(p.tag)


       #    print("length of tags to remove", len(tags_to_remove))

       #     for t in tags_to_remove:
       #         system.particles.remove(t)



        particle_type_list = system.particles.types
        print("Particle types:",particle_type_list)

        if seed_positions is not None:
            first_idx = 0
            for pidx, pxyz in enumerate(seed_positions):
                particle_type_id = seed_particle_types[pidx]
                particle_type = particle_type_list[particle_type_id]
                t = system.particles.add(particle_type)
                if first_idx ==0: first_idx = int(t)
                system.particles[t].position = pxyz
                system.particles[t].diameter = radius_list[particle_type_id]*2
                system.particles[t].mass = mass_dict[particle_type]
            # search for overlaps
            for pidx, pxyz in enumerate(seed_positions):
                tags_to_remove = []
                for p in system.particles:
                    pid = p.typeid
                    if particle_type_id == pid:
                        cutoff = (2*radius_list[pid]*1.3)**2
                    else:
                        cutoff = (2*radius_list[-1]*1.3)**2
                    #dist_cut = 2*radius_
                    dr = pxyz - p.position
                    dr = system.box.min_image(dr)
                    dr2_mag = np.dot(dr,dr)
                    if dr2_mag < cutoff and p.tag<first_idx: tags_to_remove.append(p.tag)
            print("Removing %i particles that overlap with seed"%len(tags_to_remove))
            for t in tags_to_remove:
                system.particles.remove(t)

    
    #nl = hoomd.md.nlist.tree(r_buff=0.8*ionic_radius)
    #nl = hoomd.md.nlist.cell(r_buff=0.8*ionic_radius)
    nl = hoomd.md.nlist.cell()
    #nl.reset_exclusions(exclusions = ['constraint'])

    if gravity>0:
        typeN = hoomd.group.type(type='N')
        typeP = hoomd.group.type(type='P')
#kbt
        fgrav = -gravity
        gravity_force_P = hoomd.md.force.constant(fvec=[0,0,fgrav],group=typeP)
#        gravity_force_N = hoomd.md.force.constant(fvec=[0,0,fgrav*massN],group=N)
    
    if zwall is True:
        print("Turning on Z-wall with max_z:",max_z)
        #create a repulsive wall
        upper_wall_x = hoomd.md.wall.plane(origin=(mod_maxX,0,0),normal=(-1,0,0),inside=True)
        lower_wall_x = hoomd.md.wall.plane(origin=(-mod_maxX,0,0),normal=(1,0,0),inside=True)
        upper_wall_y = hoomd.md.wall.plane(origin=(0,mod_maxY,0),normal=(0,-1,0),inside=True)
        lower_wall_y = hoomd.md.wall.plane(origin=(0,-mod_maxY,0),normal=(0,1,0),inside=True)
        upper_wall_z = hoomd.md.wall.plane(origin=(0,0,mod_maxZ),normal=(0,0,-1),inside=True)
        lower_wall_z = hoomd.md.wall.plane(origin=(0,0,-mod_maxZ),normal=(0,0,1),inside=True)

        wall_group = hoomd.md.wall.group(upper_wall_x,lower_wall_x,upper_wall_y,lower_wall_y,upper_wall_z,lower_wall_z)
        wall_force = hoomd.md.wall.slj(wall_group,r_cut=radiusN*(2.0**(1.0/6.0)))
        wall_force.force_coeff.set('N',epsilon=1,sigma=radiusN,alpha=0,r_cut=radiusN*(2.0**(1.0/6.0)))
        wall_force.force_coeff.set('P',epsilon=1,sigma=radiusP,alpha=0,r_cut=radiusP*(2.0**(1.0/6.0)))

    
    if chargewall!=0 and not zwall:
        eps=np.abs(chargewall)
        upper_wall = hoomd.md.wall.plane(origin=(0,0,0),normal=(0,0,-1),inside=True)
        lower_wall = hoomd.md.wall.plane(origin=(0,0,0),normal=(0,0,1),inside=True)
        wall_group = hoomd.md.wall.group(upper_wall,lower_wall)
    #pos is 0, neg is 1
        if chargewall < 0:
            attractive_wall_force = hoomd.md.wall.lj(wall_group,r_cut=radius_list[0]*5)
            attractive_wall_force.force_coeff.set('N',epsilon=0,sigma=radius_list[1]*2)
            attractive_wall_force.force_coeff.set('P',epsilon=eps,sigma=radius_list[0]*2)
    
    
        if chargewall > 0:
            attractive_wall_force = hoomd.md.wall.lj(wall_group,r_cut=radius_list[1]*5)
            attractive_wall_force.force_coeff.set('N',epsilon=eps,sigma=radius_list[1]*2)
            attractive_wall_force.force_coeff.set('P',epsilon=0,sigma=radius_list[0]*2)
    
    if hoomd.comm.get_rank() == 0:
        pot_list = []
        force_list = []
        count = 0
        for my_radius in radius_list:
            #sign = 1
            radius_sum = 2*my_radius
            if my_radius == radius_list[-1]: 
                #sign = -1 
                radius_sum = radius_list[0]+radius_list[1]
            pot_min = 1.00005*radius_sum
            pot_max = radius_sum+20*debye_length
            dpot = (pot_max-pot_min)/5000.
            test_range = np.arange(pot_min,pot_max+dpot,dpot)
            pot,force = screened_potential_shifted_force(test_range,rmin=pot_min, rmax=pot_max, radius_sum=radius_sum, steric_prefactor=steric_prefactor,electrostatic_prefactor=electrostatic_prefactors[count],H=brush2,d=debye_length)
            pot_list.append(pot.reshape(-1,1))
            force_list.append(force.reshape(-1,1))
            np.savetxt(outputprefix+"_potential%i.txt"%count, np.concatenate( (test_range.reshape(-1,1),pot.reshape(-1,1),force.reshape(-1,1)),axis=-1) )
            count = count+1
        #example of how to write out potential for plotting it
    
    table = hoomd.md.pair.table(width=5000,nlist=nl)
    pair_type = 0
    print("len_radius_list",len(radius_list))
    print("len_particle_type_list", len(particle_type_list))
    print(particle_type_list)
    for i in range(0,2):
        typei = particle_type_list[i]
        for j in range(i,2):
            typej = particle_type_list[j]
            #sign = (-1)**((typei==typej) +1)
            if i == j:
                my_radius = radius_list[i]
                radius_sum = 2*my_radius
                pair_type = i
            else:
                my_radius = radius_list[-1]
                radius_sum = radius_list[0]+radius_list[1]
                pair_type = 2
            print("Setting potential for type pair %i (%i, %i) with radius_sum %f"%(pair_type,i,j, radius_sum))
            potential_range = radius_sum + 10*debye_length
            table.pair_coeff.set(typei,typej, func=screened_potential_shifted_force, 
                                            rmin=1.00005*radius_sum, rmax=potential_range,
                                            coeff=dict(radius_sum=radius_sum, steric_prefactor=steric_prefactor,
                                            electrostatic_prefactor=electrostatic_prefactors[pair_type],H=brush2,d=debye_length),
                                )
            pair_type = pair_type+1


    
    

    if temperature.find(':')>0:
        temp_list = np.array( temperature.split(':') ,dtype=float)
        assert len(temp_list)%2==0 and len(temp_list)>=0,"must give an even number of temperature arguments"
        temp_list = [tuple(x) for x in temp_list.reshape((-1,2))]
        temperature = hoomd.variant.linear_interp(points=temp_list)
    else:
        temperature = float(temperature)
    
   # nonrigid = hoomd.group.nonrigid()
    typeN = hoomd.group.type(type='N')
    typeP = hoomd.group.type(type='P')
    bothNP = hoomd.group.union(name='bothNP',a=typeN,b=typeP)
   # rigid_center = hoomd.group.rigid_center()
    all = hoomd.group.all()

    hoomd.analyze.log(filename=outputprefix+'.log',
                      quantities=['potential_energy', 'temperature'],
                      period=dump_frequency//10,
                      overwrite=True)
    
    if mode.lower() == "minimize":
        fire = hoomd.md.integrate.mode_minimize_fire(dt=dt,Etol=1e-7,min_steps=100,group=all)
        hoomd.dump.gsd(outputprefix+'.gsd', period=500, group=all, overwrite=True)
        while not(fire.has_converged()):
            hoomd.run(500)
        sys.exit()
    
    elif mode.lower() == "langevin":
        hoomd.md.integrate.mode_standard(dt=dt)
        bd = hoomd.md.integrate.langevin(group=bothNP, kT=temperature, seed=seed)
        print("Using gamma value as:", gamma)
        bd.set_gamma('N',0.001)
        bd.set_gamma('P',0.001)
       # for particle_type in particle_type_list:
       #     bd.set_gamma(particle_type,gamma)
      #  bdrigid = hoomd.md.integrate.langevin(group=rigid_center, kT=0.2, seed=seed)
      #  bdrigid.set_gamma('S',gamma=1.0)
    elif mode.upper() == "NVT":
        hoomd.md.integrate.mode_standard(dt=dt)
        integrator = hoomd.md.integrate.nvt(group=all, kT=temperature, tau=1/gamma)
        integrator.randomize_velocities(seed=seed)
    elif mode.upper() == "NPT":
        hoomd.md.integrate.mode_standard(dt=dt)
        integrator1 = hoomd.md.integrate.nvt(group=all, kT=temperature, tau=1/gamma)
        eq = hoomd.dump.gsd(outputprefix+'.eq.gsd', period=dump_frequency, group=all, overwrite=True)
        integrator1.randomize_velocities(seed=seed)
        hoomd.run(100000)
        eq.disable()
        integrator1.disable()
        #integrator = hoomd.md.integrate.npt(group=all, kT=1.0, tau=1/gamma, tauP=10/gamma, P=5e-10)
        #integrator = hoomd.md.integrate.npt(group=all, kT=1.0, tau=1/gamma, tauP=10/gamma, P=0.1)
        #integrator = hoomd.md.integrate.npt(group=all, kT=1.0, tau=100*dt, tauP=100*dt, P=1e-8)
        integrator = hoomd.md.integrate.npt(group=all, kT=1.0, tau=1.0, tauP=1.2, P=1.0)
        #integrator.randomize_velocities(seed=seed)
    else:
        print("Mode '%s' not supported"%mode)
        sys.exit(1)
    #integrator = hoomd.md.integrate.nvt(group=all, kT=1.0, tau=1/gamma)
    #print("NVT equilibrating")
    #hoomd.run(5000)
    #integrator.disable()

    #nl.tune()
    
    hoomd.dump.gsd(outputprefix+'.gsd', period=dump_frequency, group=all, overwrite=True)
#    hoomd.dump.dcd(outputprefix+'.gsd', period=dump_frequency, group=all, overwrite=True, unwrap_full=True)
    
#    rigid.create_bodies()
    hoomd.run(nsteps)
