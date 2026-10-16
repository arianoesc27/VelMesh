"""
###########################################################################
#                  Partitioning approach for Cybershake                   #
#--------------------------------------------------------------------------
# This script contains a series of functions designed to generate the     #
# partition parameters for running cybershake                             #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #  
# Date:  January 2025                                                     #
###########################################################################                                          
"""

import os
import sys 
import math
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from itertools import permutations

__version__ = "1.0.0"

import math
from itertools import permutations

def min_odd_factor_greater_than_one(n):
    for i in range(3, n + 1, 2):
        if n % i == 0:
            return i
    return None  # Shouldn't happen unless n is even and power of 2

def make_even(n):
    return n if n % 2 == 0 else n + 1

def closest_multiple_of_two(n):
    return n if n % 2 == 0 else n + (2 - n % 2)

def closest_divisible_by(n, factor):
    return n if n % factor == 0 else n + (factor - n % factor)

def closest_divisible_by(n, factor):
    # Finds the closest number to n that is divisible by 'factor'
    below = n - (n % factor)
    above = below + factor
    return below if abs(below - n) <= abs(above - n) else above

def get_factors(n):
    """Devuelve todas las tuplas (a, b, c) tal que a * b * c = n"""
    fact = []
    for a in range(1, int(n ** (1 / 3)) + 2):
        if n % a != 0:
            continue
        for b in range(1, int((n // a) ** 0.5) + 2):
            if (n // a) % b != 0:
                continue
            c = n // (a * b)
            if a * b * c == n:
                fact.append((a, b, c))
    return fact

def classify_partition(px, py, pz):
    ratios = sorted([px, py, pz])
    if ratios[0] == 1 and ratios[1] <= 4:
        return "planar"
    elif ratios[2] >= 2 * ratios[1] and ratios[0] == ratios[1]:
        return "columns"
    else:
        return "box"

def get_partition_weights(Lx, Ly, Lz):
    # Compute aspect ratios
    aspect_ratios = sorted([Lx, Ly, Lz])
    aspect_xy = aspect_ratios[2] / aspect_ratios[1]
    aspect_xz = aspect_ratios[2] / aspect_ratios[0]

    # Planar best for very flat domains, columns for tall/thin, box otherwise
    if aspect_xz > 6 or aspect_xy > 6:
        return {"planar": 1.0, "columns": 1.2, "box": 1.5}
    elif aspect_xz > 3 or aspect_xy > 3:
        return {"columns": 1.0, "planar": 1.2, "box": 1.5}
    else:
        return {"box": 1.0, "columns": 1.2, "planar": 1.5}

# desviación estándar o rango (diferencia max - min) para obtener particion mas balanceada
def balance_metric_alldim(n_x,n_y,n_z,partition):
    px, py, pz = partition[0]
    elements_per_core = [
        n_x / px,
        n_y / py,
        n_z / pz
    ]
    return max(elements_per_core) - min(elements_per_core)

def yz_balance_metric(n_y, n_z, partition):
    # cuanto más cercano a 0, más balance entre Y y Z
    py, pz = partition[0][1], partition[0][2]
    ey = n_y / py
    ez = n_z / pz
    return abs(ey - ez)  

# Buscar el múltiplo más cercano a target_Nx 
def closest_multiple(value, base):
    lower = (value // base) * base
    upper = lower + base
    return lower if abs(lower - value) <= abs(upper - value) else upper

def ajustar_dimensiones(Nz_elem, Ny_elem, Nx_elem):
# Cuando tengo un solo nodo quiero siempre garantizar que al menos se pueda usar el Px, Py, Pz = 7,8,2
# Esta funcion es exclusiva de 112 cores, si eso llega a cambiar puede necesitar modificaciones    
    
    # Asegurar que Nz_elem sea par y >= 80 (Nz/2 > 40)
    if Nz_elem % 2 != 0:
        Nz_elem += 1
    if Nz_elem < 80:
        Nz_elem = 80

    # Ajustar Ny_elem: mientras no cumpla la cercanía al valor de nz_elem 
    # o sea menor a 320 (40*8), ajustar
    target = Nz_elem / 2
    Ny_elem = round(Ny_elem / 8) * 8
    
    while abs((Ny_elem / 8) - target) > 20 or (Ny_elem / 8) < 40:
        if (Ny_elem / 8) > target:
            Ny_elem -= 8
        else:
            Ny_elem += 8

    # Ajustar Nx_elem: El valor objetivo para Nx_elem debe ser multiplo de 7 y par
    # Asegurar que Nx_elem / 7 >= 40 (o sea, Nx_elem >= 280)
    Nx_elem = closest_multiple(Nx_elem, 14)
    if Nx_elem < 280:
        Nx_elem = 280
        if Nx_elem % 14 != 0:
            Nx_elem = closest_multiple(Nx_elem, 14)

    return Nz_elem, Ny_elem, Nx_elem

def compute_partition(Lx, Ly, Lz, Vs_min, f_max, N_lambda,
                      cores_per_node=112, points_per_core=450000):

    print("\n--- START OF PARTITION CALCULATION ---\n") 

    # 1. Calculate lambda_min and cell size
    lambda_min = Vs_min / f_max
    dx = math.floor(lambda_min / N_lambda,)
    # Make sure dx is even
    if dx % 2 != 0:
        dx -= 1  
    dy = dz = dx
    print(f"Step 1: λ_min = {lambda_min:.2f} m → dx = dy = dz = {dx}")

    # 2. Calculate number of elements in each dimension
    Nx_elem = round(Lx * 1000 / dx)
    Ny_elem = round(Ly * 1000 / dy)
    Nz_elem = round(Lz * 1000 / dz)
    print(f"Step 2: Number of elements before adjustment: Nx_elem = {Nx_elem}, Ny_elem = {Ny_elem}, Nz_elem = {Nz_elem}")

    Lx = (Nx_elem * dx) / 1000
    Ly = (Ny_elem * dy) / 1000
    Lz = (Nz_elem * dz) / 1000

    print("\n--- CHECKING PROVIDED DIMENSION WITH CORES_PER_NODE ---\n") 
    # Before computing partition, check that one of dimensions are divisible by the minimum odd factor different from 1 and that the other two dimensions are even. 
    # If not, find the closest dimension to the one provided, the criterias are the following 
    # a. largest dimmension will be the closest to the provided or the provided if it is divisible by the minimum odd factor different from 1
    # b. the other two dimensions will be the clossest to the provided or, the provided that are even. 

    dims = {'Nx_elem': Nx_elem, 'Ny_elem': Ny_elem, 'Nz_elem': Nz_elem}
    min_odd_factor = min_odd_factor_greater_than_one(cores_per_node)
    print(f"Minimum odd factor > 1 of cores_per_node={cores_per_node} is {min_odd_factor}")

    divisible_by_odd = [name for name, val in dims.items() if val % (min_odd_factor*2) == 0]
    even_dims = [name for name, val in dims.items() if val % (2*2) == 0]

    if len(divisible_by_odd) == 1 and len(even_dims) == 2:
        print("Number of grid points already meet the criteria. No adjustment needed.")
    else:
        # Step 1: Adjust one to be divisible by the smallest odd factor > 1
        sorted_dims = sorted(dims.items(), key=lambda x: -x[1])  # Largest to smallest
        adjusted_name = None

        for name, val in sorted_dims:
            if val % min_odd_factor == 0:
                adjusted_name = name
                break
        else:
            # None divisible, so adjust the largest one
            adjusted_name, val = sorted_dims[0]
            new_val = closest_divisible_by(val, min_odd_factor*2)
            print(f"Adjusting {adjusted_name} from {val} to {new_val} to be divisible by {min_odd_factor} and even number")
            dims[adjusted_name] = new_val

        # Step 2: Adjust the other two to be multiple of 2
        for name, val in dims.items():
            if name == adjusted_name:
                continue
            if val % 2 != 0:
                new_val = closest_multiple_of_two(val)
                print(f"Adjusting {name} from {val} to {new_val} to make it multiple of 2")
                dims[name] = new_val

    Nx_elem, Ny_elem, Nz_elem = dims['Nx_elem'], dims['Ny_elem'], dims['Nz_elem']
    print(f"Final adjusted number of grid points: Nx_elem={Nx_elem}, Ny_elem={Ny_elem}, Nz_elem={Nz_elem}")

    print("\n--- CONTINUING PARTITION CALCULATION AFTER DIMENSIONS ADJUSTMENT ---\n") 
    # Paso 3 y 4 : Obtener nodos, cores, y todas las particiones posibles
    valid_partitions = []
    filtered_out = []

    while not valid_partitions:

        total_points = Nx_elem * Ny_elem * Nz_elem
        points_per_node = cores_per_node * points_per_core
        n_nodes = max(1, math.ceil(total_points / points_per_node))
        n_cores = n_nodes * cores_per_node

        print(f"Trying with cores_per_node = {cores_per_node}")
        print(f"Nodes = {n_nodes}, Cores = {n_cores}")

        valid_partitions.clear()
        filtered_out = []
        multiplier = 2
        
        if n_nodes == 1:
            # Ajustar todo
            Nz_elem, Ny_elem, Nx_elem = ajustar_dimensiones(Nz_elem, Ny_elem, Nx_elem)

        for (px, py, pz) in get_factors(n_cores):
            for perm in permutations((px, py, pz)):
                x_points_per_core = Nx_elem / perm[0]
                y_points_per_core = Ny_elem / perm[1]
                z_points_per_core = Nz_elem / perm[2]
                # print(px,py,pz, x_points_per_core,y_points_per_core,z_points_per_core)
                
                if x_points_per_core < 40:
                    filtered_out.append((perm, f"Nx_elem/px = {x_points_per_core:.2f} < 40 elements per core in x-dir"))
                    continue
                if y_points_per_core < 40:
                    filtered_out.append((perm, f"Ny_elem/py = {y_points_per_core:.2f} < 40 elements per core in y-dir"))
                    continue
                if z_points_per_core < 40:
                    filtered_out.append((perm, f"Nz_elem/pz = {z_points_per_core:.2f} < 40 elements per core in z-dir"))
                    continue

                if Nx_elem % perm[0] != 0 or Ny_elem % perm[1] != 0 or Nz_elem % perm[2] != 0:
                    filtered_out.append((perm, "Grid is not exactly divisible in one or more directions"))
                    continue

                ptype = classify_partition(*perm)
                if (perm, ptype) not in valid_partitions:
                    valid_partitions.append((perm, ptype))

        if not valid_partitions:
            print(f"No valid partitions found with {points_per_core} cores per node. Increasing by 2...")
            points_per_core = points_per_core  * multiplier
            multiplier += 1  # siguiente multiplicador

    
    if not valid_partitions:
        error_message = (
            "NO VALID PARTITIONS FOUND:\n"
            "(Criteria not met: grid points / core > 40 after partitioning). This may be due to:\n"
            f"- A high maximum frequency, greater than 1 Hz.\n"
            f"- A very low shear-wave velocity, which may be too small to partition at the given f_max.\n"
            f"- A low number of grid points per core.\n\n"
            "Consider reducing 'f_max' (maximum frequency) or increasing 'vs_min' (minimum shear-wave velocity) to allow a coarser grid.\n"
            "If modifying these values is not possible, try adjusting the number of grid points per core (points_per_core).\n"
        )
        print(error_message)
        sys.exit()
    
    '''
    # Balance en todas las direcciones -> funciona pero puede ser de ejecucion mas lenta
    valid_partitions.sort(
        key=lambda x: (
            balance_metric(Nx_elem, Ny_elem, Nz_elem, x),  # 1. balance X,Y,Z
            -x[0][2], -x[0][1], -x[0][0]                   # 2. orden descendente por (pz, py, px)
        )
    )
    '''

    # Balance Y-Z primero, luego x -> Esta opcion me da mas rapido al correr AWP-ODC-SGT por el AWP format (Z fast, X, Y slow)
    valid_partitions.sort(
        key=lambda x: (
            yz_balance_metric(Ny_elem, Nz_elem, x),   # 1. balance entre Y y Z
            -x[0][2], -x[0][1], -x[0][0]              # 2. orden descendente (pz, py, px)
        )
    )
    
    # Balance Y-Z primero, luego x
    # valid_partitions.sort(key=lambda x: (x[0][2], x[0][1], x[0][0]), reverse=True)

    # Pesos para priorizar tipos de partición
    partition_weights = get_partition_weights(Lx, Ly, Lz)

    # Obtener el tipo de partición para la mejor partición
    best_partition = max(valid_partitions, key=lambda x: partition_weights[classify_partition(*x[0])])
    #partition_type = classify_partition(*best_partition[0])

    # Imprimir numero de nodes y cores
    print(f"Step 3: Nodes = {n_nodes}, Cores = {n_cores}")

    # Imprimir la partición y su tipo
    print(f"Step 4: Best partition = {best_partition[0]}") # Prints only the permutation (px, py, pz)
    #print("Partition type:", best_partition[1])  # Prints the type (planar, columns, box)
    
    #partition_type = classify_partition(*best_partition[0])
    #print("Partition type from classify_partition:", partition_type)

    Px, Py, Pz = best_partition[0] 
    Nx = math.ceil(Nx_elem / Px) * Px
    Ny = math.ceil(Ny_elem / Py) * Py
    Nz = math.ceil(Nz_elem / Pz) * Pz

    Lx_adj = (Nx * dx) / 1000
    Ly_adj = (Ny * dy) / 1000
    Lz_adj = (Nz * dz) / 1000

    block_size = (Nx // Px, Ny // Py, Nz // Pz)

    print(f"Step 5: Optimal partition → Px = {Px}, Py = {Py}, Pz = {Pz}")
    print(f"        Elements per core → {block_size}")
    print(f"Step 6: Adjusted dimensions → Lx = {Lx_adj:.2f} km, Ly = {Ly_adj:.2f} km, Lz = {Lz_adj:.2f} km")
    print(f"Step 7: Number of elements after adjustment: Nx_elem = {Nx}, Ny_elem = {Ny}, Nz_elem = {Nz}")
    print("\n--- END OF PARTITION CALCULATION ---\n")

    with open("outputData/Partition/partition_otheroptions.out", "w") as f:
        with redirect_stdout(f):
            print("\nOther partitions that might work:")
            for (perm, ptype) in valid_partitions:
                if perm != best_partition:
                    block = (Nx // perm[0], Ny // perm[1], Nz // perm[2])
                    print(f"{perm} → tipo: {ptype} → block size: {block}")
            
            print("\nDiscarded options:")
            for (perm, reason) in filtered_out:
                print(f"Descartado {perm} → {reason}")

    return Lx_adj, Ly_adj, Lz_adj, dx, Nx, Ny, Nz, Px, Py, Pz, n_cores, n_nodes

def factor_pairs(n):
    pairs = []
    for i in range(1, int(abs(n)**0.5) + 1):
        if n % i == 0:
            a, b = i, n // i
            pairs.append((a, b))
            if a != b:
                pairs.append((b, a))
    return pairs

def filter_valid_pairs(pairs, width, height):
    return [(a, b) for a, b in pairs if width % a == 0 and height % b == 0]

def suggest_closest_multiple(target_x, target_y, best):
    a=best[0]*2 # in Cybershake, it has to be even number 
    b=best[1]*2 # in Cybershake, it has to be even number 
    new_x = round(target_x / a) * a
    new_y = round(target_y / b) * b
    return new_x, new_y

def adjust_to_nearest_multiple_of_2(n):
    return round(n / 2) * 2

def select_most_balanced_pair(valid_pairs):
    if not valid_pairs:
        return None
    return min(valid_pairs, key=lambda pair: abs(pair[0] - pair[1]))

def factor_pairs(n):
    pairs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs

def factor_pairs_permuted(n):
    pairs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            a, b = i, n // i
            if a == b:
                pairs.append((a, b))  # Solo una vez si son iguales
            else:
                # Agrega ambas permutaciones
                pairs.append((a, b))
                pairs.append((b, a))
    return pairs

def score_pair(pair):
    a, b = pair
    return abs(a - b)  # más balanceado → menor diferencia

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

# Block sizes AWP has a binary for, the ones Makefile.bsc-mn5 of AWP-GPU-SGT builds (one
# pmcl3d<N> per size). BLOCK_SIZE_Z is fixed when AWP is compiled, so a block size without a
# binary cannot be run until someone compiles it. The same list as the workflow's planner
# (steps/partitionVelocityModel/awp_grid.py). Every one is at most 512, the CUDA limit of
# 1024 threads per block divided by BLOCK_SIZE_Y = 2.
COMPILED_Z_BLOCKS = (32, 64, 96, 128, 160, 192, 200, 224, 252, 256, 288, 320,
                     352, 384, 416, 448, 480, 512)


def suggest_z_block(nz_original, prefer_multiple_of_32=True, max_relative_error=0.10,
                    available_blocks=None):
    """
    Suggest a GPU Z block size automatically, among the sizes AWP has a binary for.

    What this has to satisfy, from AWP's own code (pmcl3d.c):

        if (NZ % BLOCK_SIZE_Z != 0) { abort }

    So the block size must divide nz exactly, and since BLOCK_SIZE_Z is a compile-time
    constant, it must be one of the sizes a binary exists for (available_blocks, by default
    COMPILED_Z_BLOCKS). It does NOT have to be a power of two.

    Strategy:
    1. Among the available sizes, take those that divide nz exactly, so the model depth
       does not change. Prefer a multiple of 32 (the GPU warp size), and the largest one.
    2. If none divides nz, nz has to move. Take the available size whose nearest multiple
       is closest to nz, so the depth changes as little as possible.

    Notes:
    - The CUDA block uses BLOCK_SIZE_Y * BLOCK_SIZE_Z threads. With BLOCK_SIZE_Y = 2 and the
      CUDA limit of 1024 threads per block, BLOCK_SIZE_Z cannot exceed 512.
    - Below 32 a block wastes half of the GPU, since threads always run in groups of 32
      (warps) and each part of the GPU holds at most 32 blocks. In AWP the threads lie
      along Z, the direction that is contiguous in memory, so a warp of 32 neighbouring Z
      points is fetched in one memory request.
    - Sizes such as 200 or 252 are not multiples of 32. They work, AWP itself uses them,
      and they leave one warp partly empty, a waste of about 4 percent.
    """
    if nz_original <= 0:
        raise ValueError("nz_original must be positive")
    if max_relative_error < 0:
        raise ValueError("max_relative_error must be non-negative")

    BLOCK_SIZE_Y = 2
    MAX_THREADS_PER_BLOCK = 1024
    max_z_block = MAX_THREADS_PER_BLOCK // BLOCK_SIZE_Y
    WARP_SIZE = 32

    blocks = sorted(set(int(b) for b in (available_blocks or COMPILED_Z_BLOCKS)))
    if not blocks or blocks[0] <= 0 or blocks[-1] > max_z_block:
        raise ValueError(
            "available_blocks must be sizes between 1 and %d (CUDA allows %d threads per "
            "block and BLOCK_SIZE_Y is %d), got %s"
            % (max_z_block, MAX_THREADS_PER_BLOCK, BLOCK_SIZE_Y, blocks))

    # Step 1, available sizes that divide nz exactly. No change to the depth.
    exact = [b for b in blocks if nz_original % b == 0]
    if exact:
        if prefer_multiple_of_32:
            aligned = [b for b in exact if b % WARP_SIZE == 0]
            if aligned:
                return max(aligned)
        return max(exact)

    # Step 2, no available size divides nz, so the depth moves. The nearest runnable nz wins.
    valid_options = []
    for z_block in blocks:
        nz_adjusted = max(1, round(nz_original / z_block)) * z_block
        valid_options.append({
            "z_block": z_block,
            "rel_error": abs(nz_adjusted - nz_original) / nz_original,
            "aligned": 0 if (z_block % WARP_SIZE == 0 or not prefer_multiple_of_32) else 1,
            "n_z_blocks": nz_adjusted // z_block,
        })

    within_tolerance = [o for o in valid_options if o["rel_error"] <= max_relative_error]
    pool = within_tolerance if within_tolerance else valid_options

    best = sorted(pool, key=lambda o: (o["rel_error"], o["aligned"], o["n_z_blocks"]))[0]
    return best["z_block"]

def find_optimal_gpu_count(starting_n, gpus_per_node=4, max_search=100):
    best_score = float('inf')
    best_n = None
    best_pair = None

    for candidate in range(starting_n, starting_n + max_search + 1, 2):  # solo múltiplos de 2
        if candidate % gpus_per_node != 0:
            continue  # tiene que poder dividirse en nodos completos

        pairs = factor_pairs(candidate)
        for pair in pairs:
            if min(pair) < 4:  # evita configuraciones tipo (1, 1310)
                continue
            s = score_pair(pair)
            if s < best_score:
                best_score = s
                best_n = candidate
                best_pair = pair

    return best_n, best_pair

def compute_partition_GPU(Lx, Ly, Lz, Vs_min, f_max, N_lambda,
                          gpus_per_node=4, points_per_gpu=100000000,
                          z_block=None, prefer_multiple_of_32=True,
                          z_block_max_relative_error=0.10, available_z_blocks=None):

    print("\n--- START OF PARTITION CALCULATION ---\n")   
    
    # 1. Calculate lambda_min and cell size
    lambda_min = Vs_min / f_max
    dx = round(lambda_min / N_lambda, 2)
    dy = dz = dx
    print(f"Step 1: λ_min = {lambda_min:.2f} m → dx = dy = dz = {dx}")

    # 2. Calculate number of elements in each direction
    Nx_elem = round(Lx * 1000 / dx)
    Ny_elem = round(Ly * 1000 / dy)
    Nz_elem_original = round(Lz * 1000 / dz)

    # Resolve GPU z_block
    if z_block is None:
        z_block = suggest_z_block(
            Nz_elem_original,
            prefer_multiple_of_32=prefer_multiple_of_32,
            max_relative_error=z_block_max_relative_error,
            available_blocks=available_z_blocks
        )
        print(f"Step 2: No z_block provided. Suggested GPU z_block = {z_block}")
    else:
        print(f"Step 2: Using user-defined GPU z_block = {z_block}")
        if z_block not in (available_z_blocks or COMPILED_Z_BLOCKS):
            print(
                f"Note: there is no AWP binary for z_block={z_block} among "
                f"{sorted(available_z_blocks or COMPILED_Z_BLOCKS)}. AWP has to be compiled "
                "with that BLOCK_SIZE_Z before this mesh can run."
            )
        if prefer_multiple_of_32 and z_block % 32 != 0:
            print(
                f"Note: z_block={z_block} is not a multiple of 32. "
                "This is allowed and AWP itself uses values like 200 and 252, it just "
                "leaves one GPU warp partly empty."
            )

    Nz_elem = round(Nz_elem_original / z_block) * z_block

    relative_error = abs(Nz_elem - Nz_elem_original) / Nz_elem_original
    
    print(
        f"Step 2b: z_block = {z_block}, "
        f"Nz(original) = {Nz_elem_original}, Nz(adjusted) = {Nz_elem}, "
        f"relative error = {relative_error:.4f}"
    )

    if Nz_elem == 0:
        raise ValueError(
            f"Computed nz=0 after applying z_block={z_block}. "
            "Please increase ppwl (smaller spacing) or reduce z_block."
        )

    print(f"Step 3: Grid points before XY adjustment: "
        f"Nx = {Nx_elem}, Ny = {Ny_elem}, Nz(original) = {Nz_elem_original}, Nz(adjusted) = {Nz_elem}")
    
    
    total_points = Nx_elem * Ny_elem * Nz_elem
    points_per_node = gpus_per_node * points_per_gpu
    n_nodes = max(1, math.ceil(total_points / points_per_node)) 
    n_gpus = adjust_to_nearest_multiple_of_2(n_nodes * gpus_per_node)
    print(f"Step 3: Nodes = {n_nodes}, GPUs/node = {gpus_per_node}, GPUs(n_gpus) = {n_gpus}")

    # 3. Factor pairs
    factor_pairs_gpu = factor_pairs_permuted(n_gpus)
    #print(factor_pairs_gpu)
    prime_factor_pairs_gpu = [(2*a, 2*b) for a, b in factor_pairs_gpu]

    # 4. Filter valid pairs
    valid_pairs = filter_valid_pairs(prime_factor_pairs_gpu, Nx_elem, Ny_elem)
    #print(valid_pairs)
    
    # 5. List or suggest
    elements_pairs=[]
    if valid_pairs:
        print("Step 4: Valid (x, y) GPU pairs that evenly divide the number of grid points")
        for a, b in valid_pairs:
            print(f"  ({a//2}, {b//2}) → Nx = {Nx_elem/(a//2)}, Ny = {Ny_elem/(b//2)}")
            elements_pairs.append((Nx_elem/(a//2),Ny_elem/(b//2)))

        #print(elements_pairs)
        #print(valid_pairs)
        
        best_pair=select_most_balanced_pair(elements_pairs) 
        #print(best_pair)
	
        index = elements_pairs.index(best_pair)
        best_GPU_pair = valid_pairs[index]
        #print(best_GPU_pair) 
        
        best_pair_h = (best_GPU_pair[0] // 2, best_GPU_pair[1] // 2)
        Nx_sub = Nx_elem // (best_GPU_pair[0]/2)
        Ny_sub = Ny_elem // (best_GPU_pair[1]/2)
        print(f"Step 5: Suggested GPU partitioning → (GPUs in X, Y) = {best_pair_h} → Subdomain size = ({int(Nx_sub)}, {int(Ny_sub)}, {Nz_elem}) grid points per GPU")

        grid_spacing = dx
        new_x_length = Nx_elem*dx/1000
        new_y_length = Ny_elem*dx/1000
        new_z_length = Nz_elem*dx/1000
        nx = Nx_elem
        ny = Ny_elem
        nz = Nz_elem
        px = best_pair_h[0]
        py = best_pair_h[1]
        pz = 1

    else:
        print("\nNo valid (a, b) pairs found for the current number Nx, Ny. Recalculating the number of GPUs...")
        
        optimal_n_gpus, best_pair_a = find_optimal_gpu_count(n_gpus, gpus_per_node)
        print(f"Step 4: Selected n_gpus = {optimal_n_gpus}, with balanced GPUs pair (x,y) {best_pair_a}")
        
        [Nx_suggested, Ny_suggested] = suggest_closest_multiple(Nx_elem, Ny_elem, best_pair_a)
        print(f"Setp 5: New suggested grid points: Nx = {Nx_suggested}, Ny = {Ny_suggested} (was Nx = {Nx_elem}, Ny = {Ny_elem})")

        # Recalculate steps with adjusted dimensions
        n_nodes = optimal_n_gpus/gpus_per_node  
        n_gpus = optimal_n_gpus
        print(f"Step 6: Nodes = {n_nodes}, GPUs/node = {gpus_per_node}, total number of GPUs = {n_gpus}")
       
        # 4. Filter valid pairs
        Nx_sub = Nx_elem // (best_pair_a[0])
        Ny_sub = Ny_elem // (best_pair_a[1])
        print(f"Step 7: Suggested GPU partitioning → (GPUs in X, Y) = {best_pair_a} → Subdomain size = ({int(Nx_sub)}, {int(Ny_sub)}, {Nz_elem}) grid points per GPU")
       
        grid_spacing = dx
        new_x_length = Nx_suggested*dx/1000
        new_y_length = Ny_suggested*dx/1000
        new_z_length = Nz_elem*dx/1000
        nx = Nx_suggested
        ny = Ny_suggested
        nz = Nz_elem
        px = best_pair_a[0]
        py = best_pair_a[1]
        pz = 1   
      
    return new_x_length, new_y_length, new_z_length, grid_spacing, nx, ny, nz, px, py, pz, n_gpus, n_nodes, z_block

def update_parameters_file(file_path, updates):
    """
    Actualiza solo las líneas correspondientes a ciertas claves
    sin alterar el resto del archivo.

    file_path: Ruta al archivo parameters.in
    updates: Diccionario con claves a actualizar y sus nuevos valores
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    found_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in updates:
                new_lines.append(f"{key} = {updates[key]}\n")
                found_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    missing_keys = [k for k in updates if k not in found_keys]
    if missing_keys:
        new_lines.append("\n# Auto-generated mesh parameters\n")
        for key in missing_keys:
            new_lines.append(f"{key} = {updates[key]}\n")

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def write_partition_info(filename, x, y, z, spacing,nx, ny, nz, px, py, pz,n_cores, n_nodes):
    """
    Escribir los resultados de la particion a un archivo de texto.
    """
    with open(filename, 'w') as f:
        f.write("========================================\n")
        f.write("        Partitioning Information        \n")
        f.write("========================================\n")
        f.write(f"x_length     = {x}\n")
        f.write(f"y_length     = {y}\n")
        f.write(f"z_length     = {z}\n")
        f.write(f"grid_spacing = {spacing}\n\n")
        f.write("========================================\n")
        f.write("      Grid points in each direction     \n")
        f.write("========================================\n")
        f.write(f"nx = {nx}\n")
        f.write(f"ny = {ny}\n")
        f.write(f"nz = {nz}\n\n")
        f.write("========================================\n")
        f.write("   nodes and cores (or GPUs) per node   \n")
        f.write("========================================\n")
        f.write(f"n_nodes = {n_nodes}\n")
        f.write(f"n_cores = {n_cores}\n\n")
        f.write("========================================\n")
        f.write("    cores (or GPUs) in each direction   \n")
        f.write("========================================\n")
        f.write(f"px = {px}\n")
        f.write(f"py = {py}\n")
        f.write(f"pz = {pz}\n")

def plot_partitioned_domain(Lx, Ly, Lz, Nx, Ny, Nz, output_folder):

    # Create output folder if it doesn´t exist
    os.makedirs(output_folder, exist_ok=True)
   
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the grid for plotting
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    
    # Colors for each partition
    jet = plt.cm.rainbow
    num_colors = 10
    colors = [jet(i / num_colors) for i in range(num_colors)]
    
    # Loop over partitions and plot subdomains
    partition_count = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Calculate the position of the partition
                x_start = i * dx
                y_start = j * dy
                z_start = k * dz
                x_end = x_start + dx
                y_end = y_start + dy
                z_end = z_start + dz
                
                # Choose a color from the list, cycling through
                color = colors[partition_count % len(colors)]
                partition_count += 1
                
                # Plot the partition as a cuboid
                ax.bar3d(x_start, y_start, z_start, dx, dy, dz, color=color, edgecolor='k', linewidth=0.3, alpha=0.9)

    # Reduce the number of Z ticks
    ax.set_zticks(np.linspace(0, Lz, 3))  # Show only 5 ticks in Z direction
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))  # Un decimal
    ax.tick_params(axis='z', pad=2)  # reduce space between figure and z-axis

    # Set labels and title
    ax.set_xlabel('C1 to C4 = C2 to C3 [km]', labelpad=20)
    ax.set_ylabel('C1 to C2 = C4 to C3 [km]', labelpad=10)
    #ax.set_zlabel('Depth [km]', labelpad=-14)

    # Set axes limits
    max_range = np.max([Lx, Ly, Lz])
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_zlim([0, Lz])
    
    # Ensure the correct aspect ratio
    ax.set_box_aspect([Lx, Ly, Lz]) 
       
    # Invert vertical axis and show plot
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.gca().invert_zaxis()
   
    # Adjust the viewing angle for a better label visibility
    ax.view_init(elev=30, azim=40)
   
    # Save and show figure
    figure_path = os.path.join(output_folder, "partition.png")
    fig.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
