# %%
import os 
from ctypes import * 
import numpy as np
import pandas as pd
from shapely.geometry import *
import argparse
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import re
# ⚠️⚠️⚠️ WARNING: CHECK THE PATH ⚠️⚠️⚠️
from sofistik_daten import * 
from find_struts import extract_coord

# ⚠️⚠️⚠️ WARNING: CHECK THE PATH ⚠️⚠️⚠️
# Set DLL dir path
os.add_dll_directory(r"C:\Program Files\SOFiSTiK\2024\SOFiSTiK 2024")

def print_last_errors(prt_file, n=20):
    with open(prt_file, encoding="latin1") as f:
        lines = f.readlines()
        for l in lines[-n:]:
            if "error" in l.lower():
                print(l.strip())

def sofistik_failed(prt_file, err_file):
    if err_file.exists() and err_file.stat().st_size > 0:
        return True

    if prt_file.exists():
        with open(prt_file, encoding="latin1") as f:
            error=False
            #liste des erreurs à afficher
            errors=[]
            # liste des warnings à afficher
            warnings=[]
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip("\n")
                l = line.lower()
                        
                # erreurs fatales explicites
                if re.search(r"\b(error|fehler|fatal)\b", l):
                    errors.append(l)
                    error=True
                # Warnings
                elif "warning" in l:
                    msg = [line]

                    # lire les lignes suivantes associées
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line == "" or next_line.startswith("+++++"):
                            break
                        msg.append(next_line)
                        j += 1

                    warnings.append(" ".join(msg))
                    i = j
                    continue

                i += 1

            if errors : 
                print("Error SOFISTIK :")
                for e in errors : 
                    print(e)
            if warnings :
                print("Warnings SOFISTIK :")
                for w in warnings :
                    print(w)


    return error


class CDBData:
    
    def __init__(self, library_path="sof_cdb_w-2024.dll"):
        """
        Initializes the CDB manager with the path to the DLL library.
        
        :param library_path: Path to the DLL library for CDB management.
        """
        self.myDLL = cdll.LoadLibrary(library_path)
        self.cdbStat = None
        self.Index = None
        self.file_path = None

    def open_cdb(self, file_name, cdb_index=99):
        """
        Opens the specified CDB file.
        
        :param file_name: Path to the CDB file.
        :param cdb_index: CDB index (default: 99).
        """
        self.Index = c_int()
        self.Index.value = self.myDLL.sof_cdb_init(file_name.encode('utf8'), cdb_index)
        self.cdbStat = c_int()
        self.cdbStat.value = self.myDLL.sof_cdb_status(self.Index.value)
        self.file_path = file_name
        print("CDB opened successfully, CDB Status =", self.cdbStat.value)

    def close_cdb(self):
        """
        Closes the CDB.
        """
        self.myDLL.sof_cdb_close(0)
        self.cdbStat.value = self.myDLL.sof_cdb_status(self.Index.value)
        if self.cdbStat.value == 0:
            print("CDB closed successfully, CDB Status = 0")
    
   
    def extract_quad_forces(self, load_case):
        """
        Extracts the forces of quadrilateral elements from the CDB.
        
        :param start_key: Initial key of the group to be read (default: 1).
        :param max_attempts: Maximum number of attempts to increment the key (default: 11).

        :return: List of forces for each quadrilateral element.
        """
        forces_data = []  # List to store the forces of each element
        ie = c_int(0)
        RecLen = c_int(sizeof(CQUAD_FOR))  # Size of the forces structure
        forces = CQUAD_FOR()  # Instance of the forces structure
        
        # Attempt to read the forces by incrementing the key up to max_attempts
        
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(self.Index, 210, load_case, byref(forces), byref(RecLen), 1)
            
            if ie.value == 0:  # No error, data has been read

                forces_data.append({
                    "quad_number": forces.m_nr,  # Quadrilateral number
                    "mxx": forces.m_mxx,        # Bending moment mxx
                    "myy": forces.m_myy,        # Bending moment myy
                    "mxy": forces.m_mxy,        # Bending moment mxy
                    "vx": forces.m_vx,          # Shear force vx
                    "vy": forces.m_vy,          # Shear force vy
                    "nx": forces.m_nx,          # Membrane force nx
                    "ny": forces.m_ny,          # Membrane force ny
                    "nxy": forces.m_nxy         # Membrane force nxy
                })

            # Reset RecLen before the next sof_cdb_get call
            RecLen = c_int(sizeof(CQUAD_FOR))
        
        return forces_data  # Return the list with all force data

    def extract_quad_membrane_forces(self,load_case=1):
        """
        Extracts the membrane forces of quadrilateral elements from the CDB.

        :param start_key: Initial key of the group to be read (default: 1).
        :param max_attempts: Maximum number of attempts to increment the key (default: 11).

        :return: DataFrame with membrane forces only.
        """
        df_forces=pd.DataFrame(self.extract_quad_forces(load_case))
        membrane_forces = df_forces[['quad_number', 'nx', 'ny', 'nxy']] #Select the membran forces
        membrane_forces = membrane_forces.iloc[2:].reset_index(drop=True)  # Delete first 2 rows that give useless data
        return membrane_forces
    
    def extract_node_coords(self):
        """
        Extracts node coordinates from the CDB.

        :return: Dictionary with node coordinates.
        """
        ie = c_int(0)
        RecLen = c_int(sizeof(CNODE))  # Size of the CNODE structure
        node = CNODE()  # Instance of the CNODE structure
        node_coords = {}  # Dictionary to store node coordinates

        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(self.Index, 20, 0, byref(node), byref(RecLen), 1)  # Read nodes from CDB
            if ie.value == 0:  # No error, node read correctly
                node_coords[node.m_nr] = (node.m_xyz[0], node.m_xyz[1],  node.m_xyz[2])  # Store X, Y, Z coordinates of the node
            RecLen = c_int(sizeof(CNODE))  # Reset the record size

        return node_coords  # Return node coordinates

    def extract_quad_elements(self):
        """
        Extracts quadrilateral elements (with associated nodes) from the CDB.

        :return: Tuple with list of nodes and list of element numbers.
        """
        ie = c_int(0)
        RecLen = c_int(sizeof(CQUAD))  # Size of the CQUAD structure
        quad_element = CQUAD()  # Instance of the CQUAD structure
        elements = {}  # Dictionary to store quadrilateral elements
        
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(self.Index, 200, 0, byref(quad_element), byref(RecLen), 1)  # Read quadrilateral elements
            if ie.value == 0:  # No error, element read correctly
                elements[quad_element.m_nr] = quad_element.m_node[:]
            RecLen = c_int(sizeof(CQUAD))  # Reset the record size

        return elements  # Return the nodes and the number of each element
    
    def extract_structure_bounds(self):
        """
        Extracts the bounds of the structure from the CDB.

        :return: List of bounds of this form: [xmin,ymin,zmin,xmax,ymax,zmax].
        """
        syst = CSYST()
        RecLen = c_int(sizeof(CSYST))
        ie = c_int(0)
        x = []
        while ie.value < 2:
            # Read SYST data (10/00)
            ie.value = self.myDLL.sof_cdb_get(self.Index, 10, 00, byref(syst), byref(RecLen), 1)
            if ie.value == 0:
                # Extract values from m_box (2x3)
                for i in range(3):
                    for j in range(2):
                        x.append(syst.m_box[i][j])
                return x



    def extract_cable(self):
        """
        Extracts the cable number (NR) and the node numbers (node).
        The relevant database key is 160/0.
        
        :return: A list of dictionaries containing { 'cable_number': ..., 'nodes': ... }.
        """
        cable_data = []
        ie = c_int(0)

        # Instantiate the structure and its record length
        cable = CCABL()
        RecLen = c_int(sizeof(CCABL))

        # Loop until ie >= 2, indicating that we have reached the end of the records
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(
                self.Index,     # CDB index (set when calling open_cdb)
                160,            # Main group number = 162
                0,      # Subkey = 0
                byref(cable),   # Pointer to the structure to be filled
                byref(RecLen),  # Size of the structure
                1               # Mode = 1 => read
            )

            if ie.value == 0:
                # No error, the structure is filled with valid data
                cable_data.append({
                    "cable_number": cable.m_nr,
                    "nodes":        (cable.m_node[0],cable.m_node[1]) 
                
                })
        
               
                
            # Reset the record length before the next sof_cdb_get call
            RecLen = c_int(sizeof(CCABL))

        return cable_data

    def extract_cable_forces(self, load_case=1):
        """
        Extracts the cable number (NR) and the cable force (N) for a given load case. Extract the cable effective stiffness too
        The relevant database key is 162/<load_case> (e.g., 162/1, 162/2, etc.).
        
        :param load_case: The load case number to read (default = 1).
        :return: A list of dictionaries containing { 'cable_number': ..., 'force': ... }.
        """
        cable_data = []
        ie = c_int(0)

        # Instantiate the structure and its record length
        cable = CCABL_RES()
        RecLen = c_int(sizeof(CCABL_RES))

        # Loop until ie >= 2, indicating that we have reached the end of the records
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(
                self.Index,     # CDB index (set when calling open_cdb)
                162,            # Main group number = 162
                load_case,      # Subkey = load_case
                byref(cable),   # Pointer to the structure to be filled
                byref(RecLen),  # Size of the structure
                1               # Mode = 1 => read
            )

            if ie.value == 0:
                # No error, the structure is filled with valid data
                cable_data.append({
                    "cable_number": cable.m_nr,
                    "force":        cable.m_n,
                    "effs" : cable.m_effs
                })
               

            # Reset the record length before the next sof_cdb_get call
            RecLen = c_int(sizeof(CCABL_RES))

        return cable_data
    
    


    def extract_structural_points(self):
        """
        Extracts the the structurla points (support element
        The relevant database key is  39/NR:0.
        
        :return: A list of dictionaries containing { 'point_number': ..., 'position': ... }.
        """
        points = []
        ie = c_int(0)

        # Instantiate the structure and its record length
        structural_point = CSPT()
        RecLen = c_int(sizeof(CSPT))
        j=-1
        # Loop until ie >= 2, indicating that we have reached the end of the records
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(
                self.Index,     # CDB index (set when calling open_cdb)
                39,            # Main group number = ?
                0,      # Subkey = 0
                byref(structural_point),   # Pointer to the structure to be filled
                byref(RecLen),  # Size of the structure
                1               # Mode = 1 => read
            )

            if ie.value == 0 and structural_point.m_id ==0:
                j+=1
                # No error, the structure is filled with valid data
                points.append({
                    "id": j,
                    "x": round(structural_point.m_xyz[0],2), "y":round(structural_point.m_xyz[1],2), "z":round(structural_point.m_xyz[2],2)
                })
                
            # Reset the record length before the next sof_cdb_get call
            RecLen = c_int(sizeof(CSPT))

        return points


def get_cable_end_points(cables, node_coords, view_axis='xy'):
    """
    Retrieves, for each cable, the 2D coordinates of its two endpoints (A and B).
    Assumes that `cables` is a list of dicts containing:
      {
        "segment_ids": [...],
        "node_chain": [n1, n2, ..., nN]
      }
    and that `node_coords[node] = (x, y, z)` for each node.

    :param cables: List of chained cables
    :param node_coords: dict { node_nr: (x, y, z) }
    :param view_axis: 'xy', 'xz', or 'yz'
    :return: A list of dicts, each in the format:
             {
               "A": (xA, yA),
               "B": (xB, yB),
             }
    """

    # Select horizontal (ax1) and vertical (ax2) axes based on the chosen plane
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map.get(view_axis, (0, 1))

    end_points = []

    for cable in cables:
        # Retrieve the node_chain: the ordered list of nodes
        node_chain = cable["node_chain"]  # e.g., [1009, 1146, 1056, ...]
        # The endpoints are the first and last nodes
        nA = node_chain[0]
        nB = node_chain[-1]

        # Get the 3D coordinates of node A
        xA_3d, yA_3d, zA_3d = node_coords[nA]
        # Get the 3D coordinates of node B
        xB_3d, yB_3d, zB_3d = node_coords[nB]

        # Project into the plane (view_axis)
        xA_2d = (xA_3d, yA_3d, zA_3d)[ax1]
        yA_2d = (xA_3d, yA_3d, zA_3d)[ax2]
        xB_2d = (xB_3d, yB_3d, zB_3d)[ax1]
        yB_2d = (xB_3d, yB_3d, zB_3d)[ax2]

        end_points.append({
            "A": (round(xA_2d, 2), round(yA_2d, 2)),
            "B": (round(xB_2d, 2), round(yB_2d, 2)),
        })

    return end_points


def chain_cable_segments(cable_elements):
    """
    Take in the list of cable segments (as returned by extract_cable), 
    and link them together into one or more 'chained' cables whenever 
    the end node of a segment matches the start node of the next.

    :param cable_elements: A list of dicts, e.g.:
        [
          {"cable_number": 20001, "nodes": (1009, 1146)},
          {"cable_number": 20002, "nodes": (1146, 1056)},
          ...
        ]
    :return: A list of dicts, each describing a complete chained cable:
        [
          {
            "segment_ids": [20001, 20002, ...],
            "node_chain":   [n1, n2, n3, ...]
          },
          ...
        ]
    """

    # Build a minimal adjacency dictionary for each 'cable_number' (segment)
    adjacency_dict = {}
    for elem in cable_elements:
        seg_id = elem["cable_number"]    # e.g., 20001
        nA, nB = elem["nodes"]           # e.g., (1009, 1146)
        adjacency_dict[seg_id] = {
            nA: [nB],
            nB: [nA]
        }

    # Sort the segment IDs and initialize variables for chaining
    segment_ids_sorted = sorted(adjacency_dict.keys())
    cables_chained = []
    current_chain_ids = []
    current_node_chain = []
    prev_end_node = None

    # Iterate through the segments in sorted order
    for seg_id in segment_ids_sorted:
        # Retrieve the two nodes of the segment (dictionary {nA:[nB], nB:[nA]})
        node_dict = adjacency_dict[seg_id]
        nA, nB = list(node_dict.keys())
        nB_ = node_dict[nA][0]  # 'end' of the segment assuming nA -> nB_

        if not current_chain_ids:
            # First segment of the current chain
            current_chain_ids = [seg_id]
            current_node_chain = [nA, nB_]
            prev_end_node = nB_
            continue

        # If the start of the current segment matches the end of the previous one, chain them
        if nA == prev_end_node:
            current_chain_ids.append(seg_id)
            current_node_chain.append(nB_)
            prev_end_node = nB_
        else:
            # Break: Close the previous chain
            cables_chained.append({
                "segment_ids": current_chain_ids,
                "node_chain": current_node_chain
            })
            # Start a new chain
            current_chain_ids = [seg_id]
            current_node_chain = [nA, nB_]
            prev_end_node = nB_

    # Don't forget to add the last chain if it exists
    if current_chain_ids:
        cables_chained.append({
            "segment_ids": current_chain_ids,
            "node_chain": current_node_chain
        })

    return cables_chained




def eigen_value_vector(a, b, c):
    """Eigenvalue and vector evaluation using Mohr's circle."""
    # Tolerance factor
    EPS = 1.e-6

    # tan(theta)
    if np.abs(a-c) <EPS:
        t = -1.
    else:
        m = 2.*b / (a-c)
        t = m / (1. + np.sqrt(1. + m**2))

    # First eigenvector
    u = 1. / np.sqrt(1. + t**2)
    v = t * u
    cs = np.array((u, v))

    # Eigenvalues lambda and mu
    u2 = u * u
    uv = u * v
    v2 = v * v

    lam = a*u2 + 2.*b*uv + c*v2
    mu = a*v2 - 2.*b*uv + c*u2

    return lam, mu, cs

def strut_compression_direction(nxx, nxy, nyy):
    """Strut compression direction. direction towards bottom """
    f1, f2, csp = eigen_value_vector(nxx, nxy, nyy)
    if f1 < f2:
        f = f1
        fp = f2
        if np.dot([0,-1],csp)>0:
            csp=-csp
        return f, csp, fp
    f = f2
    fp = f1
    cs = np.array((-csp[1], csp[0]))
    if np.dot([0,-1],cs)>0:
            cs=-cs
    return f, cs, fp

def perpendicular(v):
    """Renvoie un vecteur perpendiculaire en 2D."""
    return np.array([-v[1], v[0]])

    


# feuille efforts membranaires principaux
def compute_principal_membrane_forces(df_quads,quad_forces,nodes):

    node_dict = nodes.set_index('node_number').T.to_dict('list')

    # Filtrage des éléments : on ne garde que ceux dont **tous** les nœuds existent dans nodes

    main_quads_membrane_forces={}
    barycenters = {
    row['quad_number']: np.mean([node_dict[n] for n in [row['Node 1'], row['Node 2'], row['Node 3'], row['Node 4']]], axis=0)
        for _, row in df_quads.iterrows()}
    for _, row in quad_forces.iterrows():
            quad_id = row['quad_number']
            if quad_id not in barycenters:
                continue  # Évite les erreurs si un élément est absent

            # Récupération des efforts membranaires

            nx, ny, nxy = row['nx'], row['ny'], row['nxy']
            
            f,u,fp =  strut_compression_direction(nx,nxy,ny)
            up = perpendicular(u)
            main_quads_membrane_forces[int(quad_id)]=[u[0],u[1],up[0],up[1],f,fp]
            


    df_principal_membrane_forces = pd.DataFrame.from_dict(main_quads_membrane_forces, orient='index', columns=['ux','uy', 'upx','upy', 'f', 'fp']).reset_index()
    df_principal_membrane_forces.rename(columns={'index': 'quad_number'}, inplace=True)
    return(df_principal_membrane_forces)


#creating excel file  with membrane forces

def create_excel(title,sheets):
    with pd.ExcelWriter(title) as writer:
        for t,sheet in sheets : 
            sheet.to_excel(writer, sheet_name=t, index=False)

             # Accès à la feuille écrite pour ajuster la largeur des colonnes
            worksheet = writer.sheets[t]

            for idx, col in enumerate(sheet.columns):
                # Longueur max dans la colonne (y compris le nom de la colonne)
                max_len = max(
                    sheet[col].astype(str).map(len).max(),
                    len(col)
                )  # marge

                # Ajuster la largeur
                col_letter = chr(65 + idx)  # 65 = 'A'
                worksheet.column_dimensions[col_letter].width = max_len
            
def extract_contour(file_path,sheet_name,id,begin,columns):


# Lire l'Excel en utilisant pandas
    df = pd.read_excel(file_path, sheet_name, header=0, usecols=columns, skiprows=begin)
    contour=[]


    for index, row in df.iterrows():
        if pd.isna(row.iloc[0]) or row.iloc[0] == '':  # Vérifier si 'Bi' est vide ou NaN
            break  # Arrêter la boucle si une cellule 'Bi' vide est trouvée
        contour.append([row.iloc[i]for i in range (id,len(row))])  
    return(contour)


def read_excel_sheet(file_path, sheet_name):
    """
    Lit une feuille d'un fichier Excel et retourne son contenu sous forme de DataFrame.

    :param file_path: Chemin du fichier Excel.
    :param sheet_name: Nom de la feuille à lire.
    :return: DataFrame contenant les données de la feuille.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)

def plot_membrane_forces(quad_elements, quad_elements_forces,nodes,contour,excel_path):
    """
    Trace les vecteurs des efforts membranaires principaux en respectant leur direction.

    :param quad_elements: DataFrame avec colonnes ['quad_number', 'Node 1', 'Node 2', 'Node 3', 'Node 4']
    :param quad_elements_forces: DataFrame avec colonnes ['quad_number', 'nx', 'ny', 'nxy']
    :param nodes: DataFrame avec colonnes ['Node', 'x', 'y', 'z']
    :return: La figure  mise à jour
    """
    fig,ax=plt.subplots(figsize=(8,4))
    # Création du dictionnaire des coordonnées des nœuds pour accès rapide
    node_dict = nodes.set_index('node_number').T.to_dict('list')

     # Filtrage des éléments : on ne garde que ceux dont **tous** les nœuds existent dans nodes
    quad_elements = quad_elements[
        quad_elements.apply(lambda row: all(n in node_dict for n in [row['Node 1'], row['Node 2'], row['Node 3'], row['Node 4']]), axis=1)
    ]

    # Filtrage des forces en ne gardant que les quad_numbers qui existent encore
    valid_quad_ids = set(quad_elements['quad_number'])
    quad_elements_forces = quad_elements_forces[quad_elements_forces['quad_number'].isin(valid_quad_ids)]

    # Calcul des barycentres des quadrilatères
    barycenters = {
        row['quad_number']: np.mean([node_dict[n] for n in [row['Node 1'], row['Node 2'], row['Node 3'], row['Node 4']]], axis=0)
        for _, row in quad_elements.iterrows()
    }


    # Détermination de l'effort principal maximal
    max_force = quad_elements_forces['f'].abs().max()
    
    # Fixer une longueur de référence pour le plus grand vecteur
    structure_size = 5.16
    max_vector_length = 0.1 * structure_size  # 10% de la taille de la structure

    # Tracé des vecteurs d'efforts principaux
    for _, row in quad_elements_forces.iterrows():
        quad_id = row['quad_number']

        if quad_id not in barycenters:
            continue  # Évite les erreurs si un élément est absent

        u,up,f,fp = [row["ux"],row["uy"]],[row['upx'],row['upy']],row['f'],row['fp']

        bx, _, bz = barycenters[quad_id]
        bz=bz
        val1,val2 = (f / max_force) * max_vector_length, (fp / max_force) * max_vector_length
        
        # Calcul des points de départ et d'arrivée du vecteur
        x_start, y_start = bx,  bz
        #traction 
        #x_end, y_end = x_start  + val2 * up[0], y_start +   val2 * up[1]
        x_end, y_end = x_start + val1 * u[0] , y_start + val1 * u[1]     

        # Ajout du vecteur au graphique avec une flèche
        ax.plot(
            [x_start, x_end],
            [y_start, y_end], color='r', linewidth=0.8

        )

        #tracer contour 
    for i in range(len(contour)):
        ax.plot([contour[i-1][0],contour[i][0]], [contour[i-1][1],contour[i][1]], color='black',linestyle='-',linewidth=1,alpha=0.8, zorder=3)


    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
 
    ax.set_title(f"Champ des efforts",fontsize=10)
    fig.savefig(f"{excel_path}\\Images\\Champ_des_efforts",bbox_inches='tight', pad_inches=0.2)
    #plt.show()
  

def plot_cables_forces(quad_elements,nodes,excel_path,contour,cables,cables_forces,bounds):
    """
        Trace les cables avec les efforts dans ces cables.

        :param quad_elements: DataFrame avec colonnes ['quad_number', 'Node 1', 'Node 2', 'Node 3', 'Node 4']
        :param quad_elements_forces: DataFrame avec colonnes ['quad_number', 'nx', 'ny', 'nxy']
        :param nodes: DataFrame avec colonnes ['Node', 'x', 'y', 'z']
        :param cables : 
        :return: La figure mise à jour
        """
    fig,ax=plt.subplots(figsize=(8,4))
    # Création du dictionnaire des coordonnées des nœuds pour accès rapide
    node_dict = nodes.set_index('node_number').T.to_dict('list')

     # Filtrage des éléments : on ne garde que ceux dont **tous** les nœuds existent dans nodes
    quad_elements = quad_elements[
        quad_elements.apply(lambda row: all(n in node_dict for n in [row['Node 1'], row['Node 2'], row['Node 3'], row['Node 4']]), axis=1)
    ]


    # Fixer une longueur de référence pour le plus grand vecteur
    scale=0.25*(bounds[5]-bounds[2])/max([cables_forces[i]["force"] for i in range(len(cables_forces))])

    # Tracé des vecteurs d'efforts principaux
    for idx, row in enumerate(cables):
        force=cables_forces[idx]["force"]
        node1,node2 = row['nodes']

        x1,z1=node_dict[node1][0],node_dict[node1][2]
        x2,z2=node_dict[node2][0],node_dict[node2][2]
        u=[x2-x1,z2-z1]
        n=-perpendicular(u)
        n/=np.linalg.norm(n)
        #afficher la force
        ax.plot(
            [x1, np.dot(x1+ n * force*scale,[1,0]),np.dot(x2+ n * force*scale,[1,0]),x2],
             [z1, np.dot(z1+ n * force*scale,[0,1]),np.dot(z2+ n * force*scale,[0,1]),z2], color='b', linewidth=0.8

        )
        #afficher le cable
        ax.plot(
            [x1,x2],
             [z1,z2], color='lime', linewidth=1.5

        )


    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
    #tracer contour 
    for i in range(len(contour)):
        ax.plot([contour[i-1][0],contour[i][0]], [contour[i-1][1],contour[i][1]], color='black',linestyle='-',linewidth=1,alpha=0.8, zorder=3)

    ax.set_title(f"Force dans les barres d'acier",fontsize=10)
    fig.savefig(f"{excel_path}\\Images\\Forces_dans_les_cables",bbox_inches='tight', pad_inches=0.2)
    #plt.show()
  

def main(name,file_path,excel_path,main_file,sofistik_exe,dat_file_name):

    dat_file= Path(dat_file_name)
    subprocess.run([
     sofistik_exe,str(dat_file) ],  stdout=subprocess.DEVNULL)
  
    if sofistik_failed(dat_file.with_suffix(".prt"), dat_file.with_suffix(".err")):
        print_last_errors(dat_file.with_suffix(".prt"), n=20)
        raise RuntimeError("erreur dans l'exécution Sofistik, voir les erreurs directement")
    
    # Example of usage
    cdb_manager = CDBData()
    try:
        cdb_manager.open_cdb(file_path)

        quad_membrane_forces = cdb_manager.extract_quad_membrane_forces()
        quad_elements = cdb_manager.extract_quad_elements()
        nodes = cdb_manager.extract_node_coords()
        cables = cdb_manager.extract_cable()
        cables_force=cdb_manager.extract_cable_forces()
      

        bounds = cdb_manager.extract_structure_bounds()
        supports = cdb_manager.extract_structural_points()
    
    finally:
        cdb_manager.close_cdb()

  
    #to DataFrame
    df_nodes = pd.DataFrame.from_dict(nodes, orient='index', columns=['x', 'y', 'z']).reset_index()
    df_nodes.rename(columns={'index': 'node_number'}, inplace=True)
    df_quads = pd.DataFrame.from_dict(quad_elements, orient='index', columns=['Node 1', 'Node 2', 'Node 3', 'Node 4']).reset_index()
    df_quads.rename(columns={'index': 'quad_number'}, inplace=True)
    df_principal_membrane_forces=compute_principal_membrane_forces(df_quads,quad_membrane_forces,df_nodes)
   
    contour=extract_coord(main_file,"TMP_BIELLES",0,0,"AA:AB")
    plot_membrane_forces(df_quads,df_principal_membrane_forces,df_nodes,contour,excel_path)
    plot_cables_forces(df_quads,df_nodes,excel_path,contour,cables,cables_force,bounds)

    #create excel
    sheets=[["Nodes_coords",df_nodes],["Quad_elements",df_quads],["Forces",quad_membrane_forces],["Principal_membrane_forces",df_principal_membrane_forces]]
    create_excel(f"{excel_path}\\Python\\{name}.xlsx",sheets)
    print(f"Le fichier Sofistik a été exporté : {name}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name", help="Nom du fichier excel")
    parser.add_argument("file_path", help="Chemin du fichier sofistik à traiter")
    parser.add_argument("excel_path", help="Chemin du fichier excel créé")
    parser.add_argument("main_file", help="Chemin du fichier excel principal")
    parser.add_argument("sofistik_exe", help="executable sofistik")
    parser.add_argument("dat_file_name", help="nom du fichier .dat")

    args = parser.parse_args()


main( args.name,args.file_path,args.excel_path,args.main_file,args.sofistik_exe,args.dat_file_name)