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
import xlwings as xw
from openpyxl.drawing.image import Image
from find_struts import extract_coord
# ⚠️⚠️⚠️ WARNING: CHECK THE PATH ⚠️⚠️⚠️
from sofistik_daten import * 


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



   
    def extract_truss(self, load_case=1):
        """
        Extracts the cable number (NR) and the cable force (N) for a given load case. Extract the cable effective stiffness too
        The relevant database key is 162/<load_case> (e.g., 162/1, 162/2, etc.).
        
        :param load_case: The load case number to read (default = 1).
        :return: A list of dictionaries containing { 'cable_number': ..., 'force': ... }.
        """
        
        truss_data = []
        ie = c_int(0)

        # Instantiate the structure and its record length
        truss =  CTRUS_RES()
        RecLen = c_int(sizeof(CTRUS_RES))

        # Loop until ie >= 2, indicating that we have reached the end of the records
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(
                self.Index,     # CDB index (set when calling open_cdb)
                152,            # Main group number = 152
                load_case,      # Subkey = load_case
                byref(truss),   # Pointer to the structure to be filled
                byref(RecLen),  # Size of the structure
                1               # Mode = 1 => read
            )

            if ie.value == 0:
                # No error, the structure is filled with valid data
                truss_data.append({
                    "truss_number": truss.m_nr,
                    "normal_force":        round(truss.m_n,2),
                    "axial_displacment" : round(truss.m_v,2)
                })
               

            # Reset the record length before the next sof_cdb_get call
            RecLen = c_int(sizeof(CTRUS_RES))

        return truss_data
    
    def extract_truss_nodes(self, load_case=1):
   
        
        truss_nodes_data = []
        ie = c_int(0)
       
        # Instantiate the structure and its record length
        truss =  CTRUS()
        RecLen = c_int(sizeof(CTRUS))

        # Loop until ie >= 2, indicating that we have reached the end of the records
        while ie.value < 2:
            ie.value = self.myDLL.sof_cdb_get(
                self.Index,     # CDB index (set when calling open_cdb)
                150,            # Main group number = 150
                0,      # Subkey = load_case
                byref(truss),   # Pointer to the structure to be filled
                byref(RecLen),  # Size of the structure
                1               # Mode = 1 => read
            )
            
            if ie.value == 0:
                # No error, the structure is filled with valid data
                truss_nodes_data.append({
                    "truss_number": truss.m_nr,
                    "nodes":       [truss.m_node[0],truss.m_node[1]],

                })
               

            # Reset the record length before the next sof_cdb_get call
            RecLen = c_int(sizeof(CTRUS))

        return truss_nodes_data
    
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



def add_forces(file_path,truss):
    wb = xw.Book(file_path)
    ws = wb.sheets["TMP_BT"]

    ws.range("H2:H1048576").clear_contents()

    for idx, row in enumerate(truss):
            force=row["normal_force"]
            ws.range((idx+2, 8)).value= force
def polygon_orientation(points):
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x2 - x1) * (y2 + y1)
    return np.sign(area)


def get_side(x,y,contour):
    """
    trouver la face sur laquelle est située le point 
    
    :param x: abcisse du point
    :param y: ordonnée du point
    :param contour: contour 
    """
    eps=1.e-5 #tolérance
    for i in range(len(contour)-1):
        p1,p2=np.array(contour[i]),contour[i+1]
        p3=np.array([x,y])
        # test de l'alignement des points
        v1,v2=p3-p1,p2-p1
        cross_product=v1[0]*v2[1]-v1[1]*v2[0]
        if abs(cross_product)<= eps :  
            # test p3 entre p1 et p2 
            if np.dot(p3-p1,p3-p2) <= eps :
                return(i)
    raise Exception("noeud non trouvé sur le contour")
    

def inward_normal(p1, p2, sign):
    """
    trouver la normale du segment p1,p2 orientée vers l'intérieur du polygone
    
    :param p1: point 1
    :param p2: point 2 
    :param sign: polygone
    """
    edge = np.array(p2) - np.array(p1)
    edge = edge / np.linalg.norm(edge)
    normal = np.array([sign*edge[1], -1*sign*edge[0]])   
    normal/=np.linalg.norm(normal)
    return normal

def get_normals(model,sign):
    normals = []
    for i in range(len(model)-1):
        p1 = model[i]
        p2 = model[(i + 1) % len(model)]
        n = inward_normal(p1, p2, sign)

        normals.append(n)
    return(normals)
def perpendicular(v):
    """returns a direct orthogonal vector in 2d by rotating the input of 90°

    Args:
        v (list): vector in 2d

    Returns:
        list: vector in 2d (perpendicular to the entrance)
    """
    return np.array([v[1], -v[0]])
  
def plot_model_a_barres(truss,truss_node,nodes,contour,excel_path):
    """
        Trace les cables avec les efforts dans ces cables.

        :param truss: DataFrame avec colonnes ['force', 'displacement']
        :param nodes: DataFrame avec colonnes ['Node', 'x', 'y', 'z']

        :return: La figure mise à jour
        """
    fig,ax=plt.subplots(figsize=(8,4))
    # Création du dictionnaire des coordonnées des nœuds pour accès rapide
    

    # Fixer une longueur de référence pour le plus grand vecteur

    #tracer les noeuds 

    for node in nodes.values() :
        ax.scatter(node[0],node[2],color='black',s=10,zorder=3)

    #tracer contour 
    for i in range(len(contour)):
        ax.plot([contour[i-1][0],contour[i][0]], [contour[i-1][1],contour[i][1]], color='black',linestyle='-',linewidth=1,alpha=0.8, zorder=3)



    #tracer forces
    input=pd.read_excel(excel_path, sheet_name="Input Sofistik", header=None, usecols="C", skiprows=4,nrows=4)
    n_supports=input.iloc[0,0]
    n_pieux=input.iloc[1,0]

    pieux=pd.read_excel(excel_path, sheet_name="TMP_BIELLES", header=None, usecols="C:D", skiprows=1,nrows=n_pieux)
    charge=pd.read_excel(excel_path, sheet_name="TMP_BIELLES", header=None, usecols="AG:AI", skiprows=2,nrows=n_pieux)
    sign = polygon_orientation(contour)
    ymax=max([contour[i][1] for i in range(len(contour))])
    ymin=min([contour[i][1] for i in range(len(contour))])
    xmax=max([contour[i][0] for i in range(len(contour))])
    xmin=min([contour[i][0] for i in range(len(contour))])
    normals= get_normals(contour,sign)
    s=(ymax-ymin)*0.3
    for i in range(len(pieux)):
        axial_force=charge.iloc[i,0]
        shear_force=charge.iloc[i,1]
        x,y=pieux.iloc[i,0],pieux.iloc[i,1]
        # --- position du noeud sur le contour
        i_side= get_side(x,y,contour) 
        normal=normals[i_side]
        p=perpendicular(normal)
        m=max([abs(axial_force),abs(shear_force)])
        
        if m!=0 : 
            
            u=shear_force/m*p +axial_force/m*normal
            
            if np.dot(u,normal)>=0 :
               arrow='->'
               sign=1
            else : 
                arrow='<-'
                sign=-1
            ax.annotate("", 
                            xy=[x,y],xytext=[x+sign*s*u[0],y+sign*s*u[1]],
                            arrowprops=dict(
                                arrowstyle=arrow, color='grey', lw=1,mutation_scale=15),ha='center'
                            )
    # Tracé des vecteurs d'efforts principaux
    for idx, row in enumerate(truss):
        force=row["normal_force"]
        node1,node2=truss_node[idx]["nodes"]

        x1,z1=nodes[node1][0],nodes[node1][2]
        x2,z2=nodes[node2][0],nodes[node2][2]
        xm,zm=(x1+x2)/2,(z1+z2)/2
        dir=[x2-x1,z2-z1]
        dir/=np.linalg.norm(dir)
        if dir[0]<0 :
            dir=-dir
        rot=np.atan2(dir[1],np.dot([1,0],dir))*180/np.pi
           
        if force<0 : #compression
        #afficher la force
            ax.plot(
                [x1,x2],
                [z1,z2], color='r',linestyle='--', linewidth=1.5, zorder=1

            )
            ax.text(xm, zm , str(force), color="r",
            fontsize=8, ha="center",va="center",rotation=rot, bbox=dict(facecolor='w',edgecolor='w'),zorder=2)
        elif force>0 : #traction
            ax.plot(
                [x1,x2],
                [z1,z2], color='b',linestyle='-', linewidth=1.5, zorder=1
            )
            ax.text(xm, zm , str(force), color="b",
            fontsize=8, ha="center",va="center", bbox=dict(facecolor='w',edgecolor='w'),zorder=2)


    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    margin=0.20
    alpha=min((xmax-xmin),ymax-ymin)
    plt.xlim(xmin-margin*alpha,xmax+margin*alpha)
    plt.ylim(ymin-1.2*margin*alpha,ymax+1.2*margin*alpha)
    

    path_images= Path(excel_path).absolute().parent
    ax.set_title(f"Modèle_bielles_tirants",fontsize=10)
    fig.savefig(f"{path_images}\\Images\\Modèle_bielles_tirants",bbox_inches='tight', pad_inches=0.2)
    #plt.show()

def main(name,file_path,excel_path,sofistik_exe,dat_file_name):
    print("Ouverture du fichier sofisitk")
    dat_file= Path(dat_file_name)
    print("Calcul des forces dans le modèle à barres")
    subprocess.run([
     sofistik_exe,str(dat_file) ],  stdout=subprocess.DEVNULL)
  
    if sofistik_failed(dat_file.with_suffix(".prt"), dat_file.with_suffix(".err")):
        print_last_errors(dat_file.with_suffix(".prt"), n=20)
        raise RuntimeError("erreur dans l'exécution Sofistik, voir les erreurs directement")
    
    # Example of usage
    cdb_manager = CDBData()
    try:
        cdb_manager.open_cdb(file_path)

        truss = cdb_manager.extract_truss()
        truss_nodes=cdb_manager.extract_truss_nodes()
        nodes = cdb_manager.extract_node_coords()
        

    finally:
        cdb_manager.close_cdb()

    contour=extract_coord(excel_path,"TMP_BIELLES",0,0,"AA:AB")
    add_forces(excel_path,truss)
    print("Finalisation modèle bielles - tirants")
    plot_model_a_barres(truss,truss_nodes,nodes,contour,excel_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name", help="Nom du fichier excel")
    parser.add_argument("file_path", help="Chemin du fichier sofistik à traiter")
    parser.add_argument("excel_path", help="Chemin du fichier excel principal")
    parser.add_argument("sofistik_exe", help="executable sofistik")
    parser.add_argument("dat_file_name", help="nom du fichier .dat")

    args = parser.parse_args()


main( args.name,args.file_path,args.excel_path,args.sofistik_exe,args.dat_file_name)