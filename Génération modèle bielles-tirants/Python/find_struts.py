"""
Filename: classification_bielles.py
Author: Arthur de Bigault de Granrut
Created: 2025-11-17
Description: This script detects clusters in concrete pieces, that are potentially struts
"""

#================== libraries to import =====================

from ctypes import * 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import argparse
import os
from pathlib import Path
import xlwings as xw
#=============================================================


def extract_data(file_path, sheet_name):
    """
    Lit une feuille d'un fichier Excel et retourne son contenu sous forme de DataFrame.

    :param file_path: Chemin du fichier Excel.
    :param sheet_name: Nom de la feuille à lire.
    :return: DataFrame contenant les données de la feuille.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def extract_coord(file_path,sheet_name,id,begin,columns):


# Lire l'Excel en utilisant pandas
    df = pd.read_excel(file_path, sheet_name, header=0, usecols=columns, skiprows=begin)
    data=[]


    for index, row in df.iterrows():
        if pd.isna(row.iloc[0]) or row.iloc[0] == '':  # Vérifier si la cellule est vide ou NaN
            break  # Arrêter la boucle si une cellulevide est trouvée
        data.append([row.iloc[i]for i in range (id,len(row))])  
    return(data)

    
def find_index(lst, x):
    try:
        return True,lst.index(x)
    except ValueError:
        return False,0

def get_barycenters(df_quad_elements,nodes):
    node_dict = nodes.set_index('node_number').T.to_dict('list')
    barycenters = {
        row['quad_number']: np.mean([node_dict[n] for n in [row['Node 1'], row['Node 2'], row['Node 3'], row['Node 4']]], axis=0)
        for _, row in df_quad_elements.iterrows()}
    return(barycenters)



def data_to_features(df,barycenters,boundaries):
    features=[]
    ymax=boundaries[-1]
    for _, row in df.iterrows():
        
        quad_id = int(row['quad_number'])
        if quad_id not in barycenters:
            continue  # Évite les erreurs si un élément est absent

        u,f = [row["ux"],row["uy"]],row['f']
        
        
        
        #[x,y,theta,f]
        
        feature=[(barycenters[quad_id][0]),barycenters[quad_id][2],u[0],u[1],f ]
        if not np.isnan(feature).any():
            features.append(feature)
    return(np.array(features))


def init_scatter(x,eps,N,boundaries):
    xmin,xmax,ymin,ymax=boundaries

    points = []
    for _ in range(N):
        # random theta
        angle = np.random.uniform(0, 2 * np.pi)
        
        # random distance
        r = np.random.uniform(0, eps)
        
        # Calculer les coordonnées du point autour de X
        x_point = x[0] + r * np.cos(angle)
        y_point = x[1] + r * np.sin(angle)
        x_point = np.clip(x_point, xmin, xmax)
        y_point = np.clip(y_point, ymin, ymax)

        # Ajouter le point généré à la liste
        points.append([x_point, y_point])
    
    return points

# Defintion of vector field
def vector_field(x, y, u_interpolator,v_interpolator):
    """Return veloctiy U,V at position x,y with interpolation"""
    return u_interpolator((y, x)), v_interpolator((y, x))

# constraints zone
def is_in_zone(x, y, zone_centers, radius,j):
    """Check if the point (x, y) is within a predefined zone."""
    for id in range(len(zone_centers)):
        cx, cy=zone_centers[id]
        if id!=j and (x - cx)**2 + (y - cy)**2 <= radius**2:
            return id,True
    return -1,False

# Implementation of RK4
def rk4_step(x, y, dt, u_interpolator,v_interpolator):
    """Integration step with RK4 """

    k1u, k1v = vector_field(x, y, u_interpolator,v_interpolator)
    k2u, k2v = vector_field(x + 0.5*dt*k1u, y + 0.5*dt*k1v,u_interpolator,v_interpolator)
    k3u, k3v = vector_field(x + 0.5*dt*k2u, y + 0.5*dt*k2v, u_interpolator,v_interpolator)
    k4u, k4v = vector_field(x + dt*k3u, y + dt*k3v,u_interpolator,v_interpolator)
    
    # Compute new position
    new_x = x + (dt/6)*(k1u + 2*k2u + 2*k3u + k4u)
    new_y = y + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
    
    return new_x, new_y

def is_in_boundaries(x,y,boundaries):

    return(x>=boundaries[0] and x <=boundaries[1] and y >=boundaries[2] and y <= boundaries[3] )

def polygon_orientation(points):
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x2 - x1) * (y2 + y1)
    return np.sign(area)

def inward_normal(p1, p2, sign):
    edge = np.array(p2) - np.array(p1)
    edge = edge / np.linalg.norm(edge)


    normal = np.array([sign*edge[1], -1*sign*edge[0]])   
    normal/=np.linalg.norm(normal)
    return normal

def get_nodes(ties,nodes_support,loads,model,delta):
    sign=polygon_orientation(model)
    #normals towards the figure
    normals = []
    for i in range(len(model)-1):
        p1 = model[i]
        p2 = model[(i + 1) % len(model)]
        n = inward_normal(p1, p2, sign)

        normals.append(n)
    nodes=[]
    #nodes with reinforcement
    eps=1.e-3
    for node in nodes_support :
        for i in range(len(model)-1):
            p1 = np.array(model[i])
            p2 = model[(i + 1) % len(model)]
            u=node-p1
            v=p2-p1
            u/=np.linalg.norm(u)
            v/=np.linalg.norm(v)

            if abs(np.dot(u,v)-1)<eps or abs(np.dot(u,v)+1)<eps:
                nodes.append(node+normals[i]*delta)
        

    for load in loads :
        for i in range(len(model)-1):
            p1 = np.array(model[i])
            p2 = model[(i + 1) % len(model)]
            u=load-p1
            v=p2-p1
            u/=np.linalg.norm(u)

            v/=np.linalg.norm(v)

            if abs(np.dot(u,v)-1)<eps or abs(np.dot(u,v)+1)<eps:

                nodes.append(load+normals[i]*delta)

    



    for i in range(len(ties)-1):
        for j in range(i+1,len(ties)):
            u1=np.array([ties[i][1][0]-ties[i][0][0],ties[i][1][1]-ties[i][0][1]])
            u2=np.array([ties[j][1][0]-ties[j][0][0],ties[j][1][1]-ties[j][0][1]] )

            u1/=np.linalg.norm(u1)
            u2/=np.linalg.norm(u2)       
            #test of non colinearity
            
            if abs(np.dot(u1,u2)-1)> 1.e-4 and abs(np.dot(u1,u2)+1)> 1.e-4 :
                A=np.array([[u1[0],-u2[0]],[u1[1],-u2[1]]])
                B=np.array([[ties[j][0][0]-ties[i][0][0]],[ties[j][0][1]-ties[i][0][1]]])
                X=np.linalg.solve(A,B)
                
                x=ties[i][0]+X[0]*u1
                nodes.append(x)

    #nodes with intersection between reinfocrement and support

    return(np.unique(nodes, axis=0))


def get_truss(reinforcements_segments,potential_nodes):
    nodes=[]
    segments=[]
    for seg in reinforcements_segments :
            nodes_on_seg=nodes_on_reinforcement(seg,potential_nodes)
            nodes+=nodes_on_seg
            for i in range(len(nodes_on_seg)-1):
                segments.append([nodes_on_seg[i],nodes_on_seg[i+1]])
            
    return(np.unique(nodes,axis=0),segments)         

def compute_trajectory(boundaries,dt,end_zones,radius,seeds,grid_x,grid_y,grid_u,grid_v,u_interpolator,v_interpolator,j,plot_b=False):
    ends=np.zeros(len(end_zones)+1)
    pltx,plty=[],[]
    for seed in seeds:
        x_start, y_start = seed
        x_vals, y_vals = [x_start], [y_start]
        while True :
            x_next, y_next = rk4_step(x_vals[-1], y_vals[-1], dt,u_interpolator, v_interpolator)

            # Vérifier si la trajectoire est dans une zone de contrainte
            i,stop=is_in_zone(x_next, y_next,end_zones ,radius,j)
            if stop :
                ends[i]+=1
                break  # Arrêter si la ligne touche une zone
            if not is_in_boundaries(x_next,y_next,boundaries):
                ends[-1]+=1
                break
            # Ajouter la position à la trajectoire
            x_vals.append(x_next)
            y_vals.append(y_next)
        pltx.append(x_vals)
        plty.append(y_vals)
    if plot_b :
        for i in range (len(pltx)):
            plt.plot(pltx[i],plty[i],color='r')
        plt.streamplot(grid_x, grid_y, grid_u, grid_v, color='grey', linewidth=0.7)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
    return(ends/len(seeds),pltx,plty)

def get_connections(field,boundaries,potential_nodes):
    X, Y = field[:,0], field[:,1]
    U = field[:,2]
    V = field[:,3]
    # Controlling the starting points of the streamlines
    
    grid_x, grid_y = np.meshgrid(np.linspace(boundaries[0], boundaries[1], 100), np.linspace(boundaries[2],boundaries[3], 100))
    
    grid_u = interpolate.griddata((X, Y), U, (grid_x, grid_y), method='nearest')
    grid_v = interpolate.griddata((X, Y), V, (grid_x, grid_y), method='nearest')
    u_interpolator = RegularGridInterpolator((grid_y[:,0], grid_x[0,:]),grid_u,fill_value=0.0,bounds_error=False)
    v_interpolator = RegularGridInterpolator((grid_y[:,0], grid_x[0,:]),grid_v,fill_value=0.0,bounds_error=False)
    N=50
    eps=0.15*(boundaries[3]-boundaries[2])
    dt=0.01
    connections=[]
    xvalues,yvalues=[],[]
    for id,x in  enumerate(potential_nodes):
        seeds=init_scatter(x,eps,N,boundaries)
        seeds=np.array(seeds)
        
        trajectory,pltx,plty=compute_trajectory(boundaries,dt,potential_nodes,eps,seeds,grid_x,grid_y,grid_u,grid_v,u_interpolator,v_interpolator,id,False)
        connections.append(trajectory)
        xvalues+=pltx
        yvalues+=plty

    
    connections= [[1 if connections[i][j]>=1/(len(potential_nodes)+1)  else 0 for j in range(len(potential_nodes))] for i in range(len(potential_nodes))]
    
    return(connections,xvalues,yvalues)

    
def plot_stream_line(path_images,field,xvalues,yvalues,contour,boundaries,potential_nodes):
    X, Y = field[:,0], field[:,1]
    U = field[:,2]
    V = field[:,3]
    # Controlling the starting points of the streamlines
    eps=0.15*(boundaries[3]-boundaries[2])

    


    grid_x, grid_y = np.meshgrid(np.linspace(boundaries[0], boundaries[1], 100), np.linspace(boundaries[2],boundaries[3], 100))

    grid_u = interpolate.griddata((X, Y), U, (grid_x, grid_y), method='nearest')
    grid_v = interpolate.griddata((X, Y), V, (grid_x, grid_y), method='nearest')
    fig, ax = plt.subplots(figsize=(8, 6))
    

    for i in range(len(contour)):
        ax.plot([contour[i-1][0],contour[i][0]], [contour[i-1][1],contour[i][1]], color='black',linestyle='-',linewidth=1,alpha=0.8, zorder=1)
    plt.streamplot(grid_x, grid_y, grid_u, grid_v, color='grey', linewidth=1)
  
    #afficher les lignes
    for i in range(len(xvalues)):
        ax.plot(xvalues[i],yvalues[i],color='r',linewidth=0.7,alpha=0.8)
    plt.title("Lignes de courant")

    for x in potential_nodes:
        circle = plt.Circle((x[0], x[1]), eps, color='b', fill=False,zorder=2)
        ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')

    #plt.show()
    
    fig.savefig(f"{path_images}\\mbt_lignes_de_courant",bbox_inches='tight', pad_inches=0.2)

def nodes_on_reinforcement(segment,nodes):
    a,b=np.array(segment[0]),segment[1]
    u=b-a
    u/=np.linalg.norm(u)
    nodes_on_segment=[]
    for node in nodes : 
        v=node-a
        if abs(np.dot(u,v)-np.linalg.norm(v)) < 1.e-3:
            nodes_on_segment.append([np.linalg.norm(v),node])

    nodes_on_segment.sort(key=lambda x: x[0])  
    
    return([node[1] for node in  nodes_on_segment] )

def plot_stm(file,graph_connections,contour,reinforcements_segments,potential_nodes,path_images):
    fig,ax=plt.subplots(figsize=(8, 6))
    for i in range(len(contour)):
        ax.plot([contour[i-1][0],contour[i][0]], [contour[i-1][1],contour[i][1]], color='black',linestyle='-',linewidth=1,alpha=0.8, zorder=1)


    for i in range(len(graph_connections)):
        for j in range(len(graph_connections[0])):
            if graph_connections[i][j]==1:
                ax.plot([potential_nodes[i][0], potential_nodes[j][0]], [potential_nodes[i][1], potential_nodes[j][1]], color='black',linestyle='--',linewidth=1.5,alpha=1, zorder=3)
    for seg in reinforcements_segments :
            nodes_on_seg=nodes_on_reinforcement(seg,potential_nodes)
            shorter_seg=[nodes_on_seg[0],nodes_on_seg[-1]]
            ax.plot([shorter_seg[0][0],shorter_seg[1][0]],[shorter_seg[0][1],shorter_seg[1][1]], color='black',linestyle='-',linewidth=1.5,alpha=1, zorder=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    for node in potential_nodes :
        ax.scatter(node[0],node[1],color='black',s=10,zorder=3)

    file_name = os.path.splitext(os.path.basename(file))[0]

    plt.title(f"Treillis du modèle")
    #plt.show()
    fig.savefig(f"{path_images}\\modèle_treillis",bbox_inches='tight', pad_inches=0.2)



def main(file_name,main_file):
    #data extraction
    df_forces = extract_data(file_name, "Principal_membrane_forces")
    df_positions=extract_data(file_name, "Nodes_coords")
    df_quad_elements=extract_data(file_name, "Quad_elements")
    reinforcements_segments_vert=extract_coord(main_file,"TMP_BIELLES",0,0,"N:S" )
    reinforcements_segments_hor=extract_coord(main_file,"TMP_BIELLES",0,0,"T:Y" )
    reinforcements_segments =  [[[p[1], p[2]], [p[3], p[4]]] for p in reinforcements_segments_vert + reinforcements_segments_hor if p[5] == "oui"]
    #feature embodiement
    barycenters=get_barycenters(df_quad_elements,df_positions)
    #distance aux bords du modèle
    delta=pd.read_excel(main_file, sheet_name="Input data", header=None, usecols="C", skiprows=14,nrows=1).iloc[0,0]
    #adapt the boundary fior 3d case
    contour=extract_coord(main_file,"TMP_BIELLES",0,0,"AA:AB")

    xmin,ymin=np.min(contour, axis=0)
    xmax,ymax=np.max(contour, axis=0)
    boundaries=[xmin,xmax,ymin,ymax]

    loads=extract_coord(main_file,"TMP_BIELLES",0,0,"C:D")

    #support nodes 
    supports=extract_coord(main_file,"TMP_BIELLES",0,0,"I:J")

    
    potential_nodes=get_nodes(reinforcements_segments,supports,loads,contour,delta)
   
    path_images= f"{Path(file_name).absolute().parent.parent}\\Images"
    print("Calcul des lignes de courant")
    X=data_to_features(df_forces,barycenters,boundaries)
    
    print("Tests des bielles")
    graph_connections,xvalues,yvalues=get_connections(X,boundaries,potential_nodes)
    plot_stream_line(path_images,X,xvalues,yvalues,contour,boundaries,potential_nodes)
    plot_stm(file_name,graph_connections,contour,reinforcements_segments,potential_nodes,path_images)  

    #rajouter les noeuds utilisés dans le fichier excel

    kept_nodes,kept_segments=get_truss(reinforcements_segments,potential_nodes)     
    wb = xw.Book(main_file)
    ws = wb.sheets["TMP_BT"]

    ws.range("2:1048576").clear_contents()

    

    #trouver les noeuds correspondant aux pieux et aux supports
    load_i=[]
    for load in loads :
        distance_nodes=[np.linalg.norm(load-kept_nodes[i]) for i in range(len(kept_nodes))]
        id=np.argmin(distance_nodes)
        #s'assurer qu'il s'agit bien de 2 noeuds identiques 
        if distance_nodes[id]<=(ymax-ymin)*0.15:
            load_i.append(id)

    support_i=[]
    for support in supports :
        distance_nodes=[np.linalg.norm(support-kept_nodes[i]) for i in range(len(kept_nodes))]
        id=np.argmin(distance_nodes)

        #s'assurer qu'il s'agit bien de 2 noeuds identiques 
        if distance_nodes[id]<=(ymax-ymin)*0.15:
            support_i.append(id)
    row = 2
    for i in range(len(kept_nodes)):

        x1, y1 = kept_nodes[i][0],kept_nodes[i][1]
        ws.range((row, 2)).value= x1
        ws.range((row, 3)).value= y1
        
        b,j=find_index(support_i,i)
        if b:
         
            ws.range((row, 5)).value= j+1
            #ws.range((row, 4)).value= 0
        else:
            b,j=find_index(load_i,i)
            if b :
                ws.range((row, 4)).value= j+1
                #ws.range((row, 5)).value= 0
            else :
                pass 
                #ws.range((row, 4)).value= 0
                #ws.range((row, 5)).value= 0


        row += 1
    
    dict_nodes={}


    #dictionnaire pour trouver l'id des noeuds
    for i in range(len(kept_nodes)):
        dict_nodes[tuple(kept_nodes[i])]=i+1
    row=2
    #barre d'armature
    for i in range(len(kept_segments)):
        nodes = kept_segments[i]
        ws.range((row, 6)).value= dict_nodes[tuple(nodes[0])]
        ws.range((row, 7)).value= dict_nodes[tuple(nodes[1])]
        row += 1
    #barre modélisant une bielle
    n,m=len(graph_connections),len(graph_connections[0])
    for i in range (n):
        for j in range (m):
            if graph_connections[i][j]==1 : 
                ws.range((row, 6)).value= i+1
                ws.range((row, 7)).value= j+1
                row += 1
    
    print("Fin du calcul des bielles")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("file_name", help="Nom du fichier à traiter")
    parser.add_argument("main_file", help="Nom du fichier avec les pieux et les supports")


    args = parser.parse_args()

    
    main(args.file_name,args.main_file)


