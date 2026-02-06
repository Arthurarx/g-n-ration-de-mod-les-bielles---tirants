import pandas as pd
import csv
import numpy as np
import argparse
import os

import xlwings as xw
from openpyxl.drawing.image import Image



def extract_contour(file_path,sheet_name,id,begin,columns):


# Lire l'Excel en utilisant pandas
    df = pd.read_excel(file_path, sheet_name, header=0, usecols=columns, skiprows=begin)
    contour=[]
    if df.empty or df.dropna(how="all").empty:
            return []

    for index, row in df.iterrows():
        if pd.isna(row.iloc[0]) or row.iloc[0] == '':  # Vérifier si 'Bi' est vide ou NaN
            break  # Arrêter la boucle si une cellule 'Bi' vide est trouvée
        contour.append([row.iloc[i]for i in range (id,len(row))])  
    return(contour)

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

def get_normals(model,sign):
    normals = []
    for i in range(len(model)-1):
        p1 = model[i]
        p2 = model[(i + 1) % len(model)]
        n = inward_normal(p1, p2, sign)

        normals.append(n)
    return(normals)

def init_barres_vert(file_path,ymax):
    input=pd.read_excel(file_path, sheet_name="Input Sofistik", header=None, usecols="C", skiprows=4,nrows=4)
    n_supports=input.iloc[0,0]
    n_pieux=input.iloc[1,0]
    i_supports=input.iloc[2,0]
    i_pieux=input.iloc[3,0]
    barres_vert=[]
    pieux=pd.read_excel(file_path, sheet_name="Input data", header=None, usecols="C:D", skiprows=i_pieux-1,nrows=n_pieux)
    
    #ajouter une barre au niveau du pieux
    for i in range (n_pieux):
        barres_vert.append([pieux.iloc[i,0],pieux.iloc[0,1],pieux.iloc[i,0],ymax])
    supports=pd.read_excel(file_path, sheet_name="Input data", header=None, usecols="C:D", skiprows=i_supports-1,nrows=n_supports)
    #ajouter une barre au niveau des supports
    for i in range (n_supports):

        barres_vert.append([supports.iloc[i,0],supports.iloc[i,1],supports.iloc[i,0],0])

    return(barres_vert)
def barres_contour(contour,inward_normals,eps):

    horizontal_ties=[]
    vertical_ties=[]
    for i in range(len(contour)-1):
        p1,p2=contour[i],contour[i+1]
        n=inward_normals[i]
        tie=[p1+eps*n,p2+eps*n]
        tie_flat=[*tie[0],*tie[1]]
        if np.dot(n,[1,0])==0:
            horizontal_ties.append(tie_flat)
        elif np.dot(n,[0,1])==0:
            vertical_ties.append(tie_flat)
        else : 
            angle1,angle2=np.dot([0,1],n),np.dot([1,0],n)
            angle1,angle2=min(angle1,np.pi-angle1),min(angle2,np.pi-angle2)

            if angle2 < angle1 :
                p1,p2=p2,p1

            p2=np.array(p2)
            u=p2-p1
            if np.sign(np.dot([0,1],u))*np.sign(np.dot([0,1],n))==1 :
                p3=p1+ np.sign(np.dot([0,1],u))*np.dot([0,1],u)*np.array([0,1])
                
                vertical_ties.append([p1[0],p1[1],p3[0],p3[1]])
            else :
                p3=p1+ np.sign(np.dot([1,0],u))*np.dot([1,0],u)*np.array([1,0])
                horizontal_ties.append([p1[0],p1[1],p3[0],p3[1]])
            u=p1-p2
            
    return(vertical_ties,horizontal_ties)


def compute_maillage(pas_maillage_tirants,tirants,vertical=True):

    #calcul les barres à rajoutr pour effectuer le maillage voulu
    maillage=[]
    if vertical :
        idx=0
    else :
        idx=1
    for j in range(len(tirants)-1):
        maillage.append(tirants[j])
        nNewRows=int(np.ceil((tirants[j+1][idx]-tirants[j][idx])//pas_maillage_tirants))
        if nNewRows>0:
            pas=(tirants[j+1][idx]-tirants[j][idx])/nNewRows
            for i in range(1,nNewRows):
                maillage.append([tirants[j][0]+i*pas*(1-idx),tirants[j][1]+i*pas*idx,tirants[j][2]+i*pas*(1-idx),tirants[j][3]+i*pas*idx])
    maillage.append(tirants[-1])
    return(maillage)



def compute_barres(barres,contour,eps):
    barres_maille=[]

    for i in range(len(barres)):
        u1=np.array([barres[i][2]-barres[i][0],barres[i][3]-barres[i][1]],dtype=float)
        tirant_i=[]
        maxu1=np.linalg.norm(u1)
        u1/=np.linalg.norm(u1)
        
        for j in range(len(contour)-1):
            
            u2=np.array([contour[j+1][0]-contour[j][0],contour[j+1][1]-contour[j][1]],dtype=float )           

            maxu2=np.linalg.norm(u2)
            u2/=np.linalg.norm(u2)   

            #test of non colinearity
            
            if abs(np.dot(u1,u2)-1)> 1.e-4 and abs(np.dot(u1,u2)+1)> 1.e-4 :
                A=np.array([[u1[0],-u2[0]],[u1[1],-u2[1]]])
                B=np.array([[contour[j][0]-barres[i][0]],[contour[j][1]-barres[i][1]]])
                X=np.linalg.solve(A,B)
                x=barres[i][0:2]+(X[0])*u1
                if  0<=X[1]<=maxu2:    #le point se trouve sur le segment 
                    tirant_i.append(x)

        #transformer la liste de noeud en plusieurs barres
        
        tirant_i=np.unique(tirant_i, axis=0)

        if len(tirant_i)%2==1 : 
            raise Exception("Le modèle doit être reconfiguré, il y a une erreur pour générer les armatures. Les pieux et supports ne doivent pas être placés le long d'une face") 
        u=[np.array(tirant_i[2*k+1]-tirant_i[2*k]) for k in range (0,int(np.ceil(len(tirant_i)/2)))]
        for k in range (0,int(np.ceil(len(tirant_i)/2))):
            barres_maille.append([np.round(tirant_i[2*k]+eps*u[k]/np.linalg.norm(u[k]),3),np.round(tirant_i[2*k+1]-eps*u[k]/np.linalg.norm(u[k]),3)])
    barres_maille=np.unique(barres_maille,axis=0)
    return(barres_maille)



def main(file_path):
    sheet_name='Input data'
    contour=extract_contour(file_path,sheet_name,1,23,"B:D")

    contour=[[contour[i][0],contour[i][1]] for i in range(len(contour))]
    ymax=max([contour[i][1] for i in range(len(contour))])

    barres_vert= init_barres_vert(file_path,ymax)
    barres_hor=[]
    eps=pd.read_excel(file_path, sheet_name="Input data", header=None, usecols="C", skiprows=14,nrows=1).iloc[0,0]
    pas_maillage_horizontal, pas_maillage_vertical= pd.read_excel(file_path, sheet_name="Armatures", header=None, usecols="D:E", skiprows=12,nrows=1).iloc[0,0],pd.read_excel(file_path, sheet_name="Armatures", header=None, usecols="D:E", skiprows=12,nrows=1).iloc[0,1]
    #eps pour ne pas faire arriver les tirants aux bords, mais finalement on veut ça 
    eps1=0
    #calculer les tirants des bordures
    sign = polygon_orientation(contour)
    normals= get_normals(contour,sign)
    ties=barres_contour(contour,normals,eps)

    barres_hor+=ties[1]
    barres_vert+=ties[0]

    barres_vert.sort(key=lambda x: x[0])

    maillage_vert=compute_maillage(pas_maillage_vertical,barres_vert,vertical=True)
    maillage_hor=compute_maillage(pas_maillage_horizontal,barres_hor,vertical=False)
    nouveaux_barres_vert=compute_barres(maillage_vert,contour,eps1)
    nouveaux_barres_hor=compute_barres(maillage_hor,contour,eps1)
    wb = xw.Book(file_path)
    ws = wb.sheets["TMP_TIRANTS"]



    ws.range("2:1048576").clear_contents()

    row = 2
    max_len = max(len(nouveaux_barres_vert), len(nouveaux_barres_hor))

    for i in range(max_len):

        if i < len(nouveaux_barres_vert):
            x1, y1 = nouveaux_barres_vert[i][0]
            x2, y2 = nouveaux_barres_vert[i][1]
            ws.range((row, 1)).value= x1
            ws.range((row, 2)).value= y1
            ws.range((row, 3)).value= x2
            ws.range((row, 4)).value= y2

        if i < len(nouveaux_barres_hor):
            x1, y1 = nouveaux_barres_hor[i][0]
            x2, y2 = nouveaux_barres_hor[i][1]
            ws.range((row, 5)).value= x1
            ws.range((row, 6)).value= y1
            ws.range((row, 7)).value= x2
            ws.range((row, 8,)).value= y2

        row += 1

    print("Tirants mis à jour")

    wb.save(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")

    args = parser.parse_args()

    main(args.file_path)