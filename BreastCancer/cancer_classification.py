
import pandas as pd
import math
import csv



def getData(filename):
    with open(filename,'r') as csv_inp, open (filenameo,'w') as csv_out:
        writer = csv.writer(csv_out)
        for row in csv.reader(csv_inp):
            if row[6] != "?":
                writer.writerow(row)
    readData = pd.read_csv(filenameo)
    df = pd.DataFrame({'id':readData[readData.columns[0]],
                       'clumb':readData[readData.columns[1]],
                       'cell_size':readData[readData.columns[2]],
                       'cell_shape':readData[readData.columns[3]],
                       'm_adhesion':readData[readData.columns[4]],
                       'sin_epi':readData[readData.columns[5]],
                       'bare_nuclei':readData[readData.columns[6]],
                       'bland_chromatin':readData[readData.columns[7]],
                       'normal_nucleoli':readData[readData.columns[8]],
                       'mitoses':readData[readData.columns[9]],
                       'cclass':readData[readData.columns[10]]})


    #select = df.apply(lambda r : any([isinstance(e, ) for e in r]),axis=1)


    dfbeningn = df[df.cclass == 2]
    dfmalignant = df[df.cclass == 4]
    return dfbeningn,dfmalignant

def compare_bening(colname,inp):
    dfb = getData(filename)[0]
    dfcol = dfb[colname]
    err = 0
    for sizes in dfcol:
        sqrt = math.pow((inp-sizes),2)
        err = err + sqrt
    ben_mean = math.sqrt(err)/len(dfcol)
    return ben_mean

def compare_malignant(colname,inp):
    dfb = getData(filename)[1]
    dfcol = dfb[colname]
    err = 0
    for sizes in dfcol:
        sqrt = math.pow((inp-sizes),2)
        err = err + sqrt
    mal_mean = math.sqrt(err)/len(dfcol)
    return mal_mean

def predict(clumb,cell_size,cell_shape,m_adhesion,sin_epi,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses):
    colnamelist = ['clumb','cell_size','cell_shape','m_adhesion','sin_epi','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']
    inplist = [clumb,cell_size,cell_shape,m_adhesion,sin_epi,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses]
    predictlist = []
    p_benign = []
    p_malignant = []

    """for i in range(len(inplist)):
        if compare_bening(colname=colnamelist[i],inp=inplist[i]) > compare_malignant(colname=colnamelist[i],inp=inplist[i]):
            predictlist.append('malignant')
        else:
            predictlist.append('benign')

    count_benign = 0
    count_malignant = 0
    for elem in predictlist:
        if elem == 'benign':
            count_benign += 1
        if elem == 'malignant':
            count_malignant += 1

    
    total = len(predictlist)
    print("--------------------------------------------")
    print(predictlist)
    print("Benign %s" %((count_benign/total)*100))
    print("Malignant %s" %((count_malignant/total)*100))"""

    for i in range(len(inplist)):
        e_b = compare_bening(colname=colnamelist[i],inp=inplist[i])
        e_m = compare_malignant(colname=colnamelist[i],inp=inplist[i])
        prob_b = e_m/(e_m+e_b)
        prob_m = e_b/(e_m+e_b)
        p_benign.append(prob_b)
        p_malignant.append(prob_m)
    avr_prob_benign = sum(p_benign)/len(p_benign)
    avr_prob_malignant = sum(p_malignant)/len(p_malignant)
    print("--------------------------------------------")
    print("Benign %s: "%(avr_prob_benign*100))
    print("Malignant %s: "%(avr_prob_malignant*100))

if __name__ == '__main__':
    file = 'cancer'
    fileo = 'cancero'
    filename = '/home/akinyilmaz/Desktop/Machine/' + file + '.csv'
    filenameo = '/home/akinyilmaz/Desktop/Machine/breast_cancer_classification/' + fileo + '.csv'

    clu_flag = True
    while clu_flag:
        clumb = int(input('enter clumb thickness(1-10):'))
        if clumb >= 1 and clumb <= 10:
            clu_flag = False
        else:
            print("enter again!")

    size_flag = True
    while size_flag:
        cell_size = int(input('enter cell size(1-10):'))
        if cell_size >= 1 and cell_size <= 10:
            size_flag = False
        else:
            print("enter again!")

    shape_flag = True
    while shape_flag:
        cell_shape = int(input('enter cell shape(1-10):'))
        if cell_shape >= 1 and cell_shape <= 10:
            shape_flag = False
        else:
            print("enter again!")

    adh_flag = True
    while adh_flag:
        m_adhesion = int(input('enter marginal adhesion(1-10):'))
        if m_adhesion >= 1 and m_adhesion <= 10:
            adh_flag = False
        else:
            print("enter again!")

    epi_flag = True
    while epi_flag:
        sin_epi = int(input('enter single epithelial cell size(1-10):'))
        if sin_epi >= 1 and sin_epi <= 10:
            epi_flag = False
        else:
            print("enter again!")

    bare_flag = True
    while bare_flag:
        bare_nuclei = int(input('enter bare nuclei(1-10):'))
        if bare_nuclei >= 1 and bare_nuclei <= 10:
            bare_flag = False
        else:
            print("enter again!")

    chro_flag = True
    while chro_flag:
        bland_chromatin = int(input('enter bland chromatin(1-10):'))
        if bland_chromatin >= 1 and bland_chromatin <= 10:
            chro_flag = False
        else:
            print("enter again!")

    nucle_flag = True
    while nucle_flag:
        normal_nucleoli = int(input('enter normal nucleoli(1-10):'))
        if normal_nucleoli >= 1 and normal_nucleoli <= 10:
            nucle_flag = False
        else:
            print("enter again!")

    mit_flag = True
    while mit_flag:
        mitoses = int(input('enter mitoses(1-10):'))
        if mitoses >= 1 and mitoses <= 10:
            mit_flag = False
        else:
            print("enter again!")

    predict(clumb,cell_size,cell_shape,m_adhesion,sin_epi,bare_nuclei,bland_chromatin,normal_nucleoli,mitoses)
