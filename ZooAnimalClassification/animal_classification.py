import pandas as pd
import math

file = 'zoo'
filename = '/home/akinyilmaz/Desktop/Machine/zoo_animal_classification/' + file + '.csv'
"""fileclass = 'class'
filenameclass = '/home/akinyilmaz/Desktop/Machine/zoo_animal_classification/' + fileclass + '.csv'"""

def getData(filename):
    dfanimal = pd.read_csv(filename)

    df_class0 = dfanimal[dfanimal.class_type == 1]
    df_class1 = dfanimal[dfanimal.class_type == 2]
    df_class2 = dfanimal[dfanimal.class_type == 3]
    df_class3 = dfanimal[dfanimal.class_type == 4]
    df_class4 = dfanimal[dfanimal.class_type == 5]
    df_class5 = dfanimal[dfanimal.class_type == 6]
    df_class6 = dfanimal[dfanimal.class_type == 7]

    return df_class0,df_class1,df_class2,df_class3,df_class4,df_class5,df_class6

def compare_class0(colname,inp):
    df0 = getData(filename)[0]
    df0col = df0[colname]
    err = 0
    for inps in df0col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c0_mean = math.sqrt(err)/len(df0col)
    return c0_mean

def compare_class1(colname,inp):
    df1 = getData(filename)[1]
    df1col = df1[colname]
    err = 0
    for inps in df1col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c1_mean = math.sqrt(err)/len(df1col)
    return c1_mean

def compare_class2(colname,inp):
    df2 = getData(filename)[2]
    df2col = df2[colname]
    err = 0
    for inps in df2col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c2_mean = math.sqrt(err)/len(df2col)
    return c2_mean

def compare_class3(colname,inp):
    df3 = getData(filename)[3]
    df3col = df3[colname]
    err = 0
    for inps in df3col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c3_mean = math.sqrt(err)/len(df3col)
    return c3_mean

def compare_class4(colname,inp):
    df4 = getData(filename)[4]
    df4col = df4[colname]
    err = 0
    for inps in df4col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c4_mean = math.sqrt(err)/len(df4col)
    return c4_mean

def compare_class5(colname,inp):
    df5 = getData(filename)[5]
    df5col = df5[colname]
    err = 0
    for inps in df5col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c5_mean = math.sqrt(err)/len(df5col)
    return c5_mean

def compare_class6(colname,inp):
    df6 = getData(filename)[6]
    df6col = df6[colname]
    err = 0
    for inps in df6col:
        sqrt = math.pow((inp-inps),2)
        err = err + sqrt
    c6_mean = math.sqrt(err)/len(df6col)
    return c6_mean

def inv_propotion(in0,in1,in2,in3,in4,in5,in6):
    inlist = [in0,in1,in2,in3,in4,in5,in6]
    prob0 = (sum(inlist) - in0) / (sum(inlist) * (len(inlist)-1))
    prob1 = (sum(inlist) - in1) / (sum(inlist) * (len(inlist)-1))
    prob2 = (sum(inlist) - in2) / (sum(inlist) * (len(inlist)-1))
    prob3 = (sum(inlist) - in3) / (sum(inlist) * (len(inlist)-1))
    prob4 = (sum(inlist) - in4) / (sum(inlist) * (len(inlist)-1))
    prob5 = (sum(inlist) - in5) / (sum(inlist) * (len(inlist)-1))
    prob6 = (sum(inlist) - in6) / (sum(inlist) * (len(inlist)-1))
    return prob0, prob1, prob2, prob3, prob4, prob5, prob6

def predict(hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize):
    colname_list = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
    inp_list = [hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize]

    p_c0 = []
    p_c1 = []
    p_c2 = []
    p_c3 = []
    p_c4 = []
    p_c5 = []
    p_c6 = []


    for i in range(len(inp_list)):
        err_c0 = compare_class0(colname=colname_list[i], inp=inp_list[i])
        err_c1 = compare_class1(colname=colname_list[i], inp=inp_list[i])
        err_c2 = compare_class2(colname=colname_list[i], inp=inp_list[i])
        err_c3 = compare_class3(colname=colname_list[i], inp=inp_list[i])
        err_c4 = compare_class4(colname=colname_list[i], inp=inp_list[i])
        err_c5 = compare_class5(colname=colname_list[i], inp=inp_list[i])
        err_c6 = compare_class6(colname=colname_list[i], inp=inp_list[i])


        prb0 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[0]
        prb1 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[1]
        prb2 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[2]
        prb3 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[3]
        prb4 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[4]
        prb5 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[5]
        prb6 = inv_propotion(in0=err_c0, in1=err_c1, in2=err_c2, in3=err_c3, in4=err_c4, in5=err_c5, in6=err_c6)[6]

        p_c0.append(prb0)
        p_c1.append(prb1)
        p_c2.append(prb2)
        p_c3.append(prb3)
        p_c4.append(prb4)
        p_c5.append(prb5)
        p_c6.append(prb6)

    av_prob_c0 = sum(p_c0)/len(p_c0)
    av_prob_c1 = sum(p_c1)/len(p_c1)
    av_prob_c2 = sum(p_c2) / len(p_c2)
    av_prob_c3 = sum(p_c3) / len(p_c3)
    av_prob_c4 = sum(p_c4) / len(p_c4)
    av_prob_c5 = sum(p_c5) / len(p_c5)
    av_prob_c6 = sum(p_c6) / len(p_c6)
    maxprob = max(av_prob_c0,av_prob_c1,av_prob_c2,av_prob_c3,av_prob_c4,av_prob_c5,av_prob_c6)

    print("---------------------------------")
    if maxprob == av_prob_c0:
        print("Mamal")
    elif maxprob == av_prob_c1:
        print("Bird")
    elif maxprob == av_prob_c2:
        print("Reptile")
    elif maxprob == av_prob_c3:
        print("Fish")
    elif maxprob == av_prob_c4:
        print("Amphibian")
    elif maxprob == av_prob_c5:
        print("Bug")
    elif maxprob == av_prob_c6:
        print("Invertebrate")
    else:
        pass




    """print("mammal %s: "%(av_prob_c0*100))
    print("bird %s: "%(av_prob_c1*100))
    print("Reptile %s: "%(av_prob_c2*100))
    print("Fish %s: "%(av_prob_c3*100))
    print("Amphibian %s: "%(av_prob_c4*100))
    print("Bug %s: "%(av_prob_c5*100))
    print("Invertebrate %s: "%(av_prob_c6*100))"""

if __name__ == '__main__':

    hair = int(input("enter hair(1/0):"))
    feathers = int(input("enter feathers(1/0):"))
    eggs = int(input("enter eggs(1/0):"))
    milk = int(input("enter milk(1/0):"))
    airborne = int(input("enter airborne(1/0):"))
    aquatic = int(input("enter aquatic(1/0):"))
    predator = int(input("enter predator(1/0):"))
    toothed = int(input("enter toothed(1/0):"))
    backbone = int(input("enter backbone(1/0):"))
    breathes = int(input("enter breathes(1/0):"))
    venomous = int(input("enter venomous(1/0):"))
    fins = int(input("enter fins(1/0):"))
    legs = int(input("enter number of legs:"))
    tail = int(input("enter tail(1/0):"))
    domestic = int(input("enter domestic(1/0):"))
    catsize = int(input("enter catsize(1/0):"))

    predict(hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize)
