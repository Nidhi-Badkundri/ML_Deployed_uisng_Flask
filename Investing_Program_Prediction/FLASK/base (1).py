from flask import *
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
X_LE = pickle.load(open("X_LE.pkl",'rb'))
Y_LE = pickle.load(open("Y_LE.pkl",'rb'))
SS = pickle.load(open("SS.pkl",'rb'))
model = pickle.load(open("GBC_model.pkl",'rb'))

def Get_prediction(x):
    x[1] = X_LE.transform([x[1]])
    x = np.array(x)
    x = SS.transform([x])
    op = model.predict(x)
    Y_LE.inverse_transform(op)
    if op==0:
        return("Eligible for Investment")
    else:
        return("Not Eligible for Investment")
    


@app.route("/")
def displayform():
    print("good to go")
    return render_template("home.html")

@app.route("/reglink",methods=["POST"])
def getinputdata():
    SE1=int(request.form["SE1"])
    SE2=(request.form["SE2"])
    BA1=float(request.form["BA1"])
    BA2=float(request.form["BA2"])
    BA3=float(request.form["BA3"])
    BA4=float(request.form["BA4"])
    BA5=float(request.form["BA5"])
    BA6=float(request.form["BA6"])
    BA7=float(request.form["BA7"])
    PE1=request.form["flag1"]
    PE2=request.form["flag2"]
    PE3=request.form["flag3"]
    PE4=request.form["flag4"]
    PE5=request.form["flag5"]
    PE6=request.form["flag6"]
    PE7=request.form["flag7"]
    PE8=request.form["flag8"]
    PE9=request.form["flag9"]
    PE10=request.form["flag10"]
    PE11=request.form["flag11"]
    PE12=request.form["flag12"]
    PE13=request.form["flag13"]
    PE14=request.form["flag14"]
    PE15=request.form["flag15"]
    IA1=int(request.form["IA1"])
    IA2=int(request.form["IA2"])
    IA3=int(request.form["IA3"])

    print(IA1, IA2, IA3)
    # input_df=pd.DataFrame(data=[[SE1,SE2,BA1,BA2,BA3,BA4,BA5,BA6,BA7,PE1,PE2,PE3,PE4,PE5,PE6,PE7,PE8,PE9,PE10,PE11,PE12,PE13,PE14,PE15,IA1,IA2,IA3]],
    #                 columns=[SE1,SE2,BA1,BA2,BA3,BA4,BA5,BA6,BA7,PE1,PE2,PE3,PE4,PE5,PE6,PE7,PE8,PE9,PE10,PE11,PE12,PE13,PE14,PE15,IA1,IA2,IA3])
    # ans=Get_prediction(input_df)
    # return render_template("display.html",data=ans)

    


if(__name__=="__main__"):
    app.run(debug=True)


