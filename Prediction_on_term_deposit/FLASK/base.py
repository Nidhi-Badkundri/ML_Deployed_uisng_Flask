from flask import *
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
OE=pickle.load(open("Ordinalencoder.pkl","rb"))
SC=pickle.load(open("StandardScaler.pkl","rb"))
BG=pickle.load(open("baggingclassifierdt.pkl","rb"))

def Get_prediction(x):
    x[1]=OE.transform(x[1])
    x=SC.tansform([x])
    op=BG.predict(x)
    if op==0:
        return("Not subscribed to term deposit")
    else:
        return("Subscribed to term deposit")
    
    
    
    
@app.route("/")
def displayform():
    print("good to go")
    return render_template("home.html")


@app.route("/reglink",method=["Post"])
def getinputdata():
    Age=int(request.form["age"])
    Job_type=request.form["Job"]
    Status=request.form["Marital_status"]
    Education_Status=request.form["Education"]
    Default=request.form["Credit"]
    balance=int(request.form["Avg"])
    housing_loan=request.form["housing"]
    personal_loan=request.form["Personal"]
    contact_type=request.form["Contact"]
    duration=int(request.form["Duration"])
    days=int(request.form["campaign"])
    pdays=int(request.form["contacted"])
    previous=int(request.form["previously"])
    poutcome=request.form["outcome"]


    x=pd.DataFrame(data=[[Age,Job_type,Status,Education_Status,Default,balance,housing_loan,personal_loan,
                          contact_type,duration,days,pdays,previous,poutcome]],columns=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                                                                                        'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous','poutcome'])
    ans=Get_prediction(x)
    return render_template("display.html",data=ans)



if(__name__=="__main__"):
    app.run(debug=True)
