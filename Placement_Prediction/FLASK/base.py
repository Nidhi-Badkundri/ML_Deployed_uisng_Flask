from flask import *
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)


def predictinpdata(input_df):
    ct=pickle.load(open("coltransformer.pkl","rb"))
    lr=pickle.load(open("logmodel.pkl","rb"))
    x=ct.fit_transform(input_df)
    ans=lr.predict(x)[0]
    if ans==1:
        return "YOU WILL GET A JOB"
    else:
        return "YOU WILL NOT GET A JOB"
    


@app.route("/")
def displayform():
    return render_template("home.html")\

@app.route("/reglink",methods=["POST"])
def getinputdata():
    gender=request.form["gender"]
    sscmarks=float(request.form["sscmarks"])
    sscbboard=request.form["sscboard"]
    hscmarks=float(request.form["hscmarks"])
    hscboard=request.form["hscboard"]
    subject=request.form["subject"]
    degreemarks=float(request.form["degreemarks"])
    degree= request.form["degree"]
    experience=request.form["experience"]
    empmarks=float(request.form["empemarks"])
    specialisation=request.form["specialisation"]
    mbamarks=float(request.form["mbamarks"])

    input_df=pd.DataFrame(data=[[gender,sscmarks,sscbboard,hscmarks,hscboard,subject,degreemarks,
    degree,experience,empmarks,specialisation,mbamarks]],columns=['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board',
       'hsc_subject', 'degree_percentage', 'undergrad_degree',
       'work_experience', 'emp_test_percentage', 'specialisation',
       'mba_percent'])
    ans=predictinpdata(input_df)
    return render_template("display.html",data=ans)

    


if(__name__=="__main__"):
    app.run(debug=True)


