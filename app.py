from logging import debug
from flask import Flask,render_template,request
import loan as l
#import pickle

app=Flask(__name__)
#pickle.load(open('loan.pkl','rb'))
@app.route("/",methods=['GET','POST'])
def hello(lp=0):
    if request.method=='POST':
        credithistory=request.form['credithistory']
        loan_pred=l.loan_prediction(credithistory)
        lp=loan_pred
        if lp==0:
            lp='YES'
        else:
            lp='NO'
    return render_template("index.html",applicant_approval=lp)

if __name__=="__main__":
    app.run(debug=True)

    
