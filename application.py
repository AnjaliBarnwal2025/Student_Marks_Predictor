from flask import Flask,request,render_template
import pickle

application=Flask(__name__)
app=application

#import ridge regressor and standard scaler pickle
elasticnetcv_model=pickle.load(open('./Model/elasticnet.pkl','rb'))
scaler_model=pickle.load(open('./Model/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predictmarks",methods=['GET','POST'])
def predict_marks():
    if request.method=="POST":
        marks=float(request.form.get('Marks'))
        new_data_scaled=scaler_model.transform([[marks]])
        result=elasticnetcv_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')
if __name__ == "__main__":
    app.run(host="0.0.0.0")