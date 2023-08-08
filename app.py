from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)
ml_model=joblib.load(open('linreg.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        # try:
            level=float(request.form['level'])

            args=[level]
            args_arr=np.array(args)
            args_arr=args_arr.reshape(1,-1)

            model_prediction=ml_model.predict(args_arr)
            model_prediction=round(float(model_prediction),3)

        # except valueError:
        #     return 'check value'

    return render_template('predict.html',prediction=model_prediction)




if __name__=='__main__':
    app.run(debug=True)    


