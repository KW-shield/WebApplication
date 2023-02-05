from flask import Flask, render_template, request
import eng_mail
import kr_mail
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

app = Flask(__name__)
# Main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kor', methods=['GET','POST'])
def kor():
    if request.method=='GET':
        return render_template('kor.html')
    elif request.method=='POST':
        mail = request.form['e_mail']
        if not mail: return render_template('kor.html', label="No Texts")
        prediction, url_pred, url_in_mail = kr_mail.predict(mail)
        if(prediction == 1):   label = "spam"
        else:                  label = "ham"
        #메일 내 포함된 URL 중 악성 URL이 존재하는지 확인
        check = 0
        for i in range(0, len(url_pred)):
            if (url_pred[i][0][1] > 0.5):
                check = 1
        return render_template('result.html', label=label, prediction=prediction, url_pred=url_pred, url_in_mail=url_in_mail, check=check)

@app.route('/eng', methods=['GET','POST'])
def eng():
    if request.method=='GET':
        return render_template('eng.html')
    elif request.method=='POST':
        mail = request.form['e_mail']
        if not mail: return render_template('eng.html', label="No Texts")
        prediction, url_pred, url_in_mail = eng_mail.predict(mail)
        if(prediction == 1):   label = "spam"
        else:                  label = "ham"

        #메일 내 포함된 URL 중 악성 URL이 존재하는지 확인
        check = 0
        for i in range(0, len(url_pred)):
            if url_pred[i][0][1] > 0.5:
                check = 1
        return render_template('result.html', label=label, prediction=prediction, url_pred=url_pred, url_in_mail=url_in_mail, check=check)

if __name__ == '__main__':
    app.run(debug=True)