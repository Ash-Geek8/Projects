from flask import Flask,render_template,request,redirect

app=Flask(__name__)
app.secret_key="123"

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/help')
def help():
    return render_template('help.html')
@app.route('/search',methods=['GET','POST'])
def search():
    if request.method=='POST':
        phrase = request.form.get('textarea')
        ragam=phrase
        return render_template('final.html',phrase=phrase,ragam=ragam)
    return render_template('search.html')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST' :
        name=request.form.get('first')
        password=request.form.get('password')
        if(chkUser(name,password)==1):
            return render_template('search.html',uname=name)
        else:
            return render_template('signIn.html',msg='* Invaid Login')
    return render_template('signIn.html')
names=['Ash','Joe','Alex']
unames=['Ash','jollyjoe','alexy']
passwords=['Ash@124','Joe#321','Giga6#']
def chkUser(name,pw):
    f=0
    if(name not in unames):
        return f
    try:
        ind=names.index(name)
    except:
        return render_template('/')
    if(pw==passwords[ind]):
        f=1
    return f
@app.route('/signUp')
def signUp():
    return render_template('signUp.html')


if __name__ == '__main__':
    app.run(debug=True,port=5000)