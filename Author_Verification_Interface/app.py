from flask import (Flask,
                   render_template,
                   flash,
                   redirect,
                   url_for,
                   session,
                   request,
                   logging)
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from werkzeug.utils import secure_filename
import os
from interface_model_link import taking_majority, testing1, making_Yes_NO, comp_labels
from keras.models import load_model

ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])

#config mysql
app.config["MYSQL_HOST"] = 'localhost'
app.config["MYSQL_USER"] = 'root'
app.config["MYSQL_PASSWORD"] = ''
app.config["MYSQL_DB"] = 'myflaskapp'
app.config["MYSQL_CURSORCLASS"] = 'DictCursor'

#init mysql
mysql = MySQL(app)

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

#home
@app.route('/')
def index():
    return render_template('index.html')

#about
@app.route('/about')
def about():
    return render_template('about.html')

#articles
@app.route('/articles')
def articles():
    cur = mysql.connection.cursor()
    result = cur.execute("Select * from articles")

    articles = cur.fetchall()

    if result > 0:
        return render_template('articles.html', articles=articles)

    else:
        msg = "No articles found"
        return render_template('articles.html', msg=msg)


    cur.close()

#single article
@app.route('/article/<string:id>/')
def single_articles(id):
    cur = mysql.connection.cursor()
    result = cur.execute("Select * from articles where id=%s", [id])

    article = cur.fetchone()

    return render_template('single_article.html', article=article)

#registration form
class RegisterForm(Form):
    name = StringField('Name', validators=[validators.Length(min=1, max=50), 
                                           validators.InputRequired()])
    username = StringField('username', validators=[validators.Length(min=4, max=25)])
    email = StringField('Email', validators=[validators.Length(min=6, max=50)])
    password = PasswordField('Password', validators=[
                             validators.DataRequired(),
                             validators.EqualTo('confirm', message="Passwords do not match")
                             ])
    confirm = PasswordField('Confirm Password')

#registration
@app.route('/registerform', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)

    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.hash(str(form.password.data))

        #create Cursor
        cur = mysql.connection.cursor()
        cur.execute("Insert into users(name, username, email, password) values(%s, %s, %s, %s)", (name, username, email, password))
        mysql.connection.commit()
        cur.close()

        flash("You are now registered", "success")
        return redirect(url_for("index"))

    return render_template('register.html', form=form)

#login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        #get form fields
        username = request.form["username"]
        password_candidate = request.form["password"]

        #create cursor
        cur = mysql.connection.cursor()
        result = cur.execute("select * from users where username=%s", [username])

        if result > 0:
            data = cur.fetchone()
            password = data["password"]

            if sha256_crypt.verify(password_candidate, password):
                session['logged_in'] = True
                session["username"] = username

                flash('You are now logged in', 'success')
                return redirect(url_for('dashboard'))
            else:
              error = "Invalid login"
              return render_template('login.html', error=error)  

            cur.close()

        else:
            error = "Username not found"
            return render_template('login.html', error=error)

    return render_template('login.html')

#dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
    cur = mysql.connection.cursor()
    result = cur.execute("Select * from articles")

    articles = cur.fetchall()

    if result > 0:
        return render_template('dashboard.html', articles=articles)

    else:
        msg = "No articles found"
        return render_template('dashboard.html', msg=msg)


    cur.close()

#article form class
class ArticleForm(Form):
    title = StringField('Title', validators=[validators.Length(min=1, max=50), 
                                             validators.InputRequired()])
    body = TextAreaField('Body', validators=[validators.Length(min=30)])

#add article
@app.route('/add_article', methods=['GET', 'POST'])
@is_logged_in
def add_article():
    form = ArticleForm(request.form)

    if request.method == 'POST' and form.validate():
        title = form.title.data
        body = form.body.data

        #create Cursor
        cur = mysql.connection.cursor()
        cur.execute("Insert into articles(title, body, author) values(%s, %s, %s)", (title, body, session['username']))
        mysql.connection.commit()
        cur.close()

        flash("Article created", "success")

        return redirect(url_for('dashboard'))

    return render_template('add_article.html', form=form)

#edit article
@app.route('/edit_article/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_article(id):
    #create cursor
    cur = mysql.connection.cursor()

    #get article by id
    result = cur.execute("Select * from articles where id=%s", [id])
    article = cur.fetchone()

    #get article form
    form = ArticleForm(request.form)

    #populate article form
    form.title.data = article["title"]
    form.body.data = article["body"]


    if request.method == 'POST' and form.validate():
        title = request.form['title']
        body = request.form['body']

        #create Cursor
        cur = mysql.connection.cursor()
        cur.execute("Update articles set title=%s, body=%s where id=%s", (title, body, id))
        mysql.connection.commit()
        cur.close()

        flash("Article Update", "success")

        return redirect(url_for('dashboard'))

    return render_template('edit_article.html', form=form)


#logout
@app.route('/logout')
def logout():
    session.clear()
    flash("You are now logged out", "success")
    return redirect(url_for("login"))

@app.route('/delete_article/<string:id>/', methods=['POST'])
@is_logged_in
def delete_article(id):
    cur = mysql.connection.cursor()
    cur.execute("Delete from articles where id=%s", (id))
    mysql.connection.commit()
    cur.close()

    flash("Article Deleted", "success")

    return redirect(url_for('dashboard'))


#train/test data upload
@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    
    if request.method == 'POST' and 'train' in request.files:
        target = os.path.join(APP_ROOT, "static/train")
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)
        else:
            print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist('train'))
        for f in request.files.getlist('train'):
            print(f)
            print('{} is the filename'.format(f.filename))
            file = secure_filename(f.filename)
            destination = '/'.join([target, file])
            print("Accepting incoming file", f.filename)
            print("Save it to", destination)
            f.save(destination)

        flash("Train data uploaded", "success")
        return redirect(url_for("upload_data"))

    if request.method == 'POST' and 'test' in request.files:
        target = os.path.join(APP_ROOT, "static/test")
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)
        else:
            print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist('test'))
        for f in request.files.getlist('test'):
            print(f)
            print('{} is the filename'.format(f.filename))
            file = secure_filename(f.filename)
            destination = '/'.join([target, file])
            print("Accepting incoming file", f.filename)
            print("Save it to", destination)
            f.save(destination)

        flash("Test data uploaded", "success")
        return redirect(url_for("upload_data"))

    return render_template("upload_data.html")

#predicting the output
@app.route('/api', methods=['GET', 'POST'])
def predict():
    best_model_path = 'lstm_50_200_0.17_0.25.h5' 
    #model = load_model(best_model_path)
    results, preds = testing1(best_model_path)
    labels = making_Yes_NO(preds)
    comp_labels(labels)
    string_result = taking_majority(labels)
    s = dict()
    s['result'] = string_result

    return render_template('predict.htm', s=s)

if __name__ == '__main__':
    app.secret_key = "secret123"
    app.run(debug=True)
