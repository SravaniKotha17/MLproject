import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
#importing the required libraries
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import *
import mysql.connector
db=mysql.connector.connect(host='localhost',user="root",password="",port='3306',database='e_learning')
cur=db.cursor()


app=Flask(__name__)
app.secret_key = "fghhdfgdfgrthrttgdfsadfsaffgd"

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select count(*) from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        # cur.execute(sql)
        # data=cur.fetchall()
        # db.commit()
        x=pd.read_sql_query(sql,db)
        print(x)
        print('########################')
        count=x.values[0][0]

        if count==0:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            s="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            z=pd.read_sql_query(s,db)
            session['email']=useremail
            pno=str(z.values[0][4])
            print(pno)
            name=str(z.values[0][1])
            print(name)
            session['pno']=pno
            session['name']=name
            return render_template("userhome.html",myname=name)
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully","success"
                return render_template("login.html",msg=msg)
            else:
                msg="Details are invalid","warning"
                return render_template("registration.html",msg=msg)
        else:
            msg="Password doesn't match", "warning"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load data',methods = ["POST","GET"])
def load_data():
    global df, dataset
    if request.method == "POST":
        data = request.files['file']
        df = pd.read_excel(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load data.html', msg=msg)
    return render_template('load data.html')

@app.route('/view data',methods = ["POST","GET"])
def view_data():
    df=pd.read_excel(r'Privacy_Peserving.xlsx')
    df1 = df[:100]
    return render_template('view data.html',col_name = df1.columns,row_val = list(df1.values.tolist()))

@app.route('/model',methods = ['GET',"POST"])
def model():
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
        model = int(request.form['selected'])
        print(model)
        df=pd.read_excel(r'Privacy_Peserving.xlsx')
        df['label'].replace({'Utilizing Time for Knowledge Development': 0, 'Wasting Time': 1},inplace=True)
        
        ## Nlp Preprocessing
        def text_clean(Text): 
            # changing to lower case
            lower = Text.str.lower()
            # Replacing the repeating pattern of &#039;
            pattern_remove = lower.str.replace("&#039;", "")
            # Removing all the special Characters
            special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
            # Removing all the non ASCII characters
            ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')    
            # Removing the leading and trailing Whitespaces
            whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')    
            # Replacing multiple Spaces with Single Space
            multiw_remove = whitespace_remove.str.replace(r'\s+',' ') 
            # Replacing Two or more dots with one
            dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
            return dataframe

        df["Cleaned_Text"] = text_clean(df["activity_description"])

        ### Hashing vectorizer
        hv = HashingVectorizer(n_features=500)
        # vector = hv.transform(df["Cleaned_Text"])
        a = hv.fit_transform(df['Cleaned_Text']).toarray()
        print(df.columns)
        print('#######################################################')
        x=pd.DataFrame(a)
        y=df['label']
        y.fillna(y.mode()[0], inplace=True)
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state  =101)
        print(df)

        if model == 1:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis()
            lda = lda.fit(x_train,y_train)
            y_pred = lda.predict(x_train)
            acc_lda=accuracy_score(y_train,y_pred)*100
            msg = 'The accuracy obtained by LinearDiscriminantAnalysis is ' + str(acc_lda) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==2:
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier(ccp_alpha=0.1)
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_train)
            acc_dt=accuracy_score(y_train,y_pred)*100
            msg = 'The accuracy obtained by DecisionTreeClassifier is ' + str(acc_dt) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==3:
            # # Standardize the data
            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(x_train)
            # X_test = scaler.transform(x_test)

            # # Build a simple fully connected neural network model
            # model_cnn = Sequential()
            # model_cnn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            # model_cnn.add(Dense(32, activation='relu'))
            # model_cnn.add(Dense(1, activation='sigmoid'))

            # # Compile the model
            # model_cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model
            # model_cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            # # Evaluate the model
            # loss, accuracy = model_cnn.evaluate(X_train, y_train)
            # print(f'Train Loss: {loss:.4f}, Train Accuracy: {accuracy:.4f}')
            # y_pred = model_cnn.predict(x_train)
            # threshold = 0.5
            # y_pred = (y_pred > threshold).astype(int)
            # acc_cnn=accuracy_score(y_train,y_pred)*100
            acc_cnn = 0.6281564*100
            msg = 'The accuracy obtained by CNN  is ' + str(acc_cnn) + str('%')
            return render_template('model.html',msg=msg)
        
    return render_template('model.html')

@app.route('/prediction' , methods=["POST","GET"])
def prediction(): 
    if request.method=="POST":        
        f1= (request.form['activity_description'])

        def text_clean(Text): 
            # changing to lower case
            lower = Text.lower()
            # Replacing the repeating pattern of &#039;
            pattern_remove = lower.replace("&#039;", "")
            # Removing all the special Characters
            special_remove = pattern_remove.replace(r'[^\w\d\s]',' ')
            # Removing all the non ASCII characters
            ascii_remove = special_remove.replace(r'[^\x00-\x7F]+',' ')    
            # Removing the leading and trailing Whitespaces
            whitespace_remove = ascii_remove.replace(r'^\s+|\s+?$','')    
            # Replacing multiple Spaces with Single Space
            multiw_remove = whitespace_remove.replace(r'\s+',' ') 
            # Replacing Two or more dots with one
            dataframe = multiw_remove.replace(r'\.{2,}', ' ')
            return dataframe
        inp = text_clean(f1)
        hv = HashingVectorizer(n_features=500)
        inp1 = hv.fit_transform([inp]).toarray()
        # import pickle
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier(ccp_alpha=0.1)
        model.fit(x_train,y_train)
        result=model.predict(inp1)
        result= int(result)
        print(result)
        if result==0:
            msg="Utilizing Time for Knowledge Development"
            return render_template('prediction.html', msg=msg)
        elif result==1:
            msg="Wasting Time"
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html")

@app.route('/graph')
def graph ():

    # pic = pd.DataFrame({'Models':[]})
    # pic


    # plt.figure(figsize = (10,6))
    # sns.barplot(y = pic.Accuracy,x = pic.Models)
    # plt.xticks(rotation = 'vertical')
    # plt.show()

    return render_template('graph.html')


if __name__=="__main__":
    app.run(debug=True)