import pickle
from flask import *
from flask_cors import cross_origin

app=Flask(__name__)

countVectorfile=open("CountVectorizer.pickle","rb")
CountVectorizer=pickle.load(countVectorfile)
countVectorfile.close()

tfidfTransformerFile=open("tfidftransformer.pickle","rb")
Tfitrans=pickle.load(tfidfTransformerFile)
tfidfTransformerFile.close()

mnbAlgoFile=open("algorithm.pickle","rb")
model=pickle.load(mnbAlgoFile)
mnbAlgoFile.close()

def getTextTransformedValue(review):
    count_data=CountVectorizer.transform([review])
    trans_data=Tfitrans.fit_transform(count_data)
    return trans_data

def getPredictions(transText):
    return model.predict(transText)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/rest/data/',methods=["POST"])
@cross_origin('f')
def processData():
    print(request.json)
    resultdict={0:"Positive",1:"Negative"}
    result=getPredictions(getTextTransformedValue(request.json))
    print(result)
    print(resultdict[result[0]])
    resp = jsonify({'data':str(resultdict[result[0]]),'message': 'OK'})
    return resp

if __name__=="__main__":
    app.run(debug=True)