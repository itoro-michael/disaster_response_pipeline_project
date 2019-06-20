import json
import plotly
import pandas as pd
import plotly.graph_objs as gob

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tbl_message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
	
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graphs = 	[
					# {
						# 'data': [
									# go.Bar
									# (
										# x=genre_names.tolist(),
										# y=genre_counts.values.tolist()
									# )
								# ],

						# 'layout': 	{
										# 'title': 'Distribution of Message Genres',
										# 'yaxis':{
													# 'title': "Count"
												# },
										# 'xaxis':{
													# 'title': "Genre"
												# }
									# }
					# }
				# ]
    graph_two = []
    graph_two.append(
      gob.Bar(
      x = genre_names.tolist(),
      y = genre_counts.values.tolist(),
      )
    )

    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'),
                )
				
    graphs = []
    graphs.append(dict(data=graph_two, layout=layout_two))
	
	
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    print("graphJSON:", graphJSON)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()