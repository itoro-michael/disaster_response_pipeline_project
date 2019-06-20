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
	# """ The tokenize function takes a string message and returns a list of string tokens.

    # Args:
        # text (str): The string message.

    # Returns:
        # List: string tokens

    # """
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
    """ The index function returns the root template.

    Returns:
        render_template():

    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre')['message'].count()
    genre_names = list(genre_counts.index)
	
	# Obtain top 10 most common labels in dataset, for visualisation
    lbl_count = df.iloc[:, 4:].mean(axis=0)
    lbl_count = lbl_count*100
    lbl_count = lbl_count.sort_values(ascending=False)
    lbl_count = lbl_count[:10]
    lbl_name = list(lbl_count.index)
    lbl_name = [lbl.replace('_', ' ') for lbl in lbl_name if '_' in lbl]
	
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = 	[
					{
						'data': [
									gob.Bar
									(
										x=genre_names,
										y=genre_counts.values.tolist()
									)
								],

						'layout': 	{
										'title': 'Distribution of Message Genres',
										'yaxis':{
													'title': "Count"
												},
										'xaxis':{
													'title': "Genre"
												}
									}
					},
					{
						'data': [
									gob.Bar
									(
										x=lbl_name,
										y=lbl_count.values.tolist()
									)
								],

						'layout': 	{
										'title': 'Most Common Labels in the Dataset',
										'yaxis':{
													'title': "Percentage"
												},
										'xaxis':{
													'title': "Labels"
												}
									}
					}
				]	
	
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
	# """ The go function returns the go template.

    # Returns:
        # render_template():

    # """
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
	# """ The web app main function.

    # """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()