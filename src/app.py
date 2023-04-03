import base64
import datetime
import io
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
import nltk
import sklearn
import dash
from dash import html
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('vader_lexicon')

stop = stopwords.words('english')
stemmer = SnowballStemmer("english")
sent = SentimentIntensityAnalyzer()

# TESTING AREA

data1 = pd.read_csv("testdata.csv")

####### SPLITTING ########

def simple_split(data,y,length,split_mark=1):
    n = int(split_mark*length)
    X_train = data[:n].copy()
    y_train = y[:n].copy()
    return X_train,y_train


vectorizer = CountVectorizer()
X_train, y_train = simple_split(data1.reviews,data1.biproduct_sentiment,len(data1))

####### TRAINING #########

X_train = vectorizer.fit_transform(X_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)



# CHANGE DIRECTORY

# os.chdir("C:/Users/DVF2/Desktop/Check 1-8-23")

# IMPORT CSV OF MODEL
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([html.H1(children='Product Reviews Sentiment Analysis Dashboard',style={'textAlign': 'center'}),
    dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
        style={'width': '100%','height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px',},multiple=True),html.Div(id='output-data-upload'),])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            data = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            
            ####### CODES PARA SA PRELIMINARY TEXT PROCESSING
            data.rename(columns = {'column-0':'reviews'}, inplace = True)

            data['reviews'] = data['reviews'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True) # REMOVE URLS

            data['reviews'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True) # REMOVE NEW LINES

            data['reviews'] = data['reviews'].str.replace("&", '', regex=True).str.replace("%", '', regex=True).str.replace("$", '', regex=True).str.replace("#", '', regex=True).str.replace("@", '', regex=True)# REMOVE NOT NEEDED SPECIAL CHARACTERS

            ###### CODES PARA SA REMOVAL OF NAN
            
            data['reviews'].replace('', np.nan, inplace=True)
            data['review_stars'].replace('', np.nan, inplace=True)
            data['reviews'].replace(' ', np.nan, inplace=True)
            data['review_stars'].replace(' ', np.nan, inplace=True)

            data = data.dropna(subset=['reviews','review_stars'])
            data = data.reset_index(drop=True)            

            ###### CODES PARA SA SENTIMENT SCORE

            polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in data['reviews']]
            data['sentiment_score'] = polarity
            data['review_stars'] = pd.to_numeric(data['review_stars'])
            data['difference'] = data['review_stars'] - data['sentiment_score']

            ###### CODES PARA SA LABELING OF HAM/SPAM and POSITIVE and NEGATIVE (SA VADER SIDE)

            condition = [
                (data['difference'] >= 0.5),
                (data['difference'] < 0.5)
            ]
            values = ['spam','ham']
            data['classification'] = np.select(condition, values)

            condition = [
                (data['sentiment_score'] > 0),
                (data['sentiment_score'] == 0),
                (data['sentiment_score'] < 0)
            ]
            values = ['positive','neutral','negative']
            data['product_sentiment'] = np.select(condition, values)
            data.head()

            ####### CODES PARA SA FINAL TEXT CLEANING

            data.reviews = data.reviews.str.replace('[^a-zA-Z]', ' ', regex=True)
            data['reviews'] = data['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            data["reviews"]= data["reviews"].astype(str).str.split(" ", expand = False)
            data["reviews"]= data["reviews"].str.join(" ")
            data["reviews"] = data["reviews"].str.lower()
            data["reviews"]= data["reviews"].astype(str).str.split(" ", expand = False)
            data["reviews"] = data["reviews"].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
            data["reviews"]= data["reviews"].str.join(" ")

            ####### CODES PARA SA MNB PREDICTION
            x = 0
            for i in data.index:
                if x <= len(data):
                    data.loc[x, ['nb_prediction']] = \
                    nb.predict(vectorizer.transform([data.iat[x, 0]]))[0]  # CHANGE COLUMN
                    x += 1

            condition = [
                (data['nb_prediction'] == 0),
                (data['nb_prediction'] == 1)
            ]
            values = ["negative", "positve"]
            data['bi_nb_prediction'] = np.select(condition, values)

            ###### CODES PARA SA MULTIPLE DATATABLE

            series1 = data['bi_nb_prediction'].value_counts()
            df_result1 = pd.DataFrame(series1)
            df_result1 = df_result1.reset_index()  
            df_result1.columns = ['product_sentiment', 'count']

            series2 = data['classification'].value_counts()
            df_result2 = pd.DataFrame(series2)
            df_result2 = df_result2.reset_index()  
            df_result2.columns = ['classification', 'count']

            series3 = data['sentiment_score'].value_counts()
            df_result3 = pd.DataFrame(series3)
            df_result3 = df_result3.reset_index()
            df_result3.columns = ['sentiment_score', 'count']
            condition = [
                (df_result3['sentiment_score'] > 0),
                (df_result3['sentiment_score'] == 0),
                (df_result3['sentiment_score'] < 0),
            ]
            values = ["positve","neutral","negative"]
            df_result3['product_sentiment'] = np.select(condition, values) 

            #######

            fig1 = px.bar(df_result1, x="product_sentiment", y="count", color = "product_sentiment", title='Bar Graph for Product Sentiments')
            fig2 = px.pie(df_result2, names="classification", values="count", hole=.3, title='Percentage of Ham and Spam')
            fig3 = px.bar(df_result3, x="sentiment_score", y="count", color = "product_sentiment", title='Bar Graph for Sentiment Score')
            fig1.update_layout(title_x=0.5)
            fig2.update_layout(title_x=0.5)
            fig3.update_layout(title_x=0.5)

            

            filters = dbc.Stack([

                html.Div(className='table', children =[
                    dash_table.DataTable(df_result2.to_dict('records'),
                    [{'name': i, 'id': i} for i in df_result2.columns],
                        id='datatable2',style_header={'backgroundColor': 'black','fontWeight': 'bold','color':'white'},
                    )
                ], style={'width': '100%', 'display': 'inline-block', 'margin': '5px'} ),

                html.Div(className='table', children =[
                    dash_table.DataTable(df_result3.to_dict('records'),[{'name': i, 'id': i} for i in df_result3.columns],id='datatable3',style_header={'backgroundColor': 'black','fontWeight': 'bold','color':'white'},)
                ], style={'width': '100%', 'display': 'inline-block', 'margin': '5px'} ),

                html.Div(className='table', children =[
                    dash_table.DataTable(df_result1.to_dict('records'),
                    [{'name': i, 'id': i} for i in df_result1.columns],
                        id='datatable1',style_header={'backgroundColor': 'black','fontWeight': 'bold','color':'white'},
                    )
                ], style={'width': '100%', 'display': 'inline-block', 'margin': '5px'} ),

            ])

            ###### INSERT DROP CODES HERE ##############

            data = data.drop("product_sentiment", axis='columns')

            ######
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            data = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
            html.Center(dash_table.DataTable(
                data.to_dict('records'),
                [{'name': i, 'id': i} for i in data.columns],
                filter_action='native',
                sort_action="native",
                sort_mode="multi",
                style_header={'backgroundColor': 'black','fontWeight': 'bold','color':'white'},
                style_cell={'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 0,
                            'textAlign': 'left'},
                style_table={'overflowX': 'auto'},
                page_size=10),
                style={'height': '200%'}
                ),
            
            html.Hr(),

            html.Div(className='parent', children=[
                html.Div(className='content',children=[filters]),
                html.Div(className='content',children=[
                                dcc.Graph(
                                    id='example-graph1',
                                    figure=fig2,
                                    style={'width': '40vw', 'height': '50vh','border-radius':'15px', 'background-color':'white'}
                                ),  
                                dcc.Graph(
                                    id='example-graph2',
                                    figure=fig3,
                                    style={'width': '40vw', 'height': '50vh','border-radius':'15px', 'background-color':'white'}
                                ),
                                dcc.Graph(
                                    id='example-graph3',
                                    figure=fig1,
                                    style={'width': '40vw', 'height': '50vh','border-radius':'15px', 'background-color':'white'}
                                )
                ]),
            ],)           
    ])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)