from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
import base64

# image_path = 'D:/desktop/Preparation for Final Defense/Dash Codes/guide.jpg'

# def b64_image(image_filename):
#     with open(image_filename, 'rb') as f:
#         image = f.read()
#     return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Create your Dataframe',
            style={'textAlign': 'center'}),
    html.P(children='Please follow this image as guide in creating your dataframe',
            style={'textAlign': 'center'}),   
    
    # html.Div([html.Img(src=b64_image(image_path)),],style={'textAlign': 'center'}),
    html.Hr(),
    html.Div([
        dcc.Input(
            id='adding-rows-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
        ),
        html.Button('Add Column', id='adding-rows-button', n_clicks=0)
    ], style={'height': 50,'marginLeft': 800,'marginTop': 50}),
    
html.Div(dash_table.DataTable(
    id='adding-rows-table',
    columns=[{
        'name': 'Review {}'.format(i),
        'id': 'column-{}'.format(i),
        'deletable': True,
        'renamable': True
    } for i in range(1)],
    data=[
        {'column-{}'.format(i): (j + (i-1)*5) for i in range(1, 5)}
        for j in range(5)
    ],
    editable=True,
    row_deletable=True,
    export_format="csv",
    virtualization=True,
),style={'width': '50%', 'display': 'inline-block','marginLeft': 480}),

html.Button('Add Row', id='editing-rows-button', n_clicks=0),
])


@app.callback(
    Output('adding-rows-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('adding-rows-table', 'data'),
    State('adding-rows-table', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback(
    Output('adding-rows-table', 'columns'),
    Input('adding-rows-button', 'n_clicks'),
    State('adding-rows-name', 'value'),
    State('adding-rows-table', 'columns'))
def update_columns(n_clicks, value, existing_columns):
    if n_clicks > 0:
        existing_columns.append({
            'id': value, 'name': value,
            'renamable': True, 'deletable': True
        })
    return existing_columns


if __name__ == '__main__':
    app.run_server(debug=True)