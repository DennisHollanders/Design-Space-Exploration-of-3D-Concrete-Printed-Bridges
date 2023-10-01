import numpy as np
from typing import Type, Tuple
import json
import plotly.graph_objs as go
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from dash.exceptions import PreventUpdate
from scipy.stats import rankdata

symbols = ['circle', 'cross', 'diamond', 'square', 'x', 'diamond-open', 'square-open', 'circle-open', ]
COLORSCALE = [[0, '#ff0000'], [0.5, '#fb0080'], [1, '#1800ff']]
COLOR_CARDS = 'rgb(242,242,242)'
BINS = 4
#dash_component = Type[dcc.Component]


def preprocess_data(df, input_columns, COLUMNS_TO_NORMALIZE):
    """
     Preprocesses the input DataFrame by performing rank transformation on specified columns.

    :param df: combined dataframe
    :param input_columns: list of strings containing the names of the input columns
    :param COLUMNS_TO_NORMALIZE: list of strings containing the names of the columns to be ranked
    :return: a dataframe with rank-transformed values, a dataframe containing the input values, the min value of domain_1, the max value of domain_1
    """
    # Create relevant dataframes
    df_input = df[input_columns]
    max_domain = max(df['domain_1'])
    min_domain = min(df['domain_1'])
    df_n = df[COLUMNS_TO_NORMALIZE].copy(deep=True)

    # Apply rank transformation to specific columns
    inverted_columns = COLUMNS_TO_NORMALIZE
    for col in inverted_columns:
        ranks = rankdata(df_n[col])
        scaled_ranks = (ranks - 1) / (len(ranks) - 1)
        df_n[col] = (scaled_ranks * 2 - 1)
    return df_n, df_input, max_domain, min_domain


def layout(df_n, df_input, max_domain, min_domain, stepsize_y,
           input_values, columns_to_normalize, W_df):
    """
    Create the layout for the DAsh application

    :param df_n: Dataframe with rank-transformed values
    :param df_input: Dataframe with input values
    :param max_domain: The maximum value of the domain_1 column
    :param min_domain: The minimum value of the domain_1 column
    :param stepsize_y: The stepsize for the y-axis
    :param input_values: A list of input column names
    :param columns_to_normalize: a list of columns names with output values
    :param W_df: A dictionairy containing the weights for the objective space plot
    :param indices: A list containing all index numbers.
    :param str_indices: A string version of indices.
    :return: A list containing the location of each dash component on the dashboard

    """
    # Create some additional variables used
    indices = df_n.index.tolist()
    str_indices = [str(x) for x in indices]

    # define the layout
    """
    The dbc Card containers create the blocks shown in the dashboard. Furthermore dash uses a column/row system to define 
    where an element is located. For this the total maximum column width is always 12. 
    """
    layout = dbc.Container([
        # CONTAINERS USED TO STORE DATA BETWEEN GRAPHS THESE ARE NOT VISUALIZED.
        html.Div(dcc.Store('Opacities')),
        html.Div(dcc.Store(id='clicked_indices', data=[])),
        html.Div(dcc.Store(id='tsne-data')),
        html.Div(dcc.Store(id='data_frames')),

        # TITLE CARD
        dbc.Card([
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    html.H1("Design space analysis of a 3D printed concrete bridge"),
                    html.P(
                        'This file can be used to explore the design space of 3D concrete printed bridges. When running the dashboard it first manipulates the dataset by ranking all'
                        'output variables. The following variables are included. \n Input: Domains(1-3), amount of prestress, manufactore  variable (sets a minimum material constraint to certain elements),type of youngs modulus penalization,'
                        '\n Output: Vmax (maximum volume), Max and mean deflection, SF_v & SF_h support forces vertically and horizontal, Epoch optimization objective (summed strain energy)'
                        ', MSU (mean stress utilization), MPU (mean printwidth utilization)')
                ], width=10)
            ]), ], body=True, style={"background-color": COLOR_CARDS}),
        # TITLE CARD

        # CONSTRAINT AND FILTER CARD
        dbc.Row([
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    html.H3("Set constraint domain", style={'textAlign': 'center'})], width=8),
                dbc.Col([
                    html.H3("Filter database ", style={'textAlign': 'center'})], width=2)
            ]),
            dbc.Col([
                dbc.Card([
                    html.P("Domain sliders"),
                    html.Div([html.Label('Domain1'),
                              dcc.RangeSlider(id='Slider-Domain1', min=min_domain, max=max_domain,
                                              step=stepsize_y, value=[min_domain, max_domain],
                                              marks={min_domain: f'{min_domain}',
                                                     max_domain: f'{max_domain}'},
                                              tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.Div([html.Label('Domain2'),
                              dcc.RangeSlider(id='Slider-Domain2', min=min_domain, max=max_domain,
                                              step=stepsize_y, value=[min_domain, max_domain],
                                              marks={min_domain: f'{min_domain}',
                                                     max_domain: f'{max_domain}'},
                                              tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.Div([html.Label('Domain3'),
                              dcc.RangeSlider(id='Slider-Domain3', min=min_domain, max=max_domain,
                                              step=stepsize_y, value=[min_domain, max_domain],
                                              marks={min_domain: f'{min_domain}',
                                                     max_domain: f'{max_domain}'},
                                              tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.P("Sider to filter the database"),
                    dcc.Slider(id='vmax-slider', min=min(df_n["Vmax "]), max=max(df_n["Vmax "]),
                               value=max(df_n["Vmax "]), step=0.0001,
                               marks={min(df_n["Vmax "]): f'{np.round(min(df_n["Vmax "]))}',
                                      max(df_n["Vmax "]): f'{np.round(max(df_n["Vmax "]))}'},
                               tooltip={"placement": "bottom", "always_visible": True}), ]
                    , body=True, style={"background-color": COLOR_CARDS})], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        html.P("Use the sliders on the left to filter the database. By changing the domain sliders,"
                               "datapoints will be filtered from the database, plotting them transparent in the plots"
                               "of the following block. This allows the user the analyze constraint design spaces.\n "
                               "The histrogram plot shows how the maximum bridge volume is distributed after the ranking process. By adapting the "
                               "slider one can remove the outliers, these datapoints will be completetly excluded from "
                               "the visualizations."
                               ""),
                        dbc.Col([html.Div(dcc.Graph(id='Domain_graph', )), ], width=8),
                        dbc.Col([html.Div(dcc.Graph(id='histogram_plot')), ], width=4),
                    ]),
                ], body=True)], width=10),
        ], align="stretch"),
        # CONSTRAINT AND FILTER CARD

        # OBJECTIVE SPACE AND T-SNE PLOT
        dbc.Row([
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    html.H3("Design Space Exploration", style={'textAlign': 'center'})], width=5),
                dbc.Col([
                    html.H3("Design Space Analysis", style={'textAlign': 'center'})], width=5)
            ]),
            dbc.Col([
                dbc.Card([
                    html.P("Adapt the visualization of the plots"),
                    html.Div([html.Label('Select category for markershape in visualization'),
                              dcc.Dropdown(input_values, placeholder="Select an input category",
                                           id='Dropdown')]),
                    html.Div([html.Label('Select category for colors in visualization'),
                              dcc.Dropdown(columns_to_normalize, placeholder="Select an output category", value="Vmax ",
                                           id='color_scaling')]),
                    html.P("Adapt the visualized bounds"),
                    html.Div([html.Label('Cost bound'),
                              dcc.Slider(id='cost-slider', min=0.001, max=1, step=0.001, value=1,
                                         marks={0: '0', 1: '1'},
                                         tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.Div([html.Label('Eco bound'),
                              dcc.Slider(id='eco-slider', min=0.001, max=1, step=0.001, value=1,
                                         marks={0: '0', 1: '1'},
                                         tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.Div([html.Label('Manu bound'),
                              dcc.Slider(id='manu-slider', min=0.001, max=1, step=0.001, value=1,
                                         marks={0: '0', 1: '1'},
                                         tooltip={"placement": "bottom", "always_visible": True}), ]),
                    html.P("Manipulate the t-SNE plot"),
                    html.Div([html.Label('Dimensions of t-SNE'),
                              dcc.Slider(id='tsne-dimension', min=2, max=3, value=3, step=1,
                                         marks={2: '2', 3: '3'},
                                         tooltip={"placement": "bottom", "always_visible": True})]),
                    html.Div([html.Label('Learning rate of t-SNE'),
                              dcc.Slider(id='Learning_rate', min=5, max=1000, value=275, step=10,
                                         marks={5: '5', 1000: '1000'},
                                         tooltip={"placement": "bottom", "always_visible": True})]),
                    html.Div([html.Label('perplexity of t-SNE'),
                              dcc.Slider(id='perplex', min=5, max=50, value=5, step=1,
                                         marks={5: '5', 50: '50'},
                                         tooltip={"placement": "bottom", "always_visible": True})]),
                ]
                    , body=True, style={"background-color": COLOR_CARDS})], width=2),
            dbc.Col([
                dbc.Card([
                    html.P("The scatterplots below can be used to explore parameters and find desired bridges"
                           "The plot on the left, plots the objective space according to the tabel below the graph."
                           "Users can thereby explore various objective spaces and the influence of parameters on the "
                           "objective space. The graph on the right is a t-SNE plot, which plots all input parameters in a two "
                           "or 3 dimensional space. Clusters of similar input will form, which can be analyzed by plotting various "
                           "output parameters as colorscale to it. the sliders on the left allow to manipulate the visualization, by setting "
                           "different colorscales, different marker types and changing the t-SNE calculation. The perplexity can be changed to change the ratio between preserving "
                           "local or global structures,while the learning rate determines the amount of change allowed to be made during each iteration. A higher learning rate causes more clustering."),
                    dbc.Row([
                        dbc.Col([html.Div(dcc.Graph(id='3D_scatterplot')), ], width=6),
                        dbc.Col([html.Div(dcc.Graph(id='tsne-plot')), ], width=6),
                        html.Div([html.Label('Set weights for the objective space'),
                                  dash_table.DataTable(id='table-CME',
                                                       columns=[{'id': i, 'name': i} for i in W_df.columns],
                                                       data=W_df.to_dict('records'), editable=True, )],
                                 style={'width': '50%', 'display': 'inline-block'}),
                    ], )
                ], body=True)], width=10),
        ], align="stretch"),
        # OBJECTIVE SPACE AND T-SNE PLOT

        # PARALLEL CATEGORIES PLOT
        dbc.Row([
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    html.H3("Parallel categories plot ", style={'textAlign': 'center'})], width=10),
            ]),
            dbc.Col([
                dbc.Card([
                    html.P("Added Output"),
                    html.Div(
                        [dcc.Dropdown(columns_to_normalize, id='Dropdown_analysis', multi=True, value=['Vmax '], )]), ]
                    , body=True, style={"background-color": COLOR_CARDS})], width=2),
            dbc.Col([
                dbc.Card([
                    html.P("Use the interactive parallel categories (PC) plot to find relations between "
                           "in- and output. The distribution plot provided above can be used to filter elements from the PC plot. "
                           "The bins shown in this distribution plot are also displayed as bins in this plot. "
                           "The dropdown menu on the right allows to add the desired output values. Multiple can be added. The last one in the list determines "
                           "the colorscale. "),
                    html.Div(dcc.Graph('PC'))
                ], body=True)], width=10),
        ], align="stretch"),
        # PARALLEL CATEGORIES PLOT

        # VISUALIZATIONS
        dbc.Row([
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    html.H3("Visualization of selected bridge", style={'textAlign': 'center'})], width=8),
                dbc.Col([
                    html.H3("Performance", style={'textAlign': 'center'})], width=2)
            ]),
            dbc.Col([
                dbc.Card([
                    html.P("Indices of the bridges visualized."),
                    html.Div([dcc.Dropdown(str_indices, id='multi-select', searchable=True, multi=True, value=[0])]),

                ], body=True, style={"background-color": COLOR_CARDS})], width=2),
            dbc.Col([
                dbc.Card([
                    html.P("This component visualizes the datapoints selected in the two scatterplots. "
                           "The indices visualized are shown in the dropdown menu to the left. Additional "
                           "datapoints can be added in the dropdown menu. The table below is added to find the "
                           "index of a specific bridge based on the input selected in the table. The input making "
                           "up visualized bridge will be presented next to it. The output values are plotted in the "
                           "radial bar plot. Due to the ranking all output is between -1 and 1. Relative high values are therefore plotted in red"
                           "whereas relative low values are plotted in blue. No bar size will indicate a median performance."),
                    html.Div([html.Label('Find index of for given input data'),
                              dash_table.DataTable(id='select_index_table', data=df_input.head(1).to_dict('records'),
                                                   columns=[{'id': i, 'name': i, 'presentation': 'dropdown'} for i in
                                                            df_input.columns],
                                                   editable=True,
                                                   dropdown={k: {'options': [{'label': str(i), 'value': i} for i in
                                                                             df_input[k].unique()]} for k in
                                                             df_input.columns},
                                                   css=[{"selector": ".Select-menu-outer",
                                                         "rule": 'display : block !important'}]
                                                   )]),
                    html.Div(id='index', children=[]),
                    dbc.Row([
                        # dbc.Col([html.Div(id='text-elements')], width=3),
                        dbc.Col([html.Div(id='Visualization')], width=7),
                        dbc.Col([html.Div(id='stats_plot')], width=5),

                    ], align="start"),
                ], body=True)], width=10),
        ], align="stretch"),
    ])
    return layout


def callbacks(app, df_n, df_input, max_domain, min_domain, _, __,columns_to_normalize, ___, CME,x_values, ____, df3, categorical_columns):
    """
    Defines the callback functions for the dash application

    :param app: The dash application
    :param CME: List containg the names for the axes of the objective plot
    :param x_values: list with x locations for each domain point
    :param df3: The unprocessed concatenated dataframe with input and output
    :param categorical_columns: List containing the names of the columns that contain categorical data
    :param df_n: Dataframe with rank-transformed values
    :param df_input: Dataframe with input values
    :param max_domain: The maximum value of the domain_1 column
    :param min_domain: The minimum value of the domain_1 column
    :param columns_to_normalize: a list of columns names with output values
    :param columns_to_plot: Column names used for the radial bar plot
    :param indices: a list containing all index numbers
    :return: None
    """

    # generate additional variables
    indices = df_input.index.tolist()
    columns_to_plot = [column for column in columns_to_normalize if column != 'a_p']

    # Domain plot
    @app.callback(
        [Output('Domain_graph', 'figure'),
         Output('Opacities', 'data')],
        [Input('Slider-Domain1', 'value'),
         Input('Slider-Domain2', 'value'),
         Input('Slider-Domain3', 'value')])
    def update_graph_a(slider1a_value, slider1b_value,slider1c_value):
        """
        take the slider range inputs and plot these ranges to show which domain is being included in the analysis or exploration.
        Create the Opacity list which is used for filtering the datapoint falling outside the domain.

        :param slider1a_value: ranges for the 1st domain point.
        :param slider1b_value: ranges for the 2nd domain point.
        :param slider1c_value: ranges for the 3th domain point.
        :param Opacity: list containing the opacity values for each datapoint
        :return: Plot showing the design space, opacity list
        """
        # copy df_input to prevent errors throughout using the dashboard
        df_input_applied = df_input.loc[:]

        # Create a list with maximum and minimum y values based on the slider ranges.
        y_high = [slider1a_value[1], slider1b_value[1], slider1c_value[1], slider1b_value[1], slider1a_value[1]]
        y_low = [slider1a_value[0], slider1b_value[0], slider1c_value[0], slider1b_value[0], slider1a_value[0]]
        Opacity = []
        # list with all data to be plotted
        data = [
            # Line showing the highest used domain values
            go.Scatter(x=x_values, y=y_high, mode='lines+markers', name='Max desired domain from sliders',
                       fill='tonexty',
                       line=dict(color='rgb(179,179,179,0.2)', width=2)),
            # Line showing the lowest used domain values
            go.Scatter(x=x_values, y=y_low, mode='lines+markers', name='min desired domain from sliders',
                       fill='tonexty',
                       line=dict(color='rgb(179,179,179,0.2)', width=2)),
            # Line showing the highest allowable domain value
            go.Scatter(x=x_values, y=np.array([max_domain] * 6), mode='lines', name='Max_allowable domain',
                       line=dict(color='grey', width=1, dash='dash')),
            # Line showing the lowest allowable domain value
            go.Scatter(x=x_values, y=np.array([min_domain] * 6), mode='lines', name='Min_allowable domain',
                       line=dict(color='grey', width=1, dash='dash')),
            # Line showing the loading path
            go.Scatter(x=x_values, y=np.array([0] * 6), mode='lines+markers', name='Loading Path',
                       line=dict(color='red', width=2))
        ]
        layout = go.Layout(template='gridon', title='Set Desired Domain')

        # Standard opacity is 1.0 if the datapoint falls outside domain set opacity to 0.5
        for i, row in enumerate(df_input_applied.iterrows()):
            if slider1a_value[0] <= df_input_applied.loc[i, 'domain_1'] <= slider1a_value[1] and \
                    slider1b_value[0] <= df_input_applied.loc[i, 'domain_2'] <= slider1b_value[1] and \
                    slider1c_value[0] <= df_input_applied.loc[i, 'domain_3'] <= slider1c_value[1]:
                Opacity.append(1.0)
            else:
                Opacity.append(0.5)
        opacity = np.array(Opacity)
        return go.Figure(data=data, layout=layout), opacity

    @app.callback(
        Output('data_frames', 'data'),
        [Input('vmax-slider', 'value'),
         Input('Opacities', 'data')])
    def import_filter_data(Vmax_values, opacity):
        """
        Filter the data based on the histogram settings add opacity to the dataframes.
        Store the dataframes in the dashboard.

        :param Vmax_values: Included maximum volumes.
        :param opacity: Numpy array containing the opacity values for each datapoint.
        :return: dataframes
        """

        df_input_used = df_input.copy()
        df_n_used = df_n.copy()
        if 'Opacity' not in df_input_used.columns:
            df_input_used['Opacity'] = 1.0
            df_n_used['Opacity'] = 1.0
        if min(opacity) == 0.5:
            df_input_used['Opacity'] = opacity
            df_n_used['Opacity'] = opacity
        input_data_filter = df_input_used[df_n_used['Vmax '] <= Vmax_values].copy()
        output_data_filter = df_n_used[df_n_used['Vmax '] <= Vmax_values].copy()  #
        datasets = {
            'df_input': df_input_used.to_json(orient='split', date_format='iso'),
            'df_n': df_n_used.to_json(orient='split', date_format='iso'),
            'df_input_filter': input_data_filter.to_json(orient='split', date_format='iso'),
            'df_n_filter': output_data_filter.to_json(orient='split', date_format='iso'), }
        return json.dumps(datasets)

    @app.callback(Output('index', 'children'),
                  [Input('data_frames', 'data'),
                   Input('select_index_table', 'data')])
    def find_index(dfs, datatable):
        """
        Finds the index of a desired bridge based on input data.
        This callback belongs to the visualization Card.

        :param dfs: JSON type dataframes
        :param datatable: table containing the input values for the index to be found
        :return: Desired index
        """
        dataframes = json.loads(dfs)
        df_datatable = pd.DataFrame.from_dict(datatable)
        df_index = pd.read_json(dataframes['df_input'], orient='split')
        df_index.drop('Opacity', axis=1)
        for k, index in enumerate(df_datatable.columns):
            df_index = df_index.loc[df_index[index] == df_datatable[index].iloc[0]]
        return df_index.index.to_list()

    @app.callback(
        Output('histogram_plot', 'figure'),
        Input('data_frames', 'data'), )
    def update_histogram(dfs):
        """
        Creates the histogram plot.

        :param dfs: dataframes in JSON type
        :param BINS: the amount of bins shown
        :return: Plotly figure showing the ranked distribution of the filtered design space.
        """
        dataframes = json.loads(dfs)
        df_f = pd.read_json(dataframes['df_n_filter'], orient='split')
        hist, bins = np.histogram(df_f['Vmax '], bins=BINS)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        fig = go.Figure(data=[go.Bar(x=bins[:-1], y=hist, marker=dict(colorscale=COLORSCALE, color=bin_centers))])
        fig.update_layout(template='plotly_white', title='Histogram of ranked maximum volume distribution',
                          xaxis_title="Maximum volume", yaxis_title="Count", xaxis=dict(autorange=True))
        return fig

    @app.callback(
        Output('3D_scatterplot', 'figure'),
        [Input('table-CME', 'data'),
         Input('cost-slider', 'value'),
         Input('eco-slider', 'value'),
         Input('manu-slider', 'value'),
         Input('Dropdown', 'value'),
         Input('color_scaling', 'value'),
         Input('clicked_indices', 'data'),
         Input('data_frames', 'data')])
    def update_3D_scatterplot(Weights, cost_bound, eco_bound, manu_bound,
                              Category, Colorscaling, c_indices, dfs):
        """
        Update the 3D scatterplot based on user inputs.

        :param Weights: DataFrame containing weights.
        :param cost_bound: Upper bound for cost.
        :param eco_bound: Upper bound for eco.
        :param manu_bound: Upper bound for manu.
        :param Category: Category defining the markershape.
        :param Colorscaling: Category selected for color scaling for data points.
        :param c_indices: List of clicked indices.
        :param dfs: JSON type dataframes.
        :return: Plotly Figure representing the 3D scatterplot.
        """

        # create required variables
        fig = go.Figure()
        Weights = pd.DataFrame(Weights).set_index('Index').apply(pd.to_numeric)
        dataframes = json.loads(dfs)
        df_f = pd.read_json(dataframes['df_n_filter'], orient='split')
        df_input_applied = pd.read_json(dataframes['df_input_filter'], orient='split')
        min_max_scaler = MinMaxScaler()

        # normalize filtered data
        for i in CME:
            weights = Weights[i]
            df_f.loc[:, i] = 0
            for j, name in enumerate(columns_to_normalize):
                df_f.loc[:, i] = df_f.loc[:, i] + abs((df_f.loc[:, name] + 1) * weights[j])
            data = df_f.loc[:, i].values.reshape(-1, 1)
            df_f.loc[:, i] = min_max_scaler.fit_transform(data)

        # Define coloring based on user input
        if Colorscaling is None:
            df_f['Color'] = 'rgba(99,110,250)'
        else:
            df_f['Color'] = df_f[Colorscaling]
        # Define markershape based on user input
        if Category is None:
            df_f['symbol'] = 'circle'
        else:
            unique_strings = df_input_applied[Category].unique()
            Dictionary = {str(string): int(i) for i, string in enumerate(unique_strings)}
            df_f['symbol'] = [str(x) for x in df_input_applied[Category]]
            df_f['symbol'] = df_f['symbol'].map(Dictionary)
            df_f['symbol'] = df_f['symbol'].apply(lambda x: symbols[x])
            Dictionary_symbol = {symbols[value]: key for key, value in Dictionary.items()}
        # Define markerborder based on user input
        if c_indices is None:
            df_f['markerborder'] = 0
        else:
            df_f['markerborder'] = [10 if index in c_indices else 0 for index in df_f.index]

        # plot each possible group in dataframe as a trace using the groupby method of pandas
        for name, group in df_f.groupby(['symbol', 'Opacity', 'markerborder']):
            if Category is not None:
                original_string = Dictionary_symbol[name[0]]
            if not group.empty:
                fig.add_trace(go.Scatter3d(
                    x=group['Cost'], y=group['Eco'], z=group['Manu'], mode='markers', showlegend=True,
                    marker=dict(size=8, color=group['Color'], coloraxis='coloraxis',
                                opacity=group['Opacity'].iloc[0],
                                symbol=group['symbol'].iloc[0],
                                line=dict(color='black', width=10 if name[2] != 0 else 0)),
                    customdata=[indices[i] for i in group.index],
                    name=f"select category {'Selected' if name[2] > 0 else ''}" if Category is None else f"{original_string}{'filtered' if name[1] < 1 else ''}{' selected' if name[2] > 0 else ''}"))
        # Update the bounds with the upper slider values.
        fig.update_layout(
            coloraxis=dict(colorscale=COLORSCALE, colorbar=dict(len=1, x=0.0, y=-0.5, xanchor='left', orientation='h')),
            annotations=[dict(x=0.5, y=-0.15, xanchor='center', yanchor='bottom', showarrow=False,
                              text=f"colorscale: {Colorscaling}", xref="paper", yref="paper")],
            scene=dict(xaxis=dict(title='Cost', range=[0, cost_bound]),
                       yaxis=dict(title='Eco', range=[0, eco_bound]),
                       zaxis=dict(title='Manu', range=[0, manu_bound])),
            title="Objective space")
        return fig

    @app.callback(Output('tsne-data', 'data'),
                  [Input('data_frames', 'data'),
                   Input('tsne-dimension', 'value'),
                   Input('perplex', 'value'),
                   Input('Learning_rate', 'value')])
    def calculate_tsne_values(dfs, Dimensions, perplexity, learning_rate):
        """
        Calculate t-SNE axis values, the projections, for the dimensionality reduction plot.

        :param dfs: JSON type dataframes.
        :param Dimensions: Number of dimensions for t-SNE.
        :param perplexity: Perplexity parameter for t-SNE.
        :param learning_rate: Learning rate for t-SNE.
        :return: DataFrame containing t-SNE projections.
        """

        dataframes = json.loads(dfs)
        df_input_applied = pd.read_json(dataframes['df_input'], orient='split')
        df_with_dummies = pd.get_dummies(df_input_applied, columns=categorical_columns, drop_first=True)
        tsne = TSNE(n_components=Dimensions, random_state=0, learning_rate=learning_rate, perplexity=perplexity)
        projections = tsne.fit_transform(df_with_dummies)
        return projections

    @app.callback(Output('tsne-plot', 'figure'),
                  [Input('vmax-slider', 'value'),
                   Input('data_frames', 'data'),
                   Input('Dropdown', 'value'),
                   Input('color_scaling', 'value'),
                   Input('clicked_indices', 'data'),
                   Input('tsne-data', 'data'),
                   Input('tsne-dimension', 'value'), ])
    def update_tsne_plot(vmax_value, dfs, Category, Colorscaling, c_indices,
                         tsne_data, Dimensions):
        """
        Visualize and update the t-SNE plot based on user inputs.

        :param vmax_value: Maximum allowable volume.
        :param dfs: JSON representation of dataframes.
        :param Category: Selected category for markershape.
        :param Colorscaling: Selected category for color scaling.
        :param c_indices: Indices of selected data points.
        :param tsne_data: t-SNE projections data.
        :param Dimensions: Number of dimensions for t-SNE.
        :return: Plotly figure for the t-SNE plot.
        """
        projections = np.asarray(tsne_data)
        dataframes = json.loads(dfs)
        df_input_applied = pd.read_json(dataframes['df_input'], orient='split')
        df_nf = pd.read_json(dataframes['df_n'], orient='split')
        # add projects to the dataframe
        for i in range(Dimensions):
            df_input_applied["projection_" + str(i)] = projections[:, i]
        if Colorscaling is None:
            df_input_applied['Color'] = 'rgba(99,110,250)'
        else:
            df_input_applied['Color'] = df_nf[Colorscaling]
        # add markershape to dataframe
        if Category is None:
            df_input_applied['symbol'] = 'circle'
        else:
            unique_strings = df_input_applied[Category].unique()
            Dictionary = {str(string): int(i) for i, string in enumerate(unique_strings)}
            df_input_applied['symbol'] = [str(x) for x in df_input_applied[Category]]
            df_input_applied['symbol'] = df_input_applied['symbol'].map(Dictionary)
            df_input_applied['symbol'] = df_input_applied['symbol'].apply(lambda x: symbols[x])
            Dictionary_symbol = {symbols[value]: key for key, value in Dictionary.items()}
        # Add markerborder to the dataframe
        if c_indices is None:
            df_input_applied['markerborder'] = 0
        else:
            df_input_applied['markerborder'] = [10 if index in c_indices else 0 for index in df_n.index]
        df_input_f = df_input_applied.loc[df_nf['Vmax '] <= vmax_value, :]
        fig = go.Figure()
        # plot a 3D figure
        if Dimensions == 3:
            for name, group in df_input_f.groupby(['symbol', 'Opacity', 'markerborder']):
                if Category is not None:
                    original_string = Dictionary_symbol[name[0]]
                if not group.empty:
                    fig.add_trace(go.Scatter3d(
                        x=group["projection_0"], y=group["projection_1"], z=group["projection_2"], mode='markers',
                        showlegend=True,
                        marker=dict(size=8, color=group['Color'], coloraxis='coloraxis',
                                    opacity=group['Opacity'].iloc[0], symbol=group['symbol'].iloc[0],
                                    line=dict(color='black', width=10 if name[2] != 0 else 0)),
                        customdata=[indices[i] for i in group.index],
                        name=f"select category {'Selected' if name[2] > 0 else ''}" if Category is None else f"{original_string}{'filtered' if name[1] < 1 else ''}{' selected' if name[2] > 0 else ''}"))
            fig.update_layout(
                coloraxis=dict(colorscale=COLORSCALE,
                               colorbar=dict(len=1, x=0.0, y=-0.5, xanchor='left', orientation='h')),
                annotations=[dict(x=0.5, y=-0.15, xanchor='center', yanchor='bottom', showarrow=False,
                                  text=f"colorscale: {Colorscaling}", xref="paper", yref="paper")],
                scene=dict(xaxis=dict(title='t-SNE 1', showticklabels=False),
                           yaxis=dict(title='t-SNE 2', showticklabels=False),
                           zaxis=dict(title='t-SNE 3', showticklabels=False)), title="t-SNE plot of the design space")
        # plot a 2D figure
        if Dimensions == 2:
            for name, group in df_input_f.groupby(['symbol', 'Opacity', 'markerborder']):
                if Category is not None:
                    original_string = Dictionary_symbol[name[0]]
                if not group.empty:
                    fig.add_trace(go.Scatter(
                        x=group["projection_0"], y=group["projection_1"], mode='markers', showlegend=True,
                        marker=dict(size=8, color=group['Color'], coloraxis='coloraxis',
                                    opacity=group['Opacity'].iloc[0], symbol=group['symbol'].iloc[0],
                                    line=dict(color='black', width=2 if name[2] != 0 else 0)),
                        customdata=[indices[i] for i in group.index],
                        name=f"select category {'Selected' if name[2] > 0 else ''}" if Category is None else f"{original_string}{'filtered' if name[1] < 1 else ''}{' selected' if name[2] > 0 else ''}"))
            fig.update_layout(
                coloraxis=dict(colorscale=COLORSCALE,
                               colorbar=dict(len=1, x=0.0, y=-0.5, xanchor='left', orientation='h')),
                annotations=[dict(x=0.5, y=-0.15, xanchor='center', yanchor='bottom', showarrow=False,
                                  text=f"colorscale: {Colorscaling}", xref="paper", yref="paper")],
                scene=dict(xaxis=dict(title='t-SNE 1', showticklabels=False),
                           yaxis=dict(title='t-SNE 2', showticklabels=False),
                           zaxis=dict(title='t-SNE 3', showticklabels=False)), title="t-SNE plot of the design space")
        return fig

    @app.callback([Output('PC', 'figure')],
                  [Input('data_frames', 'data'),
                   Input('Dropdown_analysis', 'value'), ])
    def update_PC_Plot(dfs, outputs):
        """
        Visualize and update the Parallel Coordinates (PC) Plot based on user inputs.

        :param dfs: JSON representation of dataframes.
        :param outputs: List of selected output values.
        :param BINS: Amount of bins to be plotted.
        :return: Plotly figure for the PC Plot.
        """
        dataframes = json.loads(dfs)
        df_input_applied = pd.read_json(dataframes['df_input_filter'], orient='split')
        df_f = pd.read_json(dataframes['df_n_filter'], orient='split')
        fig = go.Figure()
        categorical_columns = ['domain_1', 'domain_2', 'domain_3', 'youngPen', 'a_p', 'manu_type', 'L_ManuP']
        dimensions = [dict(values=df_input_applied[label], label=label) for label in categorical_columns]

        # Adding numerical output as a dimension after binning
        for output in outputs:
            df_f[output + '_binned'], bins = pd.cut(df_f[output], bins=4, labels=False, retbins=True)
            bin_labels = []
            for i in range(BINS):
                bin_data = df_f[(df_f[output + '_binned'] == i)][output]
                mean_value = bin_data.mean() / df_f[output].max()
                bin_labels.append(f'[{round(bins[i], 2)}, {round(bins[i + 1], 2)}): Mean={mean_value:.2f}')
            dimensions.append({'label': output, 'values': df_f[output + '_binned'], 'ticktext': bin_labels})
        if outputs != []:
            line = {'color': df_f[output + '_binned'], 'colorscale': COLORSCALE}
        else:
            line = {}
        fig.add_trace(go.Parcats(dimensions=dimensions, line=line, ))
        return (fig,)

    @app.callback(
        [Output('clicked_indices', 'data'),
         Output('multi-select', 'value')],
        [Input('tsne-plot', 'clickData'),
         Input('3D_scatterplot', 'clickData'),
         Input('multi-select', 'value'),
         Input('clicked_indices', 'data'), ])
    def update_clicked_indices(TSNE_click, Scatter_click, multi_select, clicked_indices):
        """
        Update the clicked indices based on user interactions in the t-SNE plot, 3D scatterplot, and multi-select dropdown.

        :param TSNE_click: Click data from the t-SNE plot.
        :param Scatter_click: Click data from the 3D scatterplot.
        :param multi_select: string representation of clicked indices.
        :param clicked_indices: list with clicked indices.
        :return: Updated clicked indices and string representation of those values
        """
        if clicked_indices is None:
            c_indices = []
        else:
            c_indices = clicked_indices
        ctx = dash.callback_context
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == 'tsne-plot' and TSNE_click is not None:
            clicked_index = TSNE_click['points'][0]['customdata']
            c_indices.append(clicked_index)
        elif input_id == '3D_scatterplot' and Scatter_click is not None:
            clicked_index = Scatter_click['points'][0]['customdata']
            c_indices.append(clicked_index)
        elif input_id == 'multi-select':
            if set(multi_select) == set(c_indices):
                raise PreventUpdate
            else:
                c_indices = [int(i) for i in multi_select if i]
        c_indices = list(set(c_indices))
        multi_select_values = [str(x) for x in c_indices]
        return c_indices, multi_select_values

    @app.callback(
        [Output('Visualization', 'children'),
         Output('stats_plot', 'children'), ],
        [Input('clicked_indices', 'data'),
         Input('data_frames', 'data')])
    def update_visualizations(clicked_indices, dfs):
        """
        Visualize and update the bridges and corresponding radial barplot.

        :param clicked_indices: List of clicked indices.
        :param dfs: JSON data containing dataframes.
        :return: Tuple containg a list containing the desired bridge visualizations and barplots
        """

        # Required variables
        dataframes = json.loads(dfs)
        df_n = pd.read_json(dataframes['df_n'], orient='split')
        # Empty container for the bridge visualizations
        Visualization = []
        # Empty container for the radial barplots
        stats_plot = []

        # for each value in clicked indices create a bridge visualization and radial barplot
        for i in clicked_indices:
            vis_data = [go.Scatter(x=x_values, y=np.array([max_domain] * 6), mode='lines', name='Max_allowable domain',
                                   line=dict(color='grey', width=1, dash='dash')),
                        go.Scatter(x=x_values, y=np.array([min_domain] * 6), mode='lines', name='Min_allowable domain',
                                   line=dict(color='grey', width=1, dash='dash')),
                        go.Scatter(x=x_values, y=np.array([0] * 6), mode='lines', name='Loading Path',
                                   line=dict(color='red', width=2))]
            ctc = np.array(df3.loc[i, 'ctc'])
            nodes = np.array(df3.loc[i, 'nodes'])
            Heights = np.array(df3.loc[i, 'Heights'])
            Vmax_scale = df3.loc[i, 'Vmax_scale']
            Forces = np.array(df3.loc[i, 'Forces'])
            for j, (start, end) in enumerate(zip(ctc[0], ctc[1])):
                x = [nodes[0, start], nodes[0, end]]
                y = [nodes[1, start], nodes[1, end]]
                Height = Heights[j] / (50 * Vmax_scale)
                line_color = 'blue' if Forces[j] < 0 else 'red'
                # plot each bar as individual line on the plot
                vis_data.insert(2, go.Scatter(x=x, y=y, mode='lines', line=dict(color=line_color, width=Height)))
            input_values = df_input.loc[i]
            # Create a table containing the a string representation of the input values of the plotted bridge
            table_strings = [f"<b>{name}</b>: {value}" for name, value in input_values.items()]
            y_positions = [1 - i / (len(table_strings) + 1) for i in range(1, len(table_strings) + 1)]
            # add the table strings as annotation to the plot.
            vis_layout = go.Layout(template='gridon', title='Selected bridge, index =' + str(i), showlegend=False,
                                   yaxis=dict(showticklabels=False, visible=False, showgrid=True),
                                   xaxis=dict(showticklabels=False, visible=False, showgrid=True),
                                   margin=dict(l=200, r=20, t=100, b=100),
                                   annotations=[
                                       dict(x=-0.5, y=y, showarrow=False, text=line, xref="paper", yref="paper",
                                            font=dict(size=14)) for line, y in zip(table_strings, y_positions)])
            # append the created visualization to visualization container
            Visualization.append(html.Div(dcc.Graph(figure=go.Figure(data=vis_data, layout=vis_layout))))
            barcolor = ['rgba(214, 39, 40,0.9)' if x < 0 else 'rgba(31, 119, 180,0.9)' for x in
                        df_n.loc[i, columns_to_plot]]
            # Create the radial barplot of the corresponding clicked index
            Barplot_Data = go.Barpolar(r=df_n.loc[i, columns_to_plot], theta=columns_to_plot,
                                       marker=dict(color=barcolor), name='Output values',
                                       hovertext=[f"Value: {value}" for value in df3.loc[i, columns_to_plot]])
            barplot_layout = go.Layout(template='plotly_white', polar=dict(radialaxis=dict(showticklabels=False)))
            zeroline = go.Barpolar(r=[0.05 for x in range(len(columns_to_plot))], theta=columns_to_plot,
                                   marker=dict(color='black'), name='0 line')
            Scatterbarplot = go.Figure()
            Scatterbarplot.add_trace(zeroline)
            Scatterbarplot.add_trace(Barplot_Data)
            Scatterbarplot.update_layout(barplot_layout)
            stats_plot.append(html.Div(dcc.Graph(figure=Scatterbarplot)))
        return Visualization, stats_plot