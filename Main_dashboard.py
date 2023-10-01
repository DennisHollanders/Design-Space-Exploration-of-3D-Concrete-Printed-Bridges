import pandas as pd
import dash
import dash_bootstrap_components as dbc
from LayoutCallbacks import layout, callbacks, preprocess_data

#initialize the dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Change directory paths
INPUT_FILEPATH: str = r'C:\Users\denni\OneDrive\Documenten\Master Thesis - Design Space Exploration - git\Design-Space-Exploration-of-3D-Concrete-Printed-Bridges\Design Space Data_Input.json'
OUTPUT_FILEPATH: str = r'C:\Users\denni\OneDrive\Documenten\Master Thesis - Design Space Exploration - git\Design-Space-Exploration-of-3D-Concrete-Printed-Bridges\Design Space Data_Output.json'

#constant values/ Column names/file names/ Weights
STEPSIZE_Y: int = 1
STEPSIZE_X: int = 2
X_STEPS:list[int] = [0, STEPSIZE_X, STEPSIZE_X * 2, STEPSIZE_X * 3, STEPSIZE_X * 4]
COLUMNS_TO_NORMALIZE: list[str] = ['MSU', 'MPU', 'Vmax ', 'SF_v', 'SF_h', 'Max_def', 'Mean_def', 'Epoch', 'a_p']
CATEGORICAL_COLUMNS: list[str] = ['youngPen', 'manu_type', 'L_ManuP']
OTHER_COLUMNS: list[str] = ['domain_1', 'domain_2', 'domain_3']
INPUT_COLUMNS: list[str] = ['domain_1', 'domain_2', 'domain_3', 'youngPen', 'a_p','manu_type', 'L_ManuP']

Weights: dict = {}
Weights["Index"] = COLUMNS_TO_NORMALIZE
Weights["Cost"] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
Weights["Manu"] = [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0]
Weights["Eco"] =  [0, 0, 0, 0, 0, 0, 0, 1, 0]
CME: list[str] = ["Cost", "Manu", "Eco"]
DF_WEIGHTS: pd.DataFrame = pd.DataFrame.from_dict(Weights, orient='columns')

def load_data(input_path:str , output_path: str):
    """
    Load input and output data from json files and concatenate them
    :param input_path: The file path to the input JSON file.
    :param output_path: The file path to the output JSON file.
    :return: A DataFrame containing concatenated data from both input and output JSON files.
    """
    df1 = pd.read_json(input_path, lines=True)
    df2 = pd.read_json(output_path, lines=True)
    df1.reset_index(drop=True, inplace=True)
    df2.loc[:, 'Epoch'] = df2['Epoch'].apply(lambda x: x[-1])
    df2.reset_index(drop=True, inplace=True)

    return pd.concat([df1, df2], axis=1)
def main():
    """"
    - Run load data
    - Run preprocess data in LayoutCallbacks script
    - Set the layout
    - Run callback function
    """
    df = load_data(INPUT_FILEPATH,OUTPUT_FILEPATH)
    df_n, df_input, max_domain, min_domain = preprocess_data(df,INPUT_COLUMNS, COLUMNS_TO_NORMALIZE)
    args = [df_n, df_input, max_domain, min_domain,STEPSIZE_Y, INPUT_COLUMNS, COLUMNS_TO_NORMALIZE, DF_WEIGHTS]
    app.layout = layout(*args)
    callbacks(app,*args, CME,X_STEPS,OTHER_COLUMNS,df,CATEGORICAL_COLUMNS)


if __name__ == "__main__":
    """
    Run main function and the server 
    """
    main()
    app.run_server(debug=True)