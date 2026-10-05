from flask import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import math
import ast
import io
import base64
import numpy
import json

app = Flask(__name__)

app.secret_key = "AARAV_AND_ISHA"  # Needed for part of functionality

#Import functions from Algorithms.py (encapsulation)
from .Algorithms import time_and_run_binaryKP
from .Algorithms import generate_input_data_binaryKP

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #Generate data/submit data/load data
        #redirect(url_for('index)) redirets the user back to the homepage (get method)
        input_data = request.form['input_data']
        if input_data == "generate_random":
            #Calls random data function
            weights, values, capacity = generate_input_data_binaryKP(int(request.form['number_of_items']), int(request.form['range_of_values']), float(request.form['correlation_strength']), float(request.form['capacity_load']))
        elif input_data == "input_manually":
            try:
                #Strips the input, splits the data by " ", and converts to integer lists
                weights = list(map(int, request.form['weights'].strip().split()))
                values = list(map(int, request.form['values'].strip().split()))
                capacity = int(request.form['capacity'].strip())
            except:
                session['error'] = "Please enter integers seperated by spaces for weights and values and a single integer for capacity."
                return redirect(url_for('index'))
        elif input_data == "import_csv":
            #Gets CSV file
            file = request.files.get("csv_file")
            if file and file.filename.endswith(".csv"):
                #Reads CSV file using pandas. Calls any error before converting
                df = pd.read_csv(file)
                if 'weights' not in df.columns:
                    session['error'] = "Column 'weights' not found in CSV."
                    return redirect(url_for('index'))
                elif 'values' not in df.columns:
                    session['error'] = "Column 'values' not found in CSV."
                    return redirect(url_for('index'))
                elif 'capacity' not in df.columns:
                    session['error'] = "Column 'capacity' not found in CSV."
                    return redirect(url_for('index'))
            else:
                session['error'] = "Not CSV File."
                return redirect(url_for('index'))
            
            try:
                weights = df['weights'].tolist()
                values = df['values'].tolist()
                capacity_list = df['capacity'].tolist()
                #Cleans capacity list so that actual value can be identified
                cleaned_list = [x for x in capacity_list if not math.isnan(x)]
            except:
                session['error'] = "Column 'weights', 'values' and 'capacity' all must contain integer values."
                return redirect(url_for('index'))

            if len(cleaned_list) == 0:
                #No value found in capacity
                session['error'] = "No valid (non-NaN) values found in 'capacity'."
                return redirect(url_for('index'))

            capacity = int(cleaned_list[0])
        
        if len(weights) != len(values):
            session['error'] = "The number of weights inputted doesn't match the number of values."
            return redirect(url_for('index'))
        
        # Store results in session or temp storage
        session['weights'] = weights
        session['values'] = values
        session['capacity'] = capacity
        
        session['number_of_items'] = request.form.get('number_of_items', 450)
        session['range_of_values'] = request.form.get('range_of_values', 200)
        session['correlation_strength'] = request.form.get('correlation_strength', 0.6)
        session['capacity_load'] = request.form.get('capacity_load', 0.75)

        # Redirect to GET route
        return redirect(url_for('index'))

    # GET request: load stored data or show empty
    weights = session.pop('weights', [])
    values = session.pop('values', [])
    capacity = session.pop('capacity', 0)
    error = session.pop('error', "")
    
    number_of_items = session.pop('number_of_items', 450)
    range_of_values = session.pop('range_of_values', 200)
    correlation_strength = session.pop('correlation_strength', 0.6)
    capacity_load = session.pop('capacity_load', 0.75)

    #Returns the index.html render template with the newly generated weights, value and capacity
    #Additionally stores the generate random data parameters if new data with same parameters needs to be generated
    return render_template(
    "index.html",
    weights=weights,
    values=values,
    capacity=capacity,
    number_of_items=number_of_items,
    range_of_values=range_of_values,
    correlation_strength=correlation_strength,
    capacity_load=capacity_load,
    error=error
    )
    
@app.route('/submit', methods=['POST'])
def submit():
    #.form[''] returns the value of the option from dropdown
    algorithm_type = request.form['algorithm_type']
    function_names = ['Randomised Search', 'Brute Force (iterative)', 'Brute Force (recursive)', 'Greedy Approximation', 'DP (Memoization)', 'DP (Tabulation)', 'Branch and Bound', 'Genetic Algorithm', 'Ant Colony Optimisation', 'Simulated Annealing', 'Tabu Search', 'Particle Swarm Optimisation']
    function_index = function_names.index(algorithm_type)
    #Efficiently converts string of weights and values into an actual python stored integer list
    weights = ast.literal_eval(request.form['weights'])
    values = ast.literal_eval(request.form['values'])
    capacity = int(request.form['capacity'])
    
    if len(weights) == 0:
        #No data submitted
        return render_template('index.html', weights=[], values=[], capacity=0, error="No data values submitted")

    #Dictionary of all algorithm parameters. All are grouped together, however only the selected algorithm's parameters will be used
    algorithm_data = {
        "time_limit" : int(request.form['time_limit']),
        "max_iterations_iterative" : int(request.form['max_iterations_iterative']),
        "max_iterations_recursive" : int(request.form['max_iterations_recursive']),
        "max_iterations_tabu_search" : int(request.form['max_iterations_tabu_search']),
        "tabu_list_size" : int(request.form['tabu_list_size']),
        "initial_temperature" : float(request.form['initial_temperature']),
        "minimum_temperature" : float(request.form['minimum_temperature']),
        "alpha" : float(request.form['alpha']),
        "max_iterations_SA" : int(request.form['max_iterations_SA']),
        "max_iterations_ACO" : int(request.form['max_iterations_ACO']),
        "ants" : int(request.form['ants']),
        "alpha_ACO" : int(request.form['alpha_ACO']),
        "beta_ACO" : int(request.form['beta_ACO']),
        "evapouration_rate" : float(request.form['evapouration_rate']),
        "max_iterations_GA" : int(request.form['max_iterations_GA']),
        "population_size_GA" : int(request.form['population_size_GA']),
        "selection_type_GA" : request.form['selection_type_GA'],
        "tournament_size_GA" : int(request.form['tournament_size_GA']),
        "cross_over_rate" : float(request.form['cross_over_rate']),
        "mutation_rate" : float(request.form['mutation_rate']),
        "elitism_rate" : float(request.form['elitism_rate']),
        "max_iterations_PSO" : int(request.form['max_iterations_PSO']),
        "population_size_PSO" : int(request.form['population_size_PSO']),
        "inertia" : float(request.form['inertia']),
        "cognitive_factor" : float(request.form['cognitive_factor']),
        "social_factor" : float(request.form['social_factor']),
        "steepness" : float(request.form['steepness']),
        "max_velocity" : float(request.form['max_velocity']),
        "is_greedy_SA" : True if "is_greedy_SA" in request.form.keys() else False,
        "is_greedy_tabu" : True if "is_greedy_tabu" in request.form.keys() else False
    }

    # Performs some processing with the form data
    #Calls algorithm on the loaded data
    generated_answer, total_weight, binary_string, elapsed_time, valid, fig = time_and_run_binaryKP(function_index, weights, values, capacity, algorithm_data)
    if type(fig) == type(numpy.array([0])):
        #If fig is from DP Tabulation, convert to json
        l_fig = fig.tolist()
        r_fig = json.dumps(l_fig)
    else:
        #If fig is a plotted graph, convert using base64 into an image that can be embedded in the html
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        r_fig =  base64.b64encode(output.getvalue()).decode('utf-8')

    #Dictionary of grouped comparative data that needs to be passed into submit.html
    Data = {
        "approach" : function_names[function_index],
        "total_value" : generated_answer,
        "total_weight" : total_weight,
        "time" : elapsed_time,
        "valid" : valid,
        "binary_string" : binary_string,
        "weights" : weights,
        "values" : values,
        "r_fig" : r_fig
    }

    return render_template("submit.html", Data=Data)