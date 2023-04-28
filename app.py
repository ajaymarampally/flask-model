from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

model = pickle.load(open('model2.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    gre_total = request.form.get('grescore')
    if gre_total is not None:
        gre_total = float(gre_total)
    else:
        return "Error: GRE Score not provided"

    ielts_or_toefl = request.form.get('EP')
    if ielts_or_toefl is not None:
        ielts_or_toefl = float(ielts_or_toefl)
    else:
        return "Error: English Proficiency not provided"

    grade = request.form.get('gpa')
    if grade is not None:
        grade = float(grade)
    else:
        return "Error: Undergraduation GPA not provided"

    publications = request.form.get('publications')
    if publications is not None:
        publications = int(publications)
    else:
        return "Error: Number of relevant publications not provided"

    input_data = [[gre_total, ielts_or_toefl, grade, publications]]
    probability_admit = model.predict_proba(input_data)[0][0]
    return "Your Probability of getting admitted is: " + str(probability_admit*100)+'%'


@app.route('/recommend', methods=['POST'])
def run_knn(weights, data, test_data, df):
    data_scaled = data.copy(deep=True)
    for feature in weights:
        data_scaled[feature] = data_scaled[feature] * weights[feature]

    k = 5 # number of nearest neighbors
    model = NearestNeighbors(n_neighbors=k+1)  
    model.fit(data_scaled)


    test_data_scaled = test_data.copy(deep=True)
    for feature in weights:
        test_data_scaled[feature] = test_data_scaled[feature] * weights[feature]
    distances, indices = model.kneighbors(test_data_scaled)

    universities = []
    for i in range(len(test_data)):
        row_indices = indices[i]  # exclude the first neighbor (itself)
        if len(row_indices) < k:
            # Skip this test point if there are less than k neighbors
            universities.append([])
            continue
        row_distances = distances[i]  # exclude the first neighbor (itself)
        weighted_distances = 1 / (row_distances ** 2)  # apply weights to the distances
        weighted_distances[~np.isfinite(weighted_distances)] = 1e10  # replace inf and -inf with a large number
        weighted_indices = [(j, weighted_distances[np.where(row_indices == j)[0][0]]) for j in row_indices]
        #print("Weighted indices:", weighted_indices,)
        university_scores = {}
        for j, w in weighted_indices:
            university = df.loc[j, "University"]
            if university in university_scores:
                university_scores[university] += w
            else:
                university_scores[university] = w
        top_universities = sorted(university_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_universities = [u[0] for u in top_universities]
        universities.append(top_universities)
    # Return the recommended universities
    return distances, indices, universities

@app.route('/recommend2', methods=['POST'])
def recommend_universities():
    #add here
    
    gre_total = request.form.get('grescore')
    if gre_total is not None:
        gre_total = float(gre_total)
    else:
        return "Error: GRE Score not provided"

    ielts_or_toefl = request.form.get('EP')
    if ielts_or_toefl is not None:
        ielts_or_toefl = float(ielts_or_toefl)
    else:
        return "Error: English Proficiency not provided"

    grade = request.form.get('gpa')
    if grade is not None:
        grade = float(grade)
    else:
        return "Error: Undergraduation GPA not provided"

    publications = request.form.get('publications')
    if publications is not None:
        publications = int(publications)
    else:
        return "Error: Number of relevant publications not provided"

    input_data = [[gre_total, ielts_or_toefl, grade, publications]]
    probability_admit = model.predict_proba(input_data)[0][0]

    
    weights = {'GRE_Total': 0.3, 'CGPA': 0.3, 'toeflScore': 0.2, 'researchExp': 0.1}
    testpoint = pd.DataFrame({"researchExp": publications, "toeflScore": ielts_or_toefl, "GRE_Total": gre_total, "CGPA": grade}, index=[0])

    # Load the training data and filter based on the user's inputs
    df = pd.read_csv(r'df_filtered.csv')
    #df_filtered = df[(df['Research'] == research_exp) & (df['TOEFL'] == toefl_score) & (df['GRE'] == gre_total) & (df['CGPA'] == cgpa)]
    
    required_columns_df=['researchExp','toeflScore','GRE_Total','CGPA','University']
    df=df[required_columns_df]
    X_train = df.drop(['University'], axis=1)
    distances, indices, results = run_knn(weights, X_train ,testpoint, df)
    string1 = "Other Universities that are recommended to your profile:"
    for i in results:
        string2 = ", ".join(i)
        
    content = "Your Probability of getting admitted to IU is: " + str(probability_admit*100)+'%' 
    content2 = string1
    content3 = string2
    return render_template('homepage.html', content=content, content2=content2,content3=content3)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)