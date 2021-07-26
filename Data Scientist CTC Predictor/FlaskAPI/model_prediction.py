import numpy as np

## Eg features: 3.0, 2000, 1, 'Health Care', 'Public', 'data analyst', '0', 'Junior', ['python', 'tableau']
def predict_salary(model, rating_scaler, company_founded_scaler, features):
    rating = features.get('rating')
    founded = features.get('founded')
    competitors = features.get('competitors')
    sector = features.get('sector')
    ownership = features.get('ownership')
    job_title = features.get('job_title')
    job_in_headquarters = features.get('job_in_headquarters')
    job_seniority = features.get('job_seniority')
    job_skills = features.get('job_skills')

    prediction_input = list()
    prediction_input.append(rating_scaler.transform(np.array(rating).reshape(1, -1)))
    prediction_input.append(company_founded_scaler.transform(np.array(founded).reshape(1, -1)))
    prediction_input.append(competitors)

    sector_columns = ['sector_Biotech & Pharmaceuticals', 'sector_Health Care',
                    'sector_Business Services','sector_Information Technology']
    temp = list(map(int, np.zeros(shape=(1, len(sector_columns)))[0]))
    for index in range(0, len(sector_columns)):
        if sector_columns[index] == 'sector_' + sector :
            temp[index] = 1
            break
    prediction_input = prediction_input + temp

    if ownership == 'Private':
        prediction_input.append(1)
    else:
        prediction_input.append(0)


    job_title_columns = ['job_title_data scientist', 'job_title_data analyst']
    temp = list(map(int, np.zeros(shape=(1, len(job_title_columns)))[0]))
    for index in range(0, len(job_title_columns)):
        if job_title_columns[index] == 'job_title_' + job_title:
            temp[index] = 1
            break
    prediction_input = prediction_input + temp

    prediction_input.append(job_in_headquarters)

    job_seniority_map = {'Other': 0, 'Junior': 1, 'Senior': 2}
    prediction_input.append(job_seniority_map[job_seniority])

    temp = list(map(int, np.zeros(shape=(1, 4))[0]))
    if 'Excel' in job_skills:
        temp[0] = 1
    if 'Python' in job_skills:
        temp[1] = 1
    if 'Tableau' in job_skills:
        temp[2] = 1
    if 'SQL' in job_skills:
        temp[3] = 1
    prediction_input = prediction_input + temp

    salary = model.predict([prediction_input])[0]
    return {'min': (int(salary)*1000)-9000, 'max': (int(salary)*1000)+9000}