import pandas as pd



def data_clean():
    '''
    Clean data for later use:
        - Exploratory analysis
        - Modeling for salary prediction
        - ...
    '''
    # read .csv data in as a data frame
    df = pd.read_csv('ds_list_glassdoor.csv')
    # print(df.shape)

    # parse salary to get
    # text only company name
    # state field
    # age of company
    # parsing of job description (python, sql, etc.)

    # add hourly and employer_provided salary
    df['hourly'] = df['Salary Estimate'].apply(
        lambda x: 1 if 'per hour' in x.lower() else 0
    )
    df['employer_provided'] = df['Salary Estimate'].apply(
        lambda x: 1 if 'employer provided salary:' in x.lower() else 0
    )
    
    # remove null data
    df = df[df['Salary Estimate'] != '-1']
    # print(df.shape)

    # use lambda to remove (Glassdoor est.)
    salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
    # print(salary.head())

    # use lambda to remove $, -, and K
    remove_Kdollar = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
    # print(remove_Kdollar.head())

    # remove 'per hour' and 'employer provided salary'
    remove_hr = remove_Kdollar.apply(
        lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', '')
    )

    # obtain min, max, and avg salary
    df['min_salary'] = remove_hr.apply(lambda x: int(x.split('-')[0]))
    df['max_salary'] = remove_hr.apply(lambda x: int(x.split('-')[1]))
    df['avg_salary'] = (df.min_salary + df.max_salary) / 2
    # print(df.head())

    # parse company name to text only
    # An example of the original format is: "Amazon 4.5"
    df['company_txt'] = df.apply(
        lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1
    )

    # US state  
    df['us_state'] = df['Location'].apply(lambda x: x.split(',')[1])
    # print(df['us_state'].value_counts())
    # if the location of the job is the same as the headquater of the company
    df['same_state'] = df.apply(
        lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1
    )

    # age of company 
    df['age'] = df['Founded'].apply(lambda x: x if x <1 else 2020 - x)

    # whether python is mentioned in the job description
    df['python'] = df['Job Description'].apply(
        lambda x: 1 if 'python' in x.lower() else 0
    )

    # whether R is mentioned in JD 
    df['R'] = df['Job Description'].apply(
        lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0
    )
    # print(df['R'].value_counts())

    # whether spark is mentioned in JD
    df['spark'] = df['Job Description'].apply(
        lambda x: 1 if 'spark' in x.lower() else 0
    )
    df['spark'].value_counts()

    # whether aws is mentioned in JD 
    df['aws'] = df['Job Description'].apply(
        lambda x: 1 if 'aws' in x.lower() else 0
    )
    df['aws'].value_counts()

    # whether excel is mentioned in JD
    df['excel'] = df['Job Description'].apply(
        lambda x: 1 if 'excel' in x.lower() else 0
    )
    df['excel'].value_counts()

    # print(df.columns)
    
    df_out = df.drop(['Unnamed: 0'], axis =1)

    df_out.to_csv('data_clean.csv', index = False)