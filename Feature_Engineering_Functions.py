# split data types into numerical and categorical data
    # returns two items
def split_types(data):
    numerical_features = data.select_dtypes(["int64", "float64"])
    categorical_features = data.select_dtypes(["object", "category"])
    
    any_numerical_object = [i for i in categorical_features.columns if any(pd.to_numeric(categorical_features[i], errors='coerce').notnull())]

    numerical_features = pd.concat([numerical_features, categorical_features[any_numerical_object]], axis=1)
    categorical_features = categorical_features.drop(any_numerical_object, axis=1)
    return numerical_features, categorical_features

# scale data
def scale(x):
    scaler = StandardScaler()
    scaler.fit(x)
    scaled_x = scaler.transform(x)
    return scaled_x


# oversampling data
def oversampling(x, y):
    ros = RandomOverSampler()
    x, y = ros.fit_resample(x, y)
    return x, y

# undersampling data
def undersampling(x,y):
    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x,y)
    return x, y

# Encoding Functions
  # encode the categorical features into ordinal numbers
def ordinal_catfeatures_encoder(data):
    encoder = OrdinalEncoder()
    encoded_data = encoder.fit_transform(data)
    encoded_dataframe =  pd.DataFrame(encoded_data)
    encoded_dataframe.columns = data.columns
    
    return encoded_dataframe

  # one-hot encoding
    # encode the categorical features into dummy variables and return encoder as well
def onehot_eoncoder(data):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(data)
    encoded_dataframe = pd.DataFrame(encoded_data.toarray())
    
    return encoded_dataframe, encoder

  # Frequency encoding
def frequency_encoder(data, is_normalize = True):
    length = data.shape[1]
    columns = data.columns
    new_data = data.copy()
    
    for i in range(length):
        frequency = new_data[columns[i]].value_counts(normalize=True)
        new_data[columns[i]] = new_data[columns[i]].map(frequency)
    
    return new_data

# count unique values in each categorical variable
def count_uniques(c_data):
  return pd.DataFrame(c_data.nunique(), columns=["count"])

# get what and how many unqie values in each categorical variable
def check_uniques(c_data):
    columns = c_data.columns
    uniqe_values = c_data[columns[0]].value_counts()

    columns = columns[1:]

    diction = { "Variables": uniqe_values.index.name, "Value": uniqe_values.index,
            "Count": uniqe_values}

    df = pd.DataFrame(diction)
    df = df.set_index(["Variables", "Value"])

    diction.clear()

    for i in columns:
        uniqe_values = c_data[i].value_counts()
        diction = { "Variables": uniqe_values.index.name, "Value": uniqe_values.index,
            "Count": uniqe_values}

        df_copy = pd.DataFrame(diction)
        diction.clear()

        df_copy = df_copy.set_index(["Variables", "Value"])
        df = pd.concat([df, df_copy], axis=0)

    return df


# check how many nonNumerical values in a numerical variable 
def check_nonNumerical(n_data, column):
  logicals = pd.to_numeric(numerical_variables[column], errors='coerce').notnull()
  index = np.where(logicals == False)[0]
  if len(index) > 0:
    return n_data.iloc[index, :]
  
  return "No non-numerical values"