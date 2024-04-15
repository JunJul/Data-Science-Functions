# split data types into numerical and categorical data
    # returns two items
def split_types(data):
    numerical_features = data.select_dtypes(["int64", "float64"])
    categorical_features = data.select_dtypes(["object", "category"])
    
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