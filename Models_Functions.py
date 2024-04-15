# The pipelineModel function returns a model, confusion matrix, summary table, and validation score.

def pipelineModel(model, x, y, ordinalEncoder=True):
    # shuffle and set corss validation to n = 5 folds
    cv = ShuffleSplit(n_splits=5, test_size = 0.2, random_state=0)
    
    # divide data into train and test
    train_x, test_x, train_y, test_y = train_test_split(
      x,
      y,
      test_size=0.2,
      shuffle=True
      )
    
    # impute missing values with median in numerical variables
    numerical_transformer = SimpleImputer(strategy="median")
    
    # impute missing values with most frequent value in categorical variables
    # encode categorical variable by ordinal numbers
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_title = "One Hot Encoding"
    
    if ordinalEncoder:
        encoder = OrdinalEncoder()
        encoder_title = "Oridnal Encoding"
        
    categorical_transformer = Pipeline(
      steps=[
      ("Imputer", SimpleImputer(strategy="most_frequent")),
      (encoder_title, encoder)
      ]
    )
    
    # use columnTransformer to combine preprocessing steps in both numerical and categorical variables
    preprocessor = ColumnTransformer(
      transformers=[
          ("num", numerical_transformer, numerical_cols),
          ("cat", categorical_transformer, categorical_cols),
      ]
    )
    
    # after preprocess data, we scale variables into the same unit, then create a model
    my_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
         ('scaler', StandardScaler()),
        ("model", model)
    ]
    )
    
    my_pipeline.fit(train_x, train_y)
    
    # get predictions
    predictions = my_pipeline.predict(test_x)
    # get confusion matrix
    cm = confusion_matrix(predictions, test_y)
    # produce a summary of the model
    report = classification_report(predictions, test_y)
    # test the model by corss validation
    validation_score = cross_val_score(my_pipeline, train_x, train_y, cv=cv)
    
    return my_pipeline, cm, report, validation_score