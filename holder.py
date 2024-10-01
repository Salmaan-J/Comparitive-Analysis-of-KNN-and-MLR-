selected_features = []
    master_feature_list = list(selected_features)
    baseline_performance = None
    feature_improvements = {}

    # Loop through all features and evaluate their individual improvement
    for feature in x_train.columns:
        if feature not in master_feature_list:
            # Temporary list of current features + the new one
            current_features = master_feature_list + [feature]
            
            # Train the model with the current features
            mlr_model = LinearRegression()
            mlr_model.fit(x_train[current_features], y_train)
            
            # Evaluate using cross-validation (to avoid overfitting)
            cv_scores = cross_val_score(mlr_model, x_train[current_features], y_train, cv=5, scoring='neg_mean_squared_error')
            current_performance = -cv_scores.mean()

            # If baseline_performance is not set, initialize it with the first feature set's performance
            if baseline_performance is None:
                baseline_performance = current_performance
                print(f"Setting baseline performance to {current_performance} as the first comparison")

            # Calculate performance improvement
            performance_improvement = baseline_performance - current_performance
            
            # Store the feature and its improvement
            feature_improvements[feature] = performance_improvement
            print(f"Feature: {feature}, Improvement: {performance_improvement}, Baseline: {baseline_performance}, Current: {current_performance}")

    # Sort features by the largest performance improvement
    sorted_features = sorted(feature_improvements.items(), key=lambda x: x[1], reverse=True)

    # Select the top 5 features
    best_5_features = [feature for feature, improvement in sorted_features[:5]]
    print(f"Best 5 features: {best_5_features}")

    # Add these features to the master list and retrain the final model
    master_feature_list += best_5_features
    print(f"Final selected features: {master_feature_list}")

    # Train final model with the best features
    final_model = LinearRegression()
    final_model.fit(x_train[master_feature_list], y_train)

    # Evaluate final model on test data
    y_pred = final_model.predict(x_test[master_feature_list])
    final_performance = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(final_performance)
    final_r2 = r2_score(y_test, y_pred)
    final_evs = explained_variance_score(y_test, y_pred)

    print(f"Final model performance:")
    print(f"RMSE: {final_rmse}")
    print(f"R-squared: {final_r2}")
    print(f"Explained Variance Score: {final_evs}")


while True:
            try:
                n_neighbors = input("\nInsert the number of neighbors")
                n_neighbors = int(n_neighbors)
                break
            except ValueError:
                print("\nInvalid input. Please enter an integer.")
    knn_model = KNeighborsRegressor(n_neighbors)
    knn_model.fit(x_train,y_train)
    y_pred=knn_model.predict(x_test)
    model_folder_path = 'Model/'  # Specify the folder path where you want to save the model
    os.makedirs(model_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    model_file_path = os.path.join(model_folder_path, 'KNN.pkl')  # Specify the file path and name
    joblib.dump(knn_model, model_file_path)
    return y_pred