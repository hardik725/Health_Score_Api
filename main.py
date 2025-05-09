import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def generate_synthetic_health_data(num_samples=5000, seed=42):
    """
    Generate synthetic health data with realistic correlations.
    
    Parameters:
    - num_samples: Number of data points to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with synthetic health data
    """
    np.random.seed(seed)
    
    # Generate basic demographic data
    age = np.random.randint(18, 75, num_samples)
    gender = np.random.choice(['Male', 'Female'], num_samples)
    
    # Generate height data based on gender (in cm)
    height_male = np.random.normal(175, 8, num_samples)  # Mean 175cm for males
    height_female = np.random.normal(162, 7, num_samples)  # Mean 162cm for females
    height = np.where(gender == 'Male', height_male, height_female)
    
    # Generate weight data correlated with height and gender (in kg)
    weight_base = (height - 100) * 0.9  # Base weight calculation
    weight_noise_male = np.random.normal(0, 10, num_samples)  # More variance for males
    weight_noise_female = np.random.normal(0, 8, num_samples)  # Less variance for females
    weight_noise = np.where(gender == 'Male', weight_noise_male, weight_noise_female)
    weight = weight_base + weight_noise
    weight = np.maximum(40, weight)  # Ensure minimum weight
    
    # Screen time in hours (daily)
    screen_time = np.random.exponential(scale=3, size=num_samples)
    screen_time = np.minimum(screen_time, 16)  # Cap at 16 hours
    
    # Viewing distance in inches
    viewing_distance = np.random.normal(20, 8, num_samples)
    viewing_distance = np.maximum(viewing_distance, 6)  # Minimum 6 inches
    
    # Device used
    device_used = np.random.choice(['Smartphone', 'Tablet', 'Laptop', 'Desktop', 'TV'], 
                                 p=[0.4, 0.2, 0.2, 0.1, 0.1], 
                                 size=num_samples)
    
    # Video brightness (scale 1-10)
    video_brightness = np.random.normal(6, 2, num_samples)
    video_brightness = np.clip(video_brightness, 1, 10)
    
    # Audio level (scale 1-10)
    audio_level = np.random.normal(5, 2, num_samples)
    audio_level = np.clip(audio_level, 1, 10)
    
    # Sleep schedule (hours)
    sleep_schedule = np.random.normal(7, 1.5, num_samples)
    sleep_schedule = np.clip(sleep_schedule, 3, 10)
    
    # Headache (No, Minor, Major)
    # We'll make this correlated with screen time, brightness, and sleep
    headache_prob = 0.2 + 0.05 * screen_time - 0.1 * sleep_schedule + 0.04 * video_brightness
    headache_prob = np.clip(headache_prob, 0.05, 0.95)
    
    headache = []
    for prob in headache_prob:
        if prob < 0.5:
            headache.append('No')
        elif prob < 0.8:
            headache.append('Minor')
        else:
            headache.append('Major')
    
    # Create health score based on all parameters
    # Higher score is better health (0-100 scale)
    health_score = (
        # Age factor (younger is better, but with diminishing returns)
        12 * (1 - np.clip((age - 18) / 57, 0, 1)**0.7) + 
        
        # BMI factor (penalize high and low BMI)
        12 * (1 - 0.15 * np.abs((weight / ((height/100)**2) - 22) / 10)) +
        
        # Screen time factor (less is better)
        12 * (1 - screen_time / 16) +
        
        # Viewing distance factor (more is better, up to a point)
        7 * (1 - np.clip(np.abs(viewing_distance - 22) / 10, 0, 1)) +
        
        # Sleep factor (more is better, up to 8 hours)
        20 * np.clip(sleep_schedule / 8, 1, 1.25) +
        
        # Device factor
        np.where(device_used == 'Smartphone', 3,
                np.where(device_used == 'Tablet', 5,
                        np.where(device_used == 'Laptop', 6,
                                np.where(device_used == 'Desktop', 8, 9)))) +
        
        # Brightness factor (middle values are better)
        8 * (1 - abs(video_brightness - 5.5) / 5.5) +
        
        # Audio factor (middle values are better)
        6 * (1 - abs(audio_level - 5) / 5) +
        
        # Headache factor
        np.where(np.array(headache) == 'No', 20, 
                np.where(np.array(headache) == 'Minor', 10, 0))
    )
    
    # Add some random noise to health score
    health_score = health_score + np.random.normal(0, 2, num_samples)
    
    # Ensure health score is between 0 and 100
    health_score = np.clip(health_score, 0, 100)
    
    # Round numerical values for readability
    height = np.round(height, 1)
    weight = np.round(weight, 1)
    screen_time = np.round(screen_time, 2)
    viewing_distance = np.round(viewing_distance, 1)
    video_brightness = np.round(video_brightness, 1)
    audio_level = np.round(audio_level, 1)
    sleep_schedule = np.round(sleep_schedule, 1)
    health_score = np.round(health_score, 1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'ScreenTime': screen_time,
        'ViewingDistance': viewing_distance,
        'DeviceUsed': device_used,
        'VideoBrightness': video_brightness,
        'AudioLevel': audio_level,
        'SleepSchedule': sleep_schedule,
        'Headache': headache,
        'HealthScore': health_score
    })
    
    return data

def visualize_data(data):
    """
    Create visualizations to explore the generated data
    
    Parameters:
    - data: DataFrame with synthetic health data
    """
    # Set style
    sns.set(style="whitegrid")
    
    # Create plot figure
    plt.figure(figsize=(15, 10))
    
    # Distribution of health score
    plt.subplot(2, 3, 1)
    sns.histplot(data['HealthScore'], kde=True)
    plt.title('Distribution of Health Score')
    
    # Health score by gender
    plt.subplot(2, 3, 2)
    sns.boxplot(x='Gender', y='HealthScore', data=data)
    plt.title('Health Score by Gender')
    
    # Screen time vs health score
    plt.subplot(2, 3, 3)
    sns.scatterplot(x='ScreenTime', y='HealthScore', hue='Gender', data=data, alpha=0.6)
    plt.title('Screen Time vs Health Score')
    
    # Age vs health score
    plt.subplot(2, 3, 4)
    sns.scatterplot(x='Age', y='HealthScore', hue='Headache', data=data, alpha=0.6)
    plt.title('Age vs Health Score')
    
    # Sleep vs health score
    plt.subplot(2, 3, 5)
    sns.scatterplot(x='SleepSchedule', y='HealthScore', hue='Headache', data=data, alpha=0.6)
    plt.title('Sleep Schedule vs Health Score')
    
    # Health score by device and headache
    plt.subplot(2, 3, 6)
    sns.boxplot(x='DeviceUsed', y='HealthScore', hue='Headache', data=data)
    plt.title('Health Score by Device and Headache')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def train_test_split_data(data, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split

    # Separate features and target
    X = data.drop('HealthScore', axis=1)
    y = data['HealthScore']

    # Define all expected categories manually
    X['Gender'] = pd.Categorical(X['Gender'], categories=['Female', 'Male'])
    X['DeviceUsed'] = pd.Categorical(X['DeviceUsed'], categories=['Smartphone', 'Tablet', 'Laptop', 'Desktop', 'TV'])

    # One-hot encode, keeping all expected columns
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train different models and return the best one
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    
    Returns:
    - best_model: The best performing model
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_score = -float('inf')
    best_model = None
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = model
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test target
    
    Returns:
    - Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def save_data_to_csv(data, filename='synthetic_health_data.csv'):
    """
    Save generated data to CSV file
    
    Parameters:
    - data: DataFrame with synthetic health data
    - filename: Name of output file
    """
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic health data...")
    data = generate_synthetic_health_data(num_samples=2000)
    
    # Save data to CSV
    save_data_to_csv(data)
    
    # Display first few rows
    print("\nFirst 5 rows of generated data:")
    print(data.head())
    
    # Summary statistics
    print("\nSummary statistics:")
    print(data.describe())
    
    # Visualize data
    print("\nCreating visualizations...")
    visualize_data(data)
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test = train_test_split_data(data)
    
    # Train model
    print("\nTraining models...")
    best_model = train_model(X_train, y_train)
    #Save the expected Column of the Model
    expected_columns = X_train.columns.tolist()
    with open('expected_columns.pkl','wb') as f:
        pickle.dump(expected_columns, f)
    print("Expected Columns Saved to expected_columns.pkl")
    # Save the trained model to a pickle file
    print("Saving model to health_model.pkl...")
    with open('health_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        print("Model saved successfully to health_model.pkl")
    
    # Evaluate model
    print("\nEvaluating best model on test data:")
    metrics = evaluate_model(best_model, X_test, y_test)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nDone!")