import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import tensorflow as tf

class DrivingAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.score_scaler = MinMaxScaler()
        self.sequence_length = 60  # 1 minute of data at 1Hz
        self.model = None
        
    def preprocess_data(self, df):
        """Preprocess data with improved error handling"""
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Convert timestamp to datetime with explicit timezone handling
        df['GPS Time'] = pd.to_datetime(df['GPS Time'].str.replace(' CDT ', ' '), format='%a %b %d %H:%M:%S %Y', errors='coerce')
        
        # Ensure all required columns exist and are numeric
        numeric_columns = [
            'GPS Speed (Meters/second)',
            'Engine RPM(rpm)',
            'Engine Load(%)',
            'Throttle Position(Manifold)(%)',
            'Mass Air Flow Rate(g/s)',
            'Engine Coolant Temperature(°C)',
            ' G(x)', ' G(y)', ' G(z)'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = 0.0  # Default value if column doesn't exist
        
        # Calculate acceleration using diff and handle NaN values
        df['speed'] = df['GPS Speed (Meters/second)'].copy()
        df['acceleration'] = df['speed'].diff() / 0.1  # assuming 0.1s between readings
        df['acceleration'] = df['acceleration'].fillna(0)
        
        # Calculate jerk with proper filling
        df['jerk'] = df['acceleration'].diff() / 0.1
        df['jerk'] = df['jerk'].fillna(0)
        
        # Calculate g-force magnitude
        df['g_force_magnitude'] = np.sqrt(
            df[' G(x)'].fillna(0)**2 + 
            df[' G(y)'].fillna(0)**2 + 
            df[' G(z)'].fillna(0)**2
        )
        
        # Fill missing values in engine data using forward fill then backward fill
        engine_columns = ['Engine RPM(rpm)', 'Engine Load(%)', 'Throttle Position(Manifold)(%)']
        for col in engine_columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Calculate engine stress
        df['engine_stress'] = (df['Engine RPM(rpm)'] * df['Engine Load(%)']) / 100.0
        
        # Define feature sets
        risk_features = [
            'speed',
            'acceleration',
            'jerk',
            'g_force_magnitude',
            'Engine RPM(rpm)',
            'Engine Load(%)',
            'Throttle Position(Manifold)(%)'
        ]
        
        eco_features = [
            'Engine RPM(rpm)',
            'Engine Load(%)',
            'Mass Air Flow Rate(g/s)',
            'Engine Coolant Temperature(°C)',
            'Throttle Position(Manifold)(%)'
        ]
        
        # Create feature matrices
        risk_data = df[risk_features].fillna(0)
        eco_data = df[eco_features].fillna(0)
        
        # Normalize features
        risk_data_scaled = self.scaler.fit_transform(risk_data)
        eco_data_scaled = self.scaler.fit_transform(eco_data)
        
        # Create sequences with proper validation
        def create_sequences(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length + 1):
                sequence = data[i:(i + sequence_length)]
                if not np.any(np.isnan(sequence)):
                    sequences.append(sequence)
            return np.array(sequences)
        
        X_risk = create_sequences(risk_data_scaled, self.sequence_length)
        X_eco = create_sequences(eco_data_scaled, self.sequence_length)
        
        if len(X_risk) == 0 or len(X_eco) == 0:
            raise ValueError("No valid sequences could be created from the data")
            
        return X_risk, X_eco, df
    
    def calculate_risk_labels(self, df):
        """Calculate risk labels with improved error handling"""
        speed_threshold = 25
        acc_threshold = 2.5
        jerk_threshold = 2.0
        g_force_threshold = 0.3
        
        risk_scores = []
        
        for i in range(len(df) - self.sequence_length + 1):
            sequence = df.iloc[i:i + self.sequence_length]
            
            # Calculate risk factors with proper validation
            speed_risk = np.clip((sequence['speed'] > speed_threshold).mean(), 0, 1)
            acc_risk = np.clip((np.abs(sequence['acceleration']) > acc_threshold).mean(), 0, 1)
            jerk_risk = np.clip((np.abs(sequence['jerk']) > jerk_threshold).mean(), 0, 1)
            g_force_risk = np.clip((sequence['g_force_magnitude'] > g_force_threshold).mean(), 0, 1)
            
            risk_score = (0.4 * speed_risk + 
                         0.3 * acc_risk + 
                         0.2 * jerk_risk + 
                         0.1 * g_force_risk)
            
            risk_scores.append(float(risk_score))
        
        return np.array(risk_scores)
    
    def calculate_eco_labels(self, df):
        """Calculate eco-driving scores with improved error handling"""
        eco_scores = []
        
        for i in range(len(df) - self.sequence_length + 1):
            sequence = df.iloc[i:i + self.sequence_length]
            
            rpm_score = np.clip(1 - (sequence['Engine RPM(rpm)'] / 4000), 0, 1).mean()
            load_score = np.clip(1 - (sequence['Engine Load(%)'] / 100), 0, 1).mean()
            throttle_score = np.clip(1 - (sequence['Throttle Position(Manifold)(%)'] / 100), 0, 1).mean()
            
            eco_score = (0.4 * rpm_score + 
                        0.4 * load_score + 
                        0.2 * throttle_score)
            
            eco_scores.append(float(eco_score))
        
        return np.array(eco_scores)
    
    def build_model(self, input_shape, output_dim=2):
        """Build LSTM model with improved architecture"""
        model = Sequential([
            BatchNormalization(input_shape=input_shape),
            LSTM(64, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(output_dim, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, epochs=50, batch_size=32):
        """Train the model with improved error handling"""
        try:
            # Preprocess data
            X_risk, X_eco, processed_df = self.preprocess_data(df)
            
            # Calculate labels
            y_risk = self.calculate_risk_labels(processed_df)
            y_eco = self.calculate_eco_labels(processed_df)
            
            # Combine scores
            y = np.column_stack([y_risk, y_eco])
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_risk, y, test_size=0.2, random_state=42
            )
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Build and train model
            self.model = self.build_model(input_shape=(X_risk.shape[1], X_risk.shape[2]))
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

def main():
    try:
        # Load data
        df = pd.read_csv('OBD.csv')
        
        # Initialize analyzer
        analyzer = DrivingAnalyzer()
        
        # Train model
        history = analyzer.train(df)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
