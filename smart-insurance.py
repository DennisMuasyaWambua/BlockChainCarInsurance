# AI Model for Driver Risk Assessment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class DriverRiskModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_data(self, data):
        # Extract relevant features for risk assessment
        risk_features = [
            'GPS Speed (Meters/second)',
            'G(x)', ' G(y)', 'G(z)', 'G(calibrated)',
            'Engine RPM(rpm)',
            'Engine Load(%)',
            'Throttle Position(Manifold)(%)',
            'Engine Coolant Temperature(°C)',
            'Mass Air Flow Rate(g/s)',
            'Altitude',
            'Bearing'
        ]
        
        # Calculate derived features
        data['harsh_acceleration'] = data['G(y)'].apply(lambda x: 1 if x > 0.2 else 0)
        data['harsh_braking'] = data['G(y)'].apply(lambda x: 1 if x < -0.2 else 0)
        data['sharp_turns'] = data['G(x)'].apply(lambda x: 1 if abs(x) > 0.2 else 0)
        
        return data[risk_features]

    def calculate_risk_score(self, obd_data):
        processed_data = self.preprocess_data(obd_data)
        scaled_data = self.scaler.transform(processed_data)
        risk_score = self.model.predict(scaled_data)
        return risk_score

class EmissionsAnalyzer:
    def __init__(self):
        self.baseline_emissions = None
        
    def calculate_emissions(self, obd_data):
        # Calculate emissions based on MAF sensor data
        # CO2 (g/km) ≈ 2.35 * MAF * (1/speed)
        emissions = []
        for _, row in obd_data.iterrows():
            maf = row['Mass Air Flow Rate(g/s)']
            speed = row['GPS Speed (Meters/second)']
            if speed > 0:
                co2 = 2.35 * maf * (1/speed) * 3.6  # Convert to g/km
                emissions.append(co2)
        return np.mean(emissions)
    
    def calculate_eco_score(self, obd_data):
        # Factors affecting eco score
        avg_speed = obd_data['GPS Speed (Meters/second)'].mean()
        avg_rpm = obd_data['Engine RPM(rpm)'].mean()
        avg_load = obd_data['Engine Load(%)'].mean()
        
        # Penalties for inefficient driving
        high_rpm_penalty = len(obd_data[obd_data['Engine RPM(rpm)'] > 3000]) / len(obd_data)
        high_load_penalty = len(obd_data[obd_data['Engine Load(%)'] > 80]) / len(obd_data)
        
        eco_score = 100 - (high_rpm_penalty * 20 + high_load_penalty * 20)
        return max(0, min(100, eco_score))

# Solidity Smart Contract for Insurance Policy
"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SmartAutoInsurance {
    struct Policy {
        address policyholder;
        uint256 startDate;
        uint256 endDate;
        uint256 baseRate;
        uint256 riskScore;
        uint256 ecoScore;
        uint256 totalPremium;
        bool isActive;
    }
    
    struct DrivingData {
        uint256 timestamp;
        uint256 riskScore;
        uint256 ecoScore;
        string dataHash;  // IPFS hash of raw OBD data
    }
    
    mapping(address => Policy) public policies;
    mapping(address => DrivingData[]) public drivingHistory;
    
    event PolicyCreated(address indexed policyholder, uint256 premium);
    event DrivingDataUpdated(address indexed policyholder, uint256 riskScore, uint256 ecoScore);
    event PremiumAdjusted(address indexed policyholder, uint256 newPremium);
    
    function createPolicy(uint256 _baseRate) public {
        require(policies[msg.sender].isActive == false, "Policy already exists");
        
        Policy memory newPolicy = Policy({
            policyholder: msg.sender,
            startDate: block.timestamp,
            endDate: block.timestamp + 365 days,
            baseRate: _baseRate,
            riskScore: 50,  // Initial neutral score
            ecoScore: 50,   // Initial neutral score
            totalPremium: _baseRate,
            isActive: true
        });
        
        policies[msg.sender] = newPolicy;
        emit PolicyCreated(msg.sender, _baseRate);
    }
    
    function updateDrivingData(uint256 _riskScore, uint256 _ecoScore, string memory _dataHash) public {
        require(policies[msg.sender].isActive, "No active policy");
        
        DrivingData memory newData = DrivingData({
            timestamp: block.timestamp,
            riskScore: _riskScore,
            ecoScore: _ecoScore,
            dataHash: _dataHash
        });
        
        drivingHistory[msg.sender].push(newData);
        
        // Update policy scores
        policies[msg.sender].riskScore = _riskScore;
        policies[msg.sender].ecoScore = _ecoScore;
        
        // Adjust premium based on scores
        uint256 riskMultiplier = (200 - _riskScore) / 100;  // Lower risk = lower multiplier
        uint256 ecoDiscount = _ecoScore / 100;  // Higher eco score = higher discount
        
        uint256 newPremium = (policies[msg.sender].baseRate * riskMultiplier * (100 - ecoDiscount)) / 100;
        policies[msg.sender].totalPremium = newPremium;
        
        emit DrivingDataUpdated(msg.sender, _riskScore, _ecoScore);
        emit PremiumAdjusted(msg.sender, newPremium);
    }
    
    function getPolicy() public view returns (
        uint256 startDate,
        uint256 endDate,
        uint256 baseRate,
        uint256 riskScore,
        uint256 ecoScore,
        uint256 totalPremium
    ) {
        Policy memory policy = policies[msg.sender];
        return (
            policy.startDate,
            policy.endDate,
            policy.baseRate,
            policy.riskScore,
            policy.ecoScore,
            policy.totalPremium
        );
    }
}
"""

# Example usage
def main():
    # Load and preprocess OBD data
    obd_data = pd.read_csv('OBD.csv')
    
    # Initialize models
    risk_model = DriverRiskModel()
    emissions_analyzer = EmissionsAnalyzer()
    
    # Calculate scores
    risk_score = risk_model.calculate_risk_score(obd_data)
    eco_score = emissions_analyzer.calculate_eco_score(obd_data)
    
    # Store data hash on IPFS (pseudo-code)
    # ipfs_hash = store_on_ipfs(obd_data)
    
    # Update blockchain contract (pseudo-code)
    # contract.updateDrivingData(risk_score, eco_score, ipfs_hash)
    
    return risk_score, eco_score

if __name__ == "__main__":
    main()
