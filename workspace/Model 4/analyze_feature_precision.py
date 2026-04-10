import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import linregress
import json

# Load preprocessed data
train_df = pd.read_csv("../train_data.csv")
val_df = pd.read_csv("../val_data.csv")

# Add lag features
train_df['temp_lag1'] = train_df['temperature'].shift(1)
train_df['humidity_lag1'] = train_df['relative_humidity'].shift(1)
val_df['temp_lag1'] = val_df['temperature'].shift(1)
val_df['humidity_lag1'] = val_df['relative_humidity'].shift(1)

# Compute temperature_delta only
def rolling_slope(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: linregress(range(len(x)), x).slope if not np.isnan(x).any() else np.nan,
        raw=True
    )

train_df['temperature_delta'] = rolling_slope(train_df['temperature'], window=15)
val_df['temperature_delta'] = rolling_slope(val_df['temperature'], window=15)

# Drop rows with NaNs
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Define base feature and target columns
base_features = [
    'illuminance', 'solar_radiation', 'uv', 'relative_humidity',
    'station_pressure', 'wind_avg', 'wind_gust', 'day_of_year', 'time_of_day',
    'temperature_delta', 'temp_lag1', 'humidity_lag1'
]

# Domain bounds for select features
domain_bounds = {
    "wind_gust": (0, None),
    "wind_avg": (0, None),
    "day_of_year": (1, 366),
    "time_of_day": (0, 24),
    "uv": (0, None),
    "relative_humidity": (0, 100),
    "humidity_lag1": (0, 100),
    "illuminance": (0, None),
    "solar_radiation": (0, None)
}

def analyze_feature_precision():
    """Analyze each feature to determine optimal precision (INT8, INT16, or INT32)"""
    
    print("=" * 80)
    print("FEATURE PRECISION ANALYSIS")
    print("=" * 80)
    print(f"{'Feature':<20} {'Min':<12} {'Max':<12} {'Range':<12} {'Optimal':<8} {'Reasoning'}")
    print("-" * 80)
    
    results = {}
    
    for feature in base_features:
        # Get raw values
        raw_values = train_df[feature].values
        
        # Apply domain bounds if specified
        floor, ceiling = domain_bounds.get(feature, (None, None))
        if floor is not None:
            raw_values = np.maximum(raw_values, floor)
        if ceiling is not None:
            raw_values = np.minimum(raw_values, ceiling)
        
        # Calculate range
        min_val = raw_values.min()
        max_val = raw_values.max()
        range_val = max_val - min_val
        
        # Determine optimal precision
        if range_val <= 255:  # Fits in 8 bits
            optimal = "INT8"
            reasoning = f"Range ≤ 255 ({range_val:.2f})"
        elif range_val <= 65535:  # Fits in 16 bits
            optimal = "INT16"
            reasoning = f"Range ≤ 65535 ({range_val:.2f})"
        else:  # Needs 32 bits
            optimal = "INT32"
            reasoning = f"Range > 65535 ({range_val:.2f})"
        
        # Store results
        results[feature] = {
            'min': float(min_val),
            'max': float(max_val),
            'range': float(range_val),
            'optimal_precision': optimal,
            'reasoning': reasoning
        }
        
        print(f"{feature:<20} {min_val:<12.2f} {max_val:<12.2f} {range_val:<12.2f} {optimal:<8} {reasoning}")
    
    return results

def analyze_normalized_precision():
    """Analyze precision requirements after normalization to [0,1]"""
    
    print("\n" + "=" * 80)
    print("NORMALIZED FEATURE PRECISION ANALYSIS")
    print("=" * 80)
    print("After normalization to [0,1], all features need the same precision")
    print("But we can analyze the effective precision based on actual data distribution")
    print()
    
    # Process features with same normalization as training
    X_train_base = train_df[base_features].copy()
    input_scaler = {}
    
    for feature in base_features:
        f_min = train_df[feature].min()
        f_max = train_df[feature].max()
        range_pad = 0.05 * (f_max - f_min)

        floor, ceiling = domain_bounds.get(feature, (None, None))
        f_min_adj = max(f_min - range_pad, floor) if floor is not None else f_min - range_pad
        f_max_adj = min(f_max + range_pad, ceiling) if ceiling is not None else f_max + range_pad

        input_scaler[feature] = {"min": f_min_adj, "max": f_max_adj}
        X_train_base[feature] = (X_train_base[feature] - f_min_adj) / (f_max_adj - f_min_adj)
    
    X_train = X_train_base.values
    
    print(f"{'Feature':<20} {'Unique Values':<15} {'Effective Bits':<15} {'Recommendation'}")
    print("-" * 80)
    
    normalized_results = {}
    
    for i, feature in enumerate(base_features):
        values = X_train[:, i]
        unique_values = len(np.unique(values))
        
        # Calculate effective bits needed
        if unique_values <= 256:
            effective_bits = 8
            recommendation = "INT8 sufficient"
        elif unique_values <= 65536:
            effective_bits = 16
            recommendation = "INT16 sufficient"
        else:
            effective_bits = 32
            recommendation = "INT32 needed"
        
        # More precise calculation
        actual_bits = int(np.ceil(np.log2(unique_values)))
        
        normalized_results[feature] = {
            'unique_values': int(unique_values),
            'effective_bits': effective_bits,
            'actual_bits_needed': actual_bits,
            'recommendation': recommendation
        }
        
        print(f"{feature:<20} {unique_values:<15} {actual_bits:<15} {recommendation}")
    
    return normalized_results

def analyze_hybrid_precision():
    """Suggest a hybrid approach with different precisions for different features"""
    
    print("\n" + "=" * 80)
    print("HYBRID PRECISION RECOMMENDATION")
    print("=" * 80)
    print("Based on analysis, here's a suggested precision allocation:")
    print()
    
    # Get normalized analysis
    normalized_results = analyze_normalized_precision()
    
    int8_features = []
    int16_features = []
    int32_features = []
    
    for feature, data in normalized_results.items():
        if data['actual_bits_needed'] <= 8:
            int8_features.append(feature)
        elif data['actual_bits_needed'] <= 16:
            int16_features.append(feature)
        else:
            int32_features.append(feature)
    
    print(f"INT8 Features ({len(int8_features)}): {', '.join(int8_features)}")
    print(f"INT16 Features ({len(int16_features)}): {', '.join(int16_features)}")
    print(f"INT32 Features ({len(int32_features)}): {', '.join(int32_features)}")
    
    # Calculate total features for each approach
    total_int8 = len(int8_features) * 1  # 1 feature per original
    total_int16 = len(int16_features) * 2  # 2 features (MSB/LSB) per original
    total_int32 = len(int32_features) * 4  # 4 features (4 bytes) per original
    
    total_features = total_int8 + total_int16 + total_int32
    
    print(f"\nTotal input features for hybrid approach: {total_features}")
    print(f"  - INT8: {total_int8} features")
    print(f"  - INT16: {total_int16} features") 
    print(f"  - INT32: {total_int32} features")
    
    return {
        'int8_features': int8_features,
        'int16_features': int16_features,
        'int32_features': int32_features,
        'total_features': total_features
    }

def main():
    print("Analyzing feature precision requirements...")
    print("This will help determine which features need INT8, INT16, or INT32 precision")
    print()
    
    # Analyze raw feature ranges
    raw_results = analyze_feature_precision()
    
    # Analyze normalized precision
    normalized_results = analyze_normalized_precision()
    
    # Suggest hybrid approach
    hybrid_results = analyze_hybrid_precision()
    
    # Save results
    analysis_results = {
        'raw_analysis': raw_results,
        'normalized_analysis': normalized_results,
        'hybrid_recommendation': hybrid_results
    }
    
    with open("feature_precision_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Analysis complete!")
    print("✅ Results saved to: feature_precision_analysis.json")
    print()
    print("Key insights:")
    print(f"- {len(hybrid_results['int8_features'])} features can use INT8 precision")
    print(f"- {len(hybrid_results['int16_features'])} features can use INT16 precision")
    print(f"- {len(hybrid_results['int32_features'])} features need INT32 precision")
    print(f"- Hybrid approach would use {hybrid_results['total_features']} total input features")
    print()
    print("This could significantly reduce model complexity while maintaining precision!")

if __name__ == "__main__":
    main()
