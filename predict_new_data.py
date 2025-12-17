"""
INTERACTIVE DROUGHT PREDICTION SYSTEM

This script allows you to input meteorological data and get drought predictions
from all 4 trained machine learning models.

Usage: python predict_new_data.py
"""

import sys
sys.path.append('src')

from predict import (
    load_models_and_scaler,
    get_feature_names,
    get_feature_descriptions,
    predict_single_sample,
    print_predictions,
    get_example_data,
    predict_from_dict,
    validate_input
)


def print_welcome():
    """Print welcome message"""
    print("\n" + "🌵" * 40)
    print("=" * 80)
    print("           DROUGHT PREDICTION SYSTEM")
    print("       Interactive Machine Learning Prediction")
    print("=" * 80)
    print("🌵" * 40)
    
    print("\n📊 ABOUT THIS SYSTEM:")
    print("   This system uses 4 trained machine learning models to predict")
    print("   drought conditions based on meteorological data:")
    
    print("\n   🤖 Models:")
    print("      1. Logistic Regression (Newton's Method)")
    print("      2. Support Vector Machine (SVM)")
    print("      3. K-Nearest Neighbors (KNN)")
    print("      4. Gaussian Naive Bayes")
    
    print("\n   📈 Input Required:")
    print("      14 meteorological and spatial features")
    
    print("\n   🎯 Output:")
    print("      - Individual predictions from each model")
    print("      - Confidence levels (when available)")
    print("      - Majority vote consensus")
    print("      - Recommendations")
    
    print("\n" + "=" * 80)


def get_user_input_interactive():
    """Get feature values from user interactively"""
    print("\n" + "=" * 80)
    print("ENTER FEATURE VALUES")
    print("=" * 80)
    print("\nPlease provide values for the following 14 features.")
    print("Press ENTER to use example drought values.\n")
    
    feature_names = get_feature_names()
    descriptions = get_feature_descriptions()
    
    # Example drought values
    example = get_example_data()['drought_example']['data']
    
    feature_values = []
    
    for feature in feature_names:
        while True:
            try:
                prompt = f"\n{feature}\n{descriptions[feature]}\n  → "
                user_input = input(prompt)
                
                if user_input.strip() == "":
                    value = example[feature]
                    print(f"     Using example: {value}")
                else:
                    value = float(user_input)
                    
                    # Validate input
                    valid, result = validate_input(value, feature)
                    if not valid:
                        print(f"     ⚠️  Warning: {result}")
                        confirm = input("     Continue anyway? (y/n): ")
                        if confirm.lower() != 'y':
                            continue
                
                feature_values.append(value)
                break
            except ValueError:
                print("     ❌ Invalid input! Please enter a number.")
    
    return feature_values


def quick_prediction_menu():
    """Show quick prediction options"""
    print("\n" + "=" * 80)
    print("PREDICTION OPTIONS")
    print("=" * 80)
    
    print("\n1. 🖊️  Enter values manually (step-by-step)")
    print("2. 🌵 Use DROUGHT example (severe conditions)")
    print("3. 💧 Use NO DROUGHT example (normal wet conditions)")
    print("4. ⚠️  Use MODERATE RISK example (borderline conditions)")
    print("5. 📋 Show feature information")
    print("6. ❌ Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    return choice


def show_feature_information():
    """Display information about all features"""
    print("\n" + "=" * 80)
    print("FEATURE INFORMATION")
    print("=" * 80)
    
    feature_names = get_feature_names()
    descriptions = get_feature_descriptions()
    
    print("\n📊 Required Features (14 total):\n")
    
    print("🌡️  TEMPERATURE & HUMIDITY:")
    temp_features = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M']
    for feat in temp_features:
        print(f"   • {feat:.<20} {descriptions[feat]}")
    
    print("\n💨 WIND & RADIATION:")
    wind_features = ['WS2M', 'ALLSKY_SFC_SW_DWN']
    for feat in wind_features:
        print(f"   • {feat:.<20} {descriptions[feat]}")
    
    print("\n🌧️  PRECIPITATION & DROUGHT INDEX:")
    precip_features = ['PRECTOTCORR', 'spei']
    for feat in precip_features:
        print(f"   • {feat:.<20} {descriptions[feat]}")
    
    print("\n🗺️  SPATIAL & TEMPORAL (Trigonometric Encoding):")
    spatial_features = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'month_sin', 'month_cos']
    for feat in spatial_features:
        print(f"   • {feat:.<20} {descriptions[feat]}")
    
    print("\n💡 TIPS:")
    print("   • SPEI is the most important feature (Drought Index)")
    print("   • Negative SPEI indicates drought conditions")
    print("   • Positive SPEI indicates wet conditions")
    print("   • Spatial/temporal features can use example values")
    
    input("\nPress ENTER to continue...")


def display_example_details(example_name, example_data):
    """Display details of an example"""
    print("\n" + "=" * 80)
    print(f"USING EXAMPLE: {example_data['name'].upper()}")
    print("=" * 80)
    print(f"\nDescription: {example_data['description']}")
    
    print("\n📊 Feature Values:")
    for feature, value in example_data['data'].items():
        desc = get_feature_descriptions()[feature]
        print(f"   {feature:.<20} {value:>8.2f}   ({desc})")


def main():
    """Main function"""
    print_welcome()
    
    # Load models
    print("\n🔄 Loading models...")
    models, scaler = load_models_and_scaler()
    
    if models is None or scaler is None:
        print("\n❌ Cannot proceed without models!")
        print("   Please run: python main.py")
        return
    
    # Main loop
    while True:
        choice = quick_prediction_menu()
        
        if choice == '1':
            # Manual input
            print("\n" + "=" * 80)
            print("MANUAL INPUT MODE")
            print("=" * 80)
            feature_values = get_user_input_interactive()
            predictions = predict_single_sample(models, scaler, feature_values)
            print_predictions(predictions, feature_values)
            
        elif choice == '2':
            # Drought example
            examples = get_example_data()
            example = examples['drought_example']
            display_example_details('drought', example)
            
            feature_values = [example['data'][name] for name in get_feature_names()]
            predictions = predict_from_dict(models, scaler, example['data'])
            print_predictions(predictions, feature_values)
            
        elif choice == '3':
            # No drought example
            examples = get_example_data()
            example = examples['no_drought_example']
            display_example_details('no_drought', example)
            
            feature_values = [example['data'][name] for name in get_feature_names()]
            predictions = predict_from_dict(models, scaler, example['data'])
            print_predictions(predictions, feature_values)
            
        elif choice == '4':
            # Moderate example
            examples = get_example_data()
            example = examples['moderate_example']
            display_example_details('moderate', example)
            
            feature_values = [example['data'][name] for name in get_feature_names()]
            predictions = predict_from_dict(models, scaler, example['data'])
            print_predictions(predictions, feature_values)
            
        elif choice == '5':
            # Show feature information
            show_feature_information()
            continue
            
        elif choice == '6':
            print("\n👋 Thank you for using the Drought Prediction System!")
            print("Exiting...\n")
            break
            
        else:
            print("\n❌ Invalid choice! Please enter 1-6.")
            continue
        
        # Ask if user wants to make another prediction
        print("\n" + "-" * 80)
        another = input("\nMake another prediction? (y/n): ")
        if another.lower() != 'y':
            print("\n👋 Thank you for using the Drought Prediction System!")
            print("Exiting...\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        print("Exiting gracefully...\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()