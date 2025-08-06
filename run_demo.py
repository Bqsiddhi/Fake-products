#!/usr/bin/env python3
"""
Fake Review Detection and IP Infringement Analysis - Demo Script

This script demonstrates how to run the complete analysis pipeline
with synthetic data. Perfect for testing and understanding the workflow.

Usage:
    python run_demo.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run the complete pipeline demonstration"""
    
    print("🚀 Fake Review Detection & IP Infringement Analysis Demo")
    print("=" * 60)
    
    try:
        # Import the pipeline
        from fake_review_detection import run_complete_pipeline
        
        print("📦 Pipeline imported successfully!")
        
        # Run the complete analysis
        print("\n🔄 Starting analysis pipeline...")
        print("This may take a few minutes for the complete analysis...")
        
        results = run_complete_pipeline(use_synthetic_data=True)
        
        print("\n✅ Analysis completed successfully!")
        print("\n📊 Results Summary:")
        print(f"   • Processed dataset: {len(results['dataframe'])} reviews")
        print(f"   • Fake review models trained: {len(results['fake_model_results'])}")
        print(f"   • Counterfeit models trained: {len(results['counterfeit_model_results'])}")
        
        # Show best model performance
        best_auc = 0
        best_model = None
        for model_name, model_results in results['fake_model_results'].items():
            auc = model_results.get('roc_auc', 0)
            if auc > best_auc:
                best_auc = auc
                best_model = model_name
        
        if best_model:
            print(f"   • Best model: {best_model.replace('_', ' ').title()} (ROC-AUC: {best_auc:.3f})")
        
        # Correlation insights
        correlation_results = results.get('correlation_results', {})
        if 'pearson' in correlation_results:
            pearson = correlation_results['pearson']
            if 'fake_probability' in pearson.columns and 'counterfeit_probability' in pearson.index:
                corr_coef = pearson.loc['fake_probability', 'counterfeit_probability']
                print(f"   • Fake-Counterfeit Correlation: {corr_coef:.3f}")
        
        print("\n🎉 Demo completed! Check the generated visualizations and results.")
        print("\n📋 Next Steps:")
        print("   1. Open 'fake_reviews_ip_infringement_pipeline.ipynb' for detailed analysis")
        print("   2. Replace synthetic data with your real dataset")
        print("   3. Customize detection heuristics for your domain")
        print("   4. Add external indicators (takedown data, price analysis, etc.)")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Please install required packages:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
        return None
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("\n📋 Please check:")
        print("   • All required packages are installed")
        print("   • Python version is 3.7 or higher")
        print("   • Sufficient memory available")
        return None


if __name__ == "__main__":
    # Set up the environment
    print("🔧 Setting up environment...")
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run the demo
    results = main()
    
    if results is not None:
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Working directory: {current_dir}")
    else:
        print(f"\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)