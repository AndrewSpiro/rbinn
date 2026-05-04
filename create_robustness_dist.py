import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obtain robustness distribution')

    parser.add_argument('models', choices = ['pixelreg', 'khmodel'], help="Models on which to obtain robustness distributions.")
    args = parser.parse_args()
    print("Creating robustness distribution...")