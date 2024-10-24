# traderes-vres

## Project Description
`traderes-vres` is a Python project integrated in the European project [TradeRES](https://traderes.eu).
It serves as a tool to easily create renewable energy power forecasts using the methodology developed within TradeRES. The project includes various modules for model specifications, error metrics evaluation, feature engineering, and other steps in the forecasting pipeline as defined by the TradeRES methodology.

Funding: This work has received funding from the EU Horizon 2020 research and innovation program under project TradeRES (grant agreement No 864276).

## DATA
## Installation
To install the required dependencies, run the following command:

```sh
pip install -r requirements.txt
```

## Usage

### Activate Virtual Environment
```sh
.\venv\Scripts\activate
```

### Run the Main Script
```sh
python train.py
```

## Modules
- **libs/error_metrics.py**: Contains functions for evaluating error metrics.
- **libs/models_specs.py**: Defines model specifications and configurations.
- **libs/models_catalog.py**: Includes functions for training models, such as LightGBM regressor.

## Updating Dependencies
To update `numpy` and `scikit-learn` to the latest versions, run:
```sh
pip install --upgrade numpy scikit-learn
```

## License


``` the latest versions, run: