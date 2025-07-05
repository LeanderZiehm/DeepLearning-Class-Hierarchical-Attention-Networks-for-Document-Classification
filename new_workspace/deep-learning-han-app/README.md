# Deep Learning Hierarchical Attention Network (HAN) for Fake News Detection

This project implements a Hierarchical Attention Network (HAN) for the classification of fake news articles. The model is designed to process documents hierarchically, capturing both word-level and sentence-level features through attention mechanisms.

## Project Structure

```
deep-learning-han-app
├── src
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── han.py          # Implementation of the Hierarchical Attention Network model
│   │   └── base.py        # Base class for models
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── fake_news.py    # Implementation of the Fake News dataset class
│   │   └── base.py        # Base class for datasets
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── text.py        # Text preprocessing utilities
│   ├── training
│   │   ├── __init__.py
│   │   └── trainer.py     # Training logic for the models
│   ├── pipeline.py        # Main entry point for the application
│   └── utils.py           # Utility functions
├── data                   # Directory for data files
├── files                  # Directory for model checkpoints and vocabularies
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd deep-learning-han-app
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your dataset files in the `data/` directory. The project is designed to work with fake news datasets.

2. **Preprocessing**: The preprocessing scripts will clean and prepare the text data for training.

3. **Training the Model**: Use the `pipeline.py` script to train the Hierarchical Attention Network on your dataset. You can customize the training parameters in the script.

4. **Evaluation**: After training, the model can be evaluated on a test set to measure its performance.

## Example

To run the training pipeline, execute the following command:

```bash
python src/pipeline.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.