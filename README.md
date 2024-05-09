# Lyrics Generation with LSTM Neural Network

This project utilizes an LSTM (Long Short-Term Memory) neural network trained on song lyrics by various artists to generate new lyrics.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:
- TensorFlow
- NumPy
- Pandas
- Streamlit

Install the dependencies using pip:
```bash
pip install tensorflow numpy pandas streamlit
```

## Files

- `1.py`: Contains the Streamlit web application code for generating lyrics.
- `model_final.h5`: Pre-trained LSTM model for lyrics generation.
- `tokenizer.pickle`: Tokenizer used for text preprocessing.
- `final_final_data.csv`: Dataset containing song lyrics used for training the model.

## Features

- **Lyrics Generation:** Generates new song lyrics based on a seed text input.
- **Customizable Output Length:** Allows users to control the length of the generated lyrics.

## Data Preprocessing

The dataset undergoes the following preprocessing steps before training the model:
1. **Text Cleaning:** Removal of non-alphabetic characters and punctuation.
2. **Tokenization:** Splitting the text into individual words or tokens.
3. **Sequence Creation:** Generating sequences of tokens with a fixed length to feed

## Model Architecture

The LSTM model architecture consists of the following layers:

1. **Embedding Layer:** Converts input sequences into dense vectors of fixed size.
2. **Bidirectional LSTM Layer:** Processes the input sequence in both forward and backward directions.
3. **Dropout Layer:** Regularizes the model to prevent overfitting.
4. **Dense (Output) Layer:** Produces the output probabilities for the next word in the sequence.

## Contributing

Contributions are welcome! Please follow the standard GitHub flow: Fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to reach out.
