# Arabic Part-of-Speech (POS) Tagging

This project implements an Arabic Part-of-Speech (POS) tagging model using a sequence-to-sequence architecture with attention mechanism. The model is built using TensorFlow and Keras.

## Actions and Configurations

### 1. Data Preparation
- The data was downloaded, and the maximum length of the segments was determined.
- The segments (X) and their corresponding POS tags (y) were stored in separate arrays.

### 2. Tokenization
- **Vocabulary and POS Tags:** A vocabulary of unique words and a set of unique POS tags were created from the data.
- **Index Mapping:** Each word and POS tag was assigned a unique index. This mapping was used to convert the words in each document and their corresponding POS tags into indices. An 'UNK' token was added for unknown words.

### 3. Model Architecture
- A sequence-to-sequence model with attention was built using TensorFlow and Keras.
- The model consists of an embedding layer, an LSTM encoder, an LSTM decoder, and an attention mechanism.
- The final layer is a time-distributed dense layer with softmax activation.

### Model Compilation
- The model was compiled with the Adam optimizer and the sparse categorical cross-entropy loss function.

### Model Training
- The model was trained for 80 epochs with a batch size of 64.
- 15% of the data was used for validation during training.

## Results and Comparison
- **Tokenizer:** Tokenizing the data manually gave better results than using Keras tokenizer due to the size of the data.
- **Hyperparameters:** The number of units in the LSTM layers and the dimension of the embeddings were tuned by trial and error. More units for LSTM and higher dimensions for embeddings led to overfitting, while fewer units and dimensions led to underfitting.
- **Accuracy:** The model achieved an accuracy of 90% on the training set and 81% on the testing set. Adding more layers led to overfitting, again due to the small size of the data.

## Inference

To perform inference with the trained model, follow these steps:

1. **Load the Model:**
    ```python
    from tensorflow.keras.models import load_model

    model = load_model('path_to_trained_model')
    ```

2. **Preprocess Input Text:**
    ```python
    def preprocess(input_text):
        # Tokenize and convert input text to indices
        # Add your preprocessing steps here
        pass
    ```

3. **Perform Inference:**
    ```python
    def infer(model, input_text):
        """This function performs inference using the trained model."""
        # Preprocess the input text
        preprocessed_text = preprocess(input_text)
        # Perform the translation
        translation = model.translate(preprocessed_text)
        # Postprocess the translation
        result = postprocess(translation)
        return result

    # Example usage of the inference function
    input_text = "Hello, how are you?"
    print(infer(model, input_text))
    ```

4. **Postprocess the Output:**
    ```python
    def postprocess(output):
        # Convert model output indices back to words
        # Add your postprocessing steps here
        pass
    ```

## Requirements

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
