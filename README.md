
# Movie Recommendation System

This project implements a collaborative filtering-based movie recommendation system using a Restricted Boltzmann Machine (RBM) approach. The system predicts and recommends movies for users based on their historical ratings.

## Project Structure

- `test-movie-recommender-utilities.ipynb`: Jupyter notebook for loading data, training the model, and generating recommendations for a user.
- `utils.py`: Python script containing utility functions for data processing, model training, and recommendation generation.

## Installation

Before running the code, ensure you have the following dependencies installed:

```bash
pip install pandas tensorflow numpy
```

## Usage

### Jupyter Notebook

The Jupyter notebook (`test-movie-recommender-utilities.ipynb`) provides a step-by-step walkthrough of the following tasks:

1. **Data Preparation**:
   - Load and preprocess movie ratings data.
   - Normalize the ratings to a range between 0 and 1.

2. **Model Weights and Biases**:
   - Load pre-trained weights and biases for the RBM model.

3. **Recommendation Generation**:
   - Generate and display movie recommendations for a specified user.

4. **Result Visualization**:
   - Display the top recommended movies and their respective scores for the user.

### Python Script

The `utils.py` script contains the following utility functions:

- **`get_data()`**: Loads and preprocesses the movie ratings data from a CSV file.
- **`normalize_data(df)`**: Normalizes the ratings data between 0 and 1.
- **`pivot_data()`**: Pivots the data to create a user-item matrix.
- **`get_normalized_data()`**: Combines pivoting and normalization to return the normalized user-item matrix.
- **`weights()`**: Loads the model's pre-trained weights from a CSV file and converts them into a TensorFlow tensor.
- **`hidden_bias()`**: Loads and converts the hidden layer biases into a TensorFlow tensor.
- **`visible_bias()`**: Loads and converts the visible layer biases into a TensorFlow tensor.
- **`user_tensor(user_ratings)`**: Converts user ratings into a TensorFlow tensor.
- **`hidden_layer(v0, W, hb)`**: Computes the hidden layer states.
- **`reconstructed_output(h0, W, vb)`**: Reconstructs the visible layer from hidden layer states.
- **`generate_recommendation(user_ratings, W, vb, hb)`**: Generates movie recommendations for a specific user based on their historical ratings.

## Running the Project

1. **Jupyter Notebook**:
   - Open the `test-movie-recommender-utilities.ipynb` notebook.
   - Execute the cells step-by-step to preprocess data, load the model, and generate recommendations for a user.

2. **Python Script**:
   - Use the utility functions from `utils.py` to integrate with other projects or run specific parts of the recommendation process.

## Example

To generate recommendations for a user (e.g., user ID 1024), you can run the following code snippet in a Jupyter notebook:

```python
import pandas as pd
import utils

# Load the normalized ratings and model parameters
normalized_ratings = utils.get_normalized_data()
W = utils.weights()
vb = utils.visible_bias()
hb = utils.hidden_bias()

# Generate recommendations
user_id = 1024
user_ratings = normalized_ratings.loc[user_id]
recommendations = utils.generate_recommendation(user_ratings, W, vb, hb)

# Display top recommendations
recommendation_df = pd.DataFrame({"movie_id": normalized_ratings.columns, "user_id": user_id, "RecommendationScore": recommendations[0].numpy()})
print(recommendation_df.sort_values("RecommendationScore", ascending=False).head(10))
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

