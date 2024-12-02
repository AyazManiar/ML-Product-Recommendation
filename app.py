from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize Flask app
app = Flask(__name__)

# Load preprocessed data and models
with open('df2_processed.pkl', 'rb') as file:
    df2 = pickle.load(file)

with open('product_df.pkl', 'rb') as file:
    product_df = pickle.load(file)

# Content-based recommendation function
def get_cosine_similarities(train_data, item_name):
    """
    This function calculates the cosine similarity between the given item and all other items.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    return similar_items[:]

def content_based_recommendations(train_data, item_name, top_n=10):
    """
    This function takes the dataset, an item name, and a number 'top_n' to return
    the top N most similar items based on their 'Tags' using cosine similarity.
    """
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()
    
    # Get cosine similarities for all items
    similar_items_with_scores = get_cosine_similarities(train_data, item_name)
    
    # Extract the indices and similarity scores
    recommended_indices = [x[0] for x in similar_items_with_scores]
    similarity_scores = [x[1] for x in similar_items_with_scores]

    # Normalize similarity scores to a 0-1 scale for better interpretation (optional)
    min_score = min(similarity_scores)
    max_score = max(similarity_scores)
    normalized_scores = [(5 * (score - min_score) / (max_score - min_score)) for score in similarity_scores]

    # Get the recommended items
    recommended_items = train_data.iloc[recommended_indices][['ProdID', 'Name', 'Rating', 'Count_Rating', 'Brand', 'Rating Score']].copy()
    
    # Add similarity scores to the DataFrame
    recommended_items['SimilarityScore'] = normalized_scores
    recommended_items['Rating Score'] = recommended_items['Rating Score'].fillna(0)
    
    # Calculate the combined score (if you want to weight the similarity and rating)
    a = 5  # This is the weight you can adjust
    recommended_items['CombinedScore'] = (a * recommended_items['SimilarityScore']) + recommended_items['Rating Score']
    
    # Sort the items by CombinedScore
    recommended_items = recommended_items.sort_values(by='CombinedScore', ascending=False)
    
    # Return the top N recommendations
    return recommended_items.head(top_n)

# Collaborative filtering function
def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = []
    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'Brand', 'ImageURL', 'Rating']]
    return recommended_items_details.head(top_n)

# Hybrid recommendation function
def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    return hybrid_rec.head(10)

# Routes
@app.route('/')
def home():
    return render_template('index.html')  # Simple HTML form for user input

@app.route('/recommend', methods=['POST'])
def recommend():
    # Getting user input from the form
    user_id = int(request.form['user_id'])
    item_name = request.form['item_name']
    top_n = int(request.form.get('top_n', 10))

    # Get hybrid recommendations based on the user ID and item name
    recommendations = hybrid_recommendations(df2, user_id, item_name, top_n)

    # Convert the recommendations DataFrame to a list of dictionaries for rendering in the template
    recommendations_list = recommendations.to_dict(orient='records')

    # Render recommendations as a table in HTML
    return render_template('recommendations.html', recommendations=recommendations_list)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
