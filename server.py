from flask import Flask, request, jsonify
from reranker import ReRanker
from rawSearch import TFIDF

app = Flask(__name__)
tf_idf = TFIDF()

@app.route('/api/search', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        query = data.get('query')
        print(query)

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Example response based on the query
        clean_query = process_query(query)

        # Search using TF-IDF
        filtered_results = tf_idf.search(clean_query, 10)

        # Re-rank using AI 
        re_ranker = ReRanker()
        scores = re_ranker.rank(query, filtered_results)

        return jsonify({'response': scores}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

def process_query(query):
    return query

if __name__ == '__main__':
    app.run(debug=True)
