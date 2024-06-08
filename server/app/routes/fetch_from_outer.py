from flask import Blueprint, request, jsonify
from app.services.crawler import crawl_blog_content
from app.services.preprocessing import preprocess_text_object
from app.services.inference import run_inference
import asyncio

fetch_from_outer_bp = Blueprint('fetch-from-outer', __name__)

@fetch_from_outer_bp.route('/fetch-from-outer', methods=['POST'])
def fetch_from_outer():
    data = request.get_json()
    blog_url = data.get('blogUrl')

    try:
        blog_data = asyncio.run(crawl_blog_content(blog_url))
        is_promo, prob = run_inference(blog_data)
        return jsonify(predictions=[is_promo], probabilities=[[prob]])
    except Exception as e:
        return jsonify(message='Crawling failed', error=str(e)), 500
