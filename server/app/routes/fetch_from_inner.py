from flask import Blueprint, request, jsonify
from app.services.inference import run_inference
from app.services.ocr import get_ocr_text

fetch_from_inner_bp = Blueprint('fetch-from-inner', __name__)

@fetch_from_inner_bp.route('/fetch-from-inner', methods=['POST'])
def fetch_from_inner():
    data = request.get_json()
    title = data.get('title', '')
    content_text = data.get('contentText', '')
    image_url = data.get('imageUrl', '')

    ocr_text = get_ocr_text(image_url) if image_url else ""

    blog_data = {
        "title": title,
        "ocr_data": ocr_text,
        "content": content_text
    }


    try:
        is_promo, prob = run_inference(blog_data)
        return jsonify(predictions=[is_promo], probabilities=[[prob]])
    except Exception as e:
        return jsonify(error=str(e)), 500
