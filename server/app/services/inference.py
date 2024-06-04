from app.services.model import predict_is_promotional
from app.services.preprocessing import preprocess_text_object

def run_inference(blog_data: dict):
    print("타이틀", blog_data.get("title"))
    processed_text = preprocess_text_object(blog_data)
    return predict_is_promotional(processed_text)
