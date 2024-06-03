import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_dir)

from app import create_app

if __name__ == '__main__':
    import ssl

    app = create_app()

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(
        os.path.join(current_dir, 'app', 'ssl', 'cert.pem'),
        os.path.join(current_dir, 'app', 'ssl', 'key.pem')
    )
    app.run(host='0.0.0.0', port=3000, debug=True, ssl_context=context)
