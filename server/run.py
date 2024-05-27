from app import create_app

app = create_app()

if __name__ == '__main__':
    import ssl
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('ssl/cert.pem', 'ssl/key.pem')
    app.run(host='0.0.0.0', port=3000, ssl_context=context)