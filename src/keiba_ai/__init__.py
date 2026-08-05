"""keiba_ai: a small, self-contained horse-racing (top-3 finish) prediction pipeline.

Modules:
    parser     - pure HTML parsing (no network), easy to unit test with fixtures
    scraper    - network layer built on top of parser, with robots.txt + rate limiting
    synth_data - synthetic race-result generator for offline demo/testing
    features   - leak-free feature engineering (expanding horse/jockey stats)
    model      - LightGBM training / inference wrapper
    app        - Streamlit UI
"""
