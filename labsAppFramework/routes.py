"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="235 Labs",
        description="Data Science from the heart of South Williamsburg",
        template="home-template"
    )
