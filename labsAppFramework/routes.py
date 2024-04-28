"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/projects")
def home():
    """Landing page."""
    return render_template(
        "projects.jinja2",
        title="Dan Ansari",
        description="Data Science for Finance and Beyond",
        template="home-template",
    )


@app.route("/")
def about_me():
    """Resume and Info"""
    return render_template(
        "aboutme.jinja2",
        title="About Me",
        description="Data Science for Finance and Beyond",
        template="home-template",
    )
