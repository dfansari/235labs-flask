"""Plotly Dash HTML layout override."""

html_layout = """
<!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body class="dash-template">
            <header>
              <div class="nav-wrapper">
                <a href="/">
                    <div class="row">
                        <!-- img src="/static/img/logo_235labs.png" class="logo" / -->
                        <h1>Dan Ansari</h1>
                    </div>
                  </a>
                <nav>
                  <div class="col-sm">
                      <a href="/" class="dash-link">
                        <span>About Me</span>
                        <i class="fas fa-arrow-right"></i>
                      </a>
                      <a href="/projects" class="dash-link">
                          <span>Projects</span>
                          <i class="fas fa-arrow-right"></i>
                      </a>
                  </div>
                </nav>
                </div>
            </header>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""
