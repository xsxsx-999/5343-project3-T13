from shiny import App, ui

url_a = "https://xsxshiny.shinyapps.io/ab-test-version-a/"
url_b = "https://xsxshiny.shinyapps.io/ab-test-version-b/"

app_ui = ui.page_fluid(
    ui.tags.script(f"""
        var assigned_url = localStorage.getItem('ab_test_version');
        
        if (!assigned_url) {{
            var target_urls = ['{url_a}', '{url_b}'];
            assigned_url = target_urls[Math.floor(Math.random() * target_urls.length)];
            localStorage.setItem('ab_test_version', assigned_url);
        }}
        
        window.location.replace(assigned_url);
    """),
    
    ui.h3("🚀 Redirecting you to our website... Please wait...", style="text-align: center; margin-top: 50px;")
)

def server(input, output, session):
    pass  

app = App(app_ui, server)