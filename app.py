from flask import Flask, render_template, request
from nlp_search_engine import author_recommend

app = Flask('EECSProject')

@app.errorhandler(404)
def error():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/author', methods=['POST'])
def author():
    keyword = request.form['keyword']
    results = author_recommend(keyword)
    return render_template('author.html', results=results[:100], length=len(results))

if __name__ == '__main__':
    app.run(debug=True)