
import numpy as np
from scipy.cluster.hierarchy import dendrogram , linkage

from sklearn.decomposition import PCA
from sklearn import cluster
from collections import defaultdict
from queue import Queue, LifoQueue
from multiprocessing.pool import ThreadPool
import json
import requests
import functools

from flask import Flask
from flask import request
from flask_cors import CORS

import openai

import os

api_key = os.env['OPENAI_KEY']
openai.api_key = api_key

app = Flask(__name__)
CORS(app)

def get_references(paper_id):
    ref_url_format = 'https://api.semanticscholar.org/graph/v1/paper/{}/references?limit=200'
    ref_url = ref_url_format.format(paper_id)
    data = json.loads(requests.get(ref_url).text)
    return data

def get_details(paper_id):
    get_details_format = 'https://api.semanticscholar.org/graph/v1/paper/{}?fields=url,year,authors,venue,embedding,title'
    url = get_details_format.format(paper_id)
    details = json.loads(requests.get(url).text)
    return details

def get_cse599_data():
    with open('cse599d-paper-data.json', 'r') as f:
        data = json.load(f)
    data = [d['s2data'] for d in data if d.get('s2data', None) is not None]
    return data

# print('fetching paper citations')
# paper_id = 'URL:https://www.semanticscholar.org/paper/Extracting-and-Retargeting-Color-Mappings-from-of-Poco-Mayhua/be492fa6af329a487e2048df6c931e7f9d41d451'
# paper_id = '6f186df47e24ab883a9feb02008664960d7a2ed6'

@functools.lru_cache(maxsize=4096)
def fetch_paper_data(paper_id):
    refs = get_references(paper_id)['data']

    # paper_id = data['data'][0]['citedPaper']['paperId']
    paper_ids = [r['citedPaper']['paperId'] for r in refs]
    pool = ThreadPool(processes=20)
    data = pool.map(get_details, paper_ids)
    data = [r for r in data if 'embedding' in r]
    return data



@functools.lru_cache(maxsize=4096)
def get_category(titles):
    if len(titles) == 1:
        return titles[0]
    prompt = "I am writing a review paper including the following papers:"
    prompt = '\n'.join([x for x in titles])
    # prompt += """\nRight now these papers are filed under the "Scholarly communication" folder.
    # However, having read them, I would file them under the better folder of "
    # """
    # prompt += '''\n\nA short but extremely descriptive category for these papers would be "'''
    prompt += '''\n\nA review paper summarizing and synthesizing all of the above papers would have the descriptive title of "'''
    # prompt += '''\n\n\nA review paper synthesizing and summarizing the above papers would be titled "'''
    complete = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt,
      max_tokens=30,
      temperature=0.0
    )
    text = complete['choices'][0]['text']
    text = text.split('"')[0].capitalize()
    text = text.replace(".", "")
    return text


class Group():
    def __init__(self, items=None, terminal=False, level=0):
        if items is None:
            items = list()
        self.items = items
        count = 0
        for item in items:
            if isinstance(item, Group):
                count += item.count
            else:
                count += 1
        self.count = count
        self.level = level
        self.terminal = terminal
        self.title = None

    def combine(self, other_group):
        return Group([self, other_group], level=max(self.level, other_group.level)+1)

    def add_item(self, item):
        if isinstance(item, Group):
            self.count += item.count
        else:
            self.count += 1
        self.items.append(item)

    def recount(self):
        count = 0
        for item in self.items:
            if isinstance(item, Group):
                count += item.recount()
            else:
                count += 1
        self.count = count
        return count
        
    def get_all_items(self):
        all_items = []
        q = LifoQueue()
        q.put(self)
        while q.qsize() > 0:
            obj = q.get()
            for item in obj.items:
                if isinstance(item, Group):
                    q.put(item)
                else:
                    all_items.append(item)
        return all_items

    def get_all_groups(self):
        all_items = []
        q = LifoQueue()
        q.put(self)
        while q.qsize() > 0:
            obj = q.get()
            for item in obj.items:
                if isinstance(item, Group):
                    q.put(item)
                    all_items.append(item)
        return all_items

    def get_items(self):
        return [item for item in self.items
                if not isinstance(item, Group)]

    def get_groups(self):
        return [item for item in self.items
                if  isinstance(item, Group)]

    def get_titles(self):
        titles = []
        for item in self.items:
            if isinstance(item, Group):
                titles.append(item.title)
            else:
                titles.append(format_paper(item))
        return titles

    def has_subtitles(self):
        for item in self.items:
            if isinstance(item, Group) and not item.has_title():
                return False
        return True

    def has_title(self):
        return self.title is not None

    def flatten(self):
        self.items = self.get_all_items()
        self.terminal = True

    def fetch_title(self):
        if self.has_title():
            return self.title
        elif not self.has_subtitles():
            raise ValueError("subtitles not present")
        else:
            self.title = get_category(tuple(self.get_titles()))
            return self.title

    def break_to_size(self, size=6, min_items=6):
        while len(self.items) < size:
            ix = np.argmax([x.count for x in self.items])
            obj = self.items[ix]
            if obj.count <= min_items:
                break
            self.items.remove(obj)
            self.items.extend(obj.items)


def format_authors(authors):
    if len(authors) == 1:
        return authors[0]['name']
    elif len(authors) == 2:
        return "{} and {}".format(authors[0]['name'],
                                  authors[1]['name'])
    else:
        return "{} et al".format(authors[0]['name'])

def format_paper(row):
    return '<a href="{url}">{author_text}. ({year}) {title}. <i>{venue}</i></a>'.format(
        author_text=format_authors(row['authors']), **row)

def render_list(rows):
    if len(rows) == 0:
        return ''
    else:
        return '<ul>' + '\n'.join([
            '<li>' + format_paper(r) + '</li>'
            for r in rows
        ]) + '</ul>'

def render_hierarchy(root):
    items = []
    for g in root.get_groups():
        html = render_hierarchy(g)
        item = "<details>\n<summary>{}</summary>\n{}</details>".format(
            g.title, html)
        items.append(item)
    items.append(render_list(root.get_items()))

    return '\n'.join(items)

def process_data(data):
    #Linkage Matrix
    vecs = np.array([r['embedding']['vector'] for r in data])
    # pcs = PCA(n_components=5).fit_transform(vecs)
    Z = linkage(vecs, method = 'weighted',
                metric='cosine',
                optimal_ordering=True)

    nodes = [Group([r], terminal=True) for r in data]
    n = len(data)
    for i in range(len(Z)):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        new_group = nodes[a].combine(nodes[b])
        nodes.append(new_group)

    root = nodes[-1]

    q = Queue()
    q.put(root)
    while q.qsize() > 0:
        obj = q.get()
        if obj.count <= 8:
            obj.flatten()
        else:
            obj.break_to_size(size=6, min_items=8)
            for item in obj.items:
                if isinstance(item, Group):
                    q.put(item)


    # root = build_hierarchy(data)

    n_max = sum([x.has_subtitles() for x in root.get_all_groups()])
    pool = ThreadPool(processes=n_max)

    while True:
        possible = [x for x in root.get_all_groups()
                    if x.has_subtitles() and not x.has_title()]
        print(len(possible))
        if len(possible) == 0: break
        pool.map(Group.fetch_title, possible)

    html = render_hierarchy(root)
    return html

@app.route('/cse599d')
def render_cse599():
    data = get_cse599_data()
    html = process_data(data)
    pre = '<div>Papers from CSE 599D: The Future of Scholarly Communication</div>'
    return pre + html

@app.route('/getpaper')
def render_paper():
    url = request.args['url']
    paper_id = 'URL:' + url
    data = fetch_paper_data(paper_id)
    html = process_data(data)
    paper_details = get_details(paper_id)
    pre = '<div>Grouping references of paper {}</div>'.format(format_paper(paper_details))
    return pre + html

@app.route('/')
def hello():
    return "Hello from paper GPT3 interface"

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=7500)
