
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
from flask_compress import Compress

import openai

import os
import io

import altair as alt
import pandas as pd
import hashlib


openai.api_key = os.environ['OPENAI_KEY']
s2_api_key = os.environ['S2KEY']

app = Flask(__name__)
CORS(app)
Compress(app)

@functools.lru_cache(maxsize=4096)
def get_references(paper_id):
    ref_url_format = 'https://api.semanticscholar.org/graph/v1/paper/{}/references?limit=200'
    ref_url = ref_url_format.format(paper_id)
    res = requests.get(ref_url, headers={'x-api-key': s2_api_key})
    data = json.loads(res.text)
    return data


@functools.lru_cache(maxsize=4096)
def get_details(paper_id):
    get_details_format = 'https://api.semanticscholar.org/graph/v1/paper/{}?fields=url,year,authors,citationCount,venue,embedding,title'
    url = get_details_format.format(paper_id)
    res = requests.get(url, headers={'x-api-key': s2_api_key})
    details = json.loads(res.text)
    return details

def get_cse599_data():
    with open('data/paper_data.json', 'r') as f:
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
    pool = ThreadPool(processes=50)
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
    prompt += '''\n\nA review paper summarizing and synthesizing all of the above papers would have the short but descriptive title of "'''
    # prompt += '''\n\n\nA review paper synthesizing and summarizing the above papers would be titled "'''
    complete = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      max_tokens=30,
      temperature=0.0
    )
    text = complete['choices'][0]['text']
    text = text.split('"')[0].capitalize()
    text = text.replace(".", "")
    return text


GROUP_CACHE = dict()

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

    def compute_hash(self):
        items = self.get_all_items()
        paper_ids = [item['paperId'] for item in items]
        h = hashlib.sha256()
        for p in paper_ids:
            h.update(p.encode('utf8'))
        self.hash_value = h.hexdigest()[:16]
        return self.hash_value

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
            ix = np.argmax([x.count if isinstance(x, Group) else 1
                            for x in self.items])
            obj = self.items[ix]
            if obj.count <= min_items:
                break
            self.items.remove(obj)
            self.items.extend(obj.items)

    def add_to_cache(self):
        h = self.compute_hash()
        GROUP_CACHE[h] = self
        for item in self.items:
            if isinstance(item, Group):
                item.add_to_cache()


def format_authors(authors):
    if len(authors) == 1:
        return authors[0]['name']
    elif len(authors) == 2:
        return "{} and {}".format(authors[0]['name'],
                                  authors[1]['name'])
    else:
        return "{} et al".format(authors[0]['name'])

def format_paper(row):
    return '<a href="{url}">{author_text}. ({year_int}) {title}. <i>{venue}</i></a>'.format(
        author_text=format_authors(row['authors']), year_int=int(row['year']), **row)

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
        item = """<details>
        <summary>
        <input type='checkbox' name='ids' value='{}'>
        {}
        </summary>
        {}
        </details>""".format(
            g.hash_value, g.title, html)
        items.append(item)
    items.append(render_list(root.get_items()))

    return '\n'.join(items)

def process_data(data):
    #Linkage Matrix
    vecs = np.array([r['embedding']['vector'] for r in data])
    # pcs = PCA(n_components=5).fit_transform(vecs)
    Z = linkage(vecs, method = 'ward', metric='euclidean',
                optimal_ordering=True)

    nodes = [Group([r], terminal=True) for r in data]
    n = len(data)
    for i in range(len(Z)):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        new_group = nodes[a].combine(nodes[b])
        nodes.append(new_group)

    root = nodes[-1]
    root.recount()

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

    root.recount()

    n_max = sum([x.has_subtitles() for x in root.get_all_groups()])
    pool = ThreadPool(processes=n_max)

    while True:
        possible = [x for x in root.get_all_groups()
                    if x.has_subtitles() and not x.has_title()]
        print(len(possible))
        if len(possible) == 0: break
        pool.map(Group.fetch_title, possible)

    root.add_to_cache()

    return root

def wrap_form(html):
    return '<form id="checked-groups">' + html + '</form>'

@app.route('/cse599d')
def render_cse599():
    data = get_cse599_data()
    root = process_data(data)
    html = render_hierarchy(root)
    html = wrap_form(html)
    pre = '<div>Papers from CSE 599D: The Future of Scholarly Communication</div>'
    return pre + html

@app.route('/getpaper')
def render_paper():
    url = request.args['url']
    paper_id = 'URL:' + url
    data = fetch_paper_data(paper_id)
    root = process_data(data)
    html = render_hierarchy(root)
    html = wrap_form(html)
    paper_details = get_details(paper_id)
    pre = '<div>Grouping references of paper {}</div>'.format(format_paper(paper_details))
    return pre + html

# @app.route("/refgraph.json")
# def influential_refs():
#     chart = get_ref_graph()
#     out = io.StringIO()
#     chart.save(out, format='json')
#     return out.getvalue()

def get_top_papers_html(textdf, topn=10):
    subdf = textdf.iloc[:topn]
    rows = [subdf.iloc[i].to_dict() for i in range(len(subdf))]
    return render_list(rows)


@app.route("/postgraph", methods=["POST"])
def influential_refs_post():
    values = request.get_json()['ids']
    groups = [GROUP_CACHE[v] for v in values if v in GROUP_CACHE]
    chart, textdf = get_ref_graph(groups)
    top_html = get_top_papers_html(textdf)
    out = io.StringIO()
    chart.save(out, format='json')
    dd = json.loads(out.getvalue())
    dd['papers'] = top_html
    # final = out.getvalue()[:-1] + ', "papers": "' + top_html + '"}'
    return json.dumps(dd)

def get_group_df(group):
    items = group.get_all_items()
    papers = [i for i in items if i['paperId'] is not None]
    df = pd.DataFrame(papers)
    return df

def get_ref_df(source_df, group):
    all_df = []
    for jx, j in source_df.iterrows():
        pool = ThreadPool(processes=60)
        p = j.paperId
        source_title = j.title
        try:
            ref_titles = get_references(p)['data']
        except KeyError:
            continue

        ref_ids = [i['citedPaper']['paperId'] for i in ref_titles]
        ref_ids = [x for x in ref_ids if x is not None]
        ref_data = pool.map(get_details,ref_ids)
        ref_papers = [i for i in ref_data if type(i) == dict]
        ref_df = pd.DataFrame(ref_papers)

        if ref_df.shape[0] > 0:
            first_author = []
            for i in ref_df.authors:
                #print(i)
                if type(i) != float and len(i)>0:
                    first_author.append(i[0]['name'])
                else:
                    first_author.append('NA')
            ref_df['first_author'] = first_author
            ref_df['group'] = [group.title] * ref_df.shape[0]
            ref_df['source_title'] = source_title
        all_df.append(ref_df)

    return pd.concat(all_df)

def get_source_df(groups): #groups is a list of group objects
    all_df = pd.DataFrame()
    for g in groups: #loop through all the selected papers as 'source papers'
        df = get_group_df(g)

        first_author = []
        for i in df.authors:
            first_author.append(i[0]['name'])

        df['first_author'] = first_author
        df['group'] = [g.title] * df.shape[0]
        df['source_title'] = [None] * df.shape[0]
        # df['ref_title'] = df['title']
        all_df = pd.concat([all_df,df])

        #add all the references from this source paper to the dataframe
        ref_df = get_ref_df(df,g)
        all_df = pd.concat([all_df,ref_df])

    all_df = all_df[~all_df['title'].isna()]

    shared_dict = dict(all_df.title.value_counts())
    all_df['shared_by'] = [shared_dict[i] for i in all_df.title]
    return all_df


def get_ref_graph(user_data_groups = None):
    if user_data_groups == None:
        merged_df = pd.read_pickle('data/reference_df_with_tags.pkl')
        merged_edges = merged_df.groupby(['ref_title', 'class_paper']).size().reset_index(name='count')

        merged = merged_df.query('citationCount != 0')
        merged = merged_df[~merged_df.year.isna()]

        source_df = merged.copy()
    else:
        source_df = get_source_df(user_data_groups)
        source_df = source_df[~source_df['citationCount'].isna()]
        source_df = source_df.query('citationCount != 0')
        source_df['value'] = source_df['group']

    source_df = source_df.reset_index()

    source_bar = source_df.copy().iloc[:,-4:].drop_duplicates()
    unique_idxs = source_bar.index.tolist()

    unique_idxs = source_bar.index.tolist()

    is_valid = [True if i in unique_idxs else False for i in source_df.index.tolist() ]
    source_df['is_valid'] = is_valid

    # plot!
    source = source_df[:5000].sort_values('shared_by')
    year_ticks = [int(i) for i in np.arange(1880,2030,10)]

    # pts = alt.selection(type="multi", encodings=['y','color'])
    # Top panel is scatter plot of temperature vs time
    points = alt.Chart(source).mark_point().encode(
        alt.X('year:N', title='Year',
             axis=alt.Axis(values=year_ticks)),
        alt.Y('citationCount:Q',
            title='Citation Count',
            scale=alt.Scale(type="log")
        ),
        alt.Color('shared_by:Q',scale=alt.Scale(scheme='goldorange', domainMin=0),
                  legend=alt.Legend(title = 'Shared References')),
        tooltip=['title','first_author', 'year'],
        # color=alt.condition(brush, color, alt.value('lightgray')),
        size=alt.Size('shared_by:Q')
    ).properties(
        width=700,
        height=450
    )
    # .transform_filter(
    #     pts
    # )

    scale = alt.Scale(domain=['theory', 'tools'],
                      range=['#249EA0', '#005F60'])
    color = alt.Color('modes:N', scale=scale, legend=None)

    # bars = alt.Chart(source.dropna()).transform_filter(
    #     alt.FieldEqualPredicate(field='is_valid', equal=True)
    # ).mark_bar().encode(
    #     y='value',
    #     x='count()',
    #     color=alt.condition(pts, color, alt.value('gray'))
    # ).properties(
    #     width=200
    # ).add_selection(pts)

    # Base chart for data tables
    source_text = source.drop_duplicates('title', keep="first").sort_values('shared_by', ascending=False)
    # ranked_text = alt.Chart(source_text).mark_text().encode(
    #     y=alt.Y('row_number:O',axis=None)
    # ).transform_window(
    #     row_number='row_number()'
    # ).transform_filter(
    #     pts
    # ).transform_window(
    #     rank='rank(row_number)'
    # ).transform_filter(
    #     alt.datum.rank<7
    # ).properties(
    #     width = 10
    # )

    # Data Tables
    # year = ranked_text.encode(text='year:N').properties(title='Year')
    # title = ranked_text.encode(text='title').properties(title='Paper Title')
    # cites = ranked_text.encode(text='citationCount:Q').properties(title='Citations')
    # sharedby = ranked_text.encode(text='shared_by:Q').properties(title='Shared')
    # text = alt.hconcat(title,sharedby,cites,year) # Combine data tables

    # # # Build chart
    # chart_pt1 = alt.hconcat(
    #     bars,
    #     text,
    # )

    # chart = alt.vconcat(
    #     text,
    #     points
    # ).configure_title(
    #     fontSize=20,
    #     font='Courier',
    #     anchor='start',
    #     color='darkorange'
    # ).configure_legend(
    #     labelLimit=0,
    #     strokeColor='gray',
    #     fillColor='#EEEEEE',
    #     padding=10,
    #     cornerRadius=10,
    #     orient='bottom-left'
    # ).configure_view(
    #     strokeWidth=0
    # )
    
    return points, source_text


@app.route('/')
def hello():
    return "Hello from CitePal server"

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=7500, threaded=True)
