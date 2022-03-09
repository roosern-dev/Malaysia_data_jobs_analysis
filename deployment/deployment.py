from threading import local
from tkinter.messagebox import NO
import streamlit as st
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
import numpy as np
import time
#from selenium import webdriver
from bs4 import BeautifulSoup
#from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import re
import os
from lxml import etree
import lxml.html, lxml.html.clean
from urllib.parse import urlparse
from urllib.parse import parse_qs
import pandas as pd
from itertools import product
import time
import regex
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


header = st.container()
dataset = st.container()
features = st.container()
inference = st.container()
results = st.container()


if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = 0



@st.cache
def setup():
    clean_path = r"C:\Users\ryeoh\Project\Malaysia_data_jobs_analysis\data\inferencing.ftr"
    df_data_jobs_clean_description = pd.read_feather(clean_path)
    st.session_state.data_model = df_data_jobs_clean_description
    st.session_state.job_locations = df_data_jobs_clean_description.state.unique().tolist()

    vec_count1 = CountVectorizer(ngram_range=(1,1), max_features=50).fit(df_data_jobs_clean_description['job_requirement'].tolist())
    vec_count2 = CountVectorizer(ngram_range=(2,2), max_features=50).fit(df_data_jobs_clean_description['job_requirement'].tolist())


def cleanData(raw_text):    
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.wordnet import WordNetLemmatizer

    # Custom Stopwords
    new_words = ["missing","abstract","we","in","the","using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","skill",
    'business', 'experience', 'team', 'work', 'job',
    'amp', 'information', 'degree', 'management', 'development',
    'skill', 'support', 'project', 'solution', 'working',
    'process',
    'skill', 'informationcareer', 'knowledge',
    'customer', 'additional', 'year', 'timejob', 'typefull',
    'product', 'design', 'solution', 'technical', 'service', 'quality',
    'requirement', 'science',
    'year',
    'application', 'tool', 'professional', 'related',
    'specializationscomputer', 'diploma', 'good', 'graduate', 'ensure',
    'company', 'problem', 'strong', 'report', 'ability',
    'year', 'provide',
    'role', 'develop', 'requirement', 'tool', 'client',
    'performance', 'specifiedjob', 'user',
    'specifiedqualificationnot', 'levelnot', 'including','skill',
    'skill','solution', 'year',
    'requirement', 'tool', 'communication', 'environment', 'system',
    'reporting', 'across', 'platform', 'â', 'post', 'project', 'able',
    'service', 'process',
    'opportunity', 'model',
    'stakeholder', 'time', 'required','global', 'master', 'industry', 'insight', 'high', 'risk', 'level',
    'marketing', 'maintain', 'need', 'understanding', 'relevant', 'test',
    'within', 'end', 'bachelor', 'key', 'internal', 'issue', 'build',
    'best', 'least', 'responsible', 'well', 'people', 'based', 'etc',
    'digital',
    'help',
    'various', 'financial', 'yearsjob', 'improvement', 'operation', 'world',
    'testing', 'identify', 'candidate', 'degreeyears', 'source',
    'implementation', 'u', '3', 'functional', 'implement', 'part',
    'standard', 'g', 'activity', 'change', 'network', 'manage', 'must',
    'field', 'descriptionjob', 'make', 'perform', 'career', 'advanced', '2',
    'drive','e','group','solving','5','research','decision', 'excellent', 'lead',
       'like','group',
       'solving', '5', 'research','decision', 'excellent', 'lead',
       'like', 'looking', 'practice','market', 'written', 'area', 'create',
       'language', 'use', 'security', 'deliver', 'complex','apply','life', 'malaysia','understand',
       'day','improve', 'leading', 'office', 'finance','closely', '1', 'equivalent','compliance', 'added', 'services',
       'initiative', 'building', 'developing', 'position','english', 'banking', 'different',
       'requirements', 'review', 'existing', 'would','meet', 'highly', 'technique', 'take',
       'code', 'advantage', 'set', 'value', 'minimum', 'strategy','member',
       'executivequalificationbachelor', 'maintenance','documentation', 'operational','responsibility','communicate',
       'procedure','assist', 'managing', 'trend','integrity', 'levelsenior', 'organization',
       'providing', 'leave', 'responsibilities', 'capability','planning', 'employee', 'production', 'document','ms', '4','multiple', 'external','partner',
       'country','governance', 'cross',
       'expertise', 'relationship','monitor', 'conduct','great', 'growth', 'effective', 'requirementsjob','self', 'way', 'oriented', 'independently', 'senior', 'supporting',
       'impact', 'continuous', 'available', 'challenge', 'posse', 'timely',
       'department','preferably', 'innovative', 'ad', 'following',
       'making', 'current', 'detail', 'effectively','higher', 'approach', 'right', 'critical', 'case', 'ensuring','fast', 'efficiency', 'culture',
       'hoc', 'expert''verbal', 'consumer', 'assigned', 'accurate', 'strategic',
       'methodology', 'core', 'benifitsepfsocsoannual', 'prepare', 'place','resource', 'creating', 'description', 'future',
       'appropriate','status', 'verbal','community','leadership', 'dynamic', 'flexible', 'include',
       'familiar', 'collaborate', 'leveljunior', 'interpersonal', 'designing',
       'individual', 'innovation', 'unit', 'regulatory', 'resolve', 'focus','matter', 'necessary', 'site',
       'daily', 'participate', 'execute', 'hand', 'date', 'others',
       'successful', 'background', 'please', 'cost', 'online', 'preferred',
       'committed', 'enhance', 'better', 'asset', 'real', 'achieve', 'portfolio', 'success', 'together', 'may', 'analyse', 'purpose', 'delivering',
       'leader', 'regional', 'every', 'kuala','brand', 'insurance', 'benefit', 'solve',
       'get', 'duty','asia', 'method', 'contribute', 'experience3', 'specification', 'talent', 'exposure','grow', 'general', 'center','track', 'positive','implementing', 'medical', 'mission', 'bank', 'personal', 'enable', 'experience5', 'human', 'contact', 'according', 'proven',
       'includes','supply', 'coordinate', 'performing', 'handling', 'passion','education','technology', 'software','region', 'order', 'domain', 'finding', 'share', 'actionable', 'identifying', 'principle', 'metric', 'range', 'medium', 'international', 'qualification', 'chain', 'lumpur', 'employment',
       'define', 'interpret', 'attention', 'backup', 'assurance', 'workflow', 'object', 'per', 'material',
       'proactive', 'point', 'attitude', 'social', 'incident', 'scalable', 'helping', 'proactively', 'hr', 'recovery', 'specializationssales', 'transform', 'national', 'collection', 'commercial', 'play', 'guideline', 'basis', 'university', 'subject', 'present', 'top', 'wide', 'sales', 'assessment', 'check',
       'provides', 'around', 'location', 'experienced','execution', 'audit', 'scientist','scale', 'record', 'concept', 'proficiency', 'basic', 'update', 'operating',
       'migration', 'variety', 'campaign', 'skills', 'structure', 'owner', 'thinking', 'goal', 'systems', 'multi','discipline', 'seek', 'predictive', 'similar', 'line','entry', 'health', 'enhancement', 'class', 'via', 'collaboration', 'action',
       '8', 'specializationsaccounting', 'growing', 'follow', 'equal', 'efficient', 'proficient', 'motivated', 'organisation', 'automated', 'component', 'player', '7', 'consulting', 'troubleshooting',
       'monthly', 'specific', 'diverse','hands', 'writing', 'availability','diversity', 'colleague', 'needed', 'many','know','passionate', 'provided', 'salary', 'hour', 'root', 'clear', 'cause', '000', 'commerce', 'rule',
       'non', 'guidance', 'fresh', 'priority', 'extraction','inclusive', 'administrative', 'staff', 'emerging', 'outcome', 'overall', 'resources', 'creative', 'used', 'meeting',
       'pre', 'demand', 'changing', 'willing', 'creation', 'demonstrate', 'event', 'excellence', 'solutions', 'limited','payment', 'tracking', 'manner', 'experience2', 'deploy', 'mapping',
       'head', 'paced','person', 'largest', 'credit', 'journey','feedback', 'established', 'possible', 'lake', 'physical', 'typecontractjob','progress', 'establish', 'experience1', 'preparation', 'yearjob', 'revenue', 'account',
       'first', 'spoken', 'cycle', 'com', 'quickly', 'consultant', 'error', 'configuration', 'specialist', 'essential', 'malaysian','focused', 'form', 'always',
        'complete', 'deadline', 'optimize', 'relational', 'channel', 'shared', 'bhd', 'scope', 'retail', 'provider', 'live', 'unique', 'defined', 'base', 'public', 'embrace', 'month', 'organizational', 'logistics', 'come', 'file', 'singapore', 'find', 'input', 'website', 'possess', 'pricing', 'quantitative'
        'tools','projects', 'processes','stakeholders', 'technologies', 'insights', 'customers','products', 'us', 'issues','operations', 'clients','opportunities','applications', 'users', 'activities', 'problems', 'delivery',
    'needs','standards', 'techniques','practices', 'integration','full','procedures','engineer', 'areas','sources','trends', 'tasks','members', 'sys', 'join','candidates','results', 'capabilities',
    'learn', 'plan','transformation', 'companies', 'employees','potential','control','projects','processes','levelentry','levelqualificationbachelor','teams', 'specializationsadmin','clerical', 'executivequalificationnot', 'specifiedyears',
    'ways', 'environments', 'structured', 'owners','goals', 'works','rules', 'accounting', 'associated', '10','pressure', 'translate', 'write', 'travel', 'solid', 'details', 'ongoing', 'sdn', 'setup', 'graduates', 'supervision', 'address', 'assets', 'centre',
    'controls', 'supports', 'markets', 'days', 'generate', 'advice','findings', 'believe', 'small', 'energy', 'corporate', 'professionals', 'extensive', 'involved', 'descriptionthe',
    'listed', 'bring', 'collaborative', 'influence', 'offers', 'values','principles', 'analysts','task', 'function', 'metrics','colleagues', 'developer', 'ideas','maintaining', 'plus', 'partners', 'policies', 'challenges',
    'countries', 'plans', 'sets','offer', 'coding', '6', 'open', 'framework', 'presentation', 'risks','documents','roles', 'media','guidelines', 'fields',
    'start', 'specializationssciences', 'automate', 'regular', 'load', 'annual', 'forward', 'capacity', 'familiarity', 'applying', 'takes', 'logical', 'terms', 'ownership', 'run', 'click', 'departments', 'care', 
    'levelmanagerqualificationbachelor', 'gender', 'seeking', 'communities', 'recommend', 'structures', 'stack', 'today', 'designs', 'deliverables', 'email', 'industries', 'back', 'comprehensive', 'tests', 'guide', 'assess', 'depth',
    'shortlisted', 'ready','executivequalificationdiploma', 'usage', 'effectiveness', 'search', 'especially', 'mindset', 'long', 'analyzing', 'sense', 'applied', 'junior','primary', 'encouraged', 'continuously',
    'sexual', 'orientation','demonstrated', 'currently', 'evaluate', 'employer', 'fully', 'engagement', 'defining', 'home', 'content', 'serve', 'validate', 'consistent','facilitate', 'handle', 'teaching', 'interested', 'reviews', 'resolution', 'fraud'
    ,'9', 'outcomes', 'act', 'involve', 'towards', 'minimal', 'www', 'matters', 'without', 'direction', 'important', 'prioritize', 'degreejob', 'productivity','recruitment', 'gathering', 'gather', 'units', 'improving', 'proud', 'jobs', 'datasets', 'economics', 'aspects', 'legacy', 'communications',
    'term', 'independent', 'brands', 'accordance', 'oral', 'actively']

    #new_words = []
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(set(new_words))
    tokenizer = nltk.RegexpTokenizer(r"\w+", )
    words = tokenizer.tokenize(raw_text)

    # lem = WordNetLemmatizer()
    # wordsLemmatized=[]
    # #Lemmatisation
    # for word in words:
    #     wordsLemmatized.append(lem.lemmatize(word))

    wordsFiltered=[]
    for word in words:
        if word.lower() not in stop_words:
            wordsFiltered.append(word)
    
    
    
    #      Convert to lowercase
    str=''
    for w in wordsFiltered:
        str = str+' '+w.lower()
    return str
        



def main(wage=0):
    print("running")
    number = 0
    with header:
        st.title("How much should you be paid, working in Data Science in Malaysia?")
        st.text("Predicting how much you should be fairly paid.")

    with inference:
        
        st.title("Predict your fair wage!")
        experience = st.number_input("Your years of experience", step=1, on_change=None, max_value=20)
        state = st.selectbox('Job location', st.session_state.job_locations, on_change=None)
        desired_job = st.selectbox("Preferred job", ['None','Data Scientist', 'Data Engineer', 'Data Analyst'], on_change=None)
        desc = st.text_input("Description", on_change=None)
        
        
            
        
        
        def run_inference():
            print('inference')
            st.session_state.predicted_salary = experience
            categories = {'Not Specified':0,'Non-Executive':1,'Entry Level':2,'Junior Executive':3,'Senior Executive':4,'Manager':5, 'Senior Manager':6}
            job_level_categories = {x[1]:x[0] for x in categories.items()}

            job_level=0
            if experience<2:
                job_level=2
            elif experience<4:
                job_level=3
            elif experience<8:
                job_level=4
            elif experience<12:
                job_level=5
            elif experience<20:
                job_level=6
            else:
                job_level=3

            job_level_cat = job_level_categories[job_level]

            '''
            input = {
            'job_description',
            'job_level':job_level_cat,
            'experience':experience,
            'job_type' :'Full-Time',
            'qualifications':"Bachelor's Degree, Post Graduate Diploma, Professional Degree, Master's Degree",
            'job_specialization':"Computer / Information Technology (Software)",
            'company_name',
            'company_registration',
            'company_size':"1 - 50 Employees",
            'company_industry',
            'job_link',
            'search_term',
            'state',
            'start_salary_range',
            'end_salary_range',
            'median_salary',
            'no_salary',
            'clean_job_description',
            'job_requirement',
            'is_data_scientist',
            'is_data_engineer',
            'is_data_analyst'}
            '''

            

        st.button("Calculate!", on_click=run_inference)
        st.title('Your recommended salary is: ')
        st.title(f'RM{st.session_state.predicted_salary}')

        




if __name__ =="__main__":
    setup()
    main()
