from random import sample
from threading import local
import sklearn
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
import pickle
from PIL import Image

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


header = st.container()
dataset = st.container()
features = st.container()
inference = st.container()
results = st.container()
visual = st.container()


if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = 0

@st.cache
def get_data_model():
    clean_path = r"./data/inferencing.ftr"
    df_data_jobs_clean_description = pd.read_feather(clean_path)
    return df_data_jobs_clean_description

@st.cache
def get_vec_count1(df:pd.DataFrame):
    vec_count1 = CountVectorizer(ngram_range=(1,1), max_features=50).fit(df['job_requirement'].tolist())
    return vec_count1

@st.cache
def get_vec_count2(df:pd.DataFrame):
    vec_count2 = CountVectorizer(ngram_range=(2,2), max_features=50).fit(df['job_requirement'].tolist())
    return vec_count2

@st.cache
def setup():
    clean_path = r"./data/inferencing.ftr"
    df_data_jobs_clean_description = pd.read_feather(clean_path)
    st.session_state.data_model = df_data_jobs_clean_description
    st.session_state.job_locations = df_data_jobs_clean_description.state.unique().tolist()

    vec_count1 = CountVectorizer(ngram_range=(1,1), max_features=50).fit(df_data_jobs_clean_description['job_requirement'].tolist())
    vec_count2 = CountVectorizer(ngram_range=(2,2), max_features=50).fit(df_data_jobs_clean_description['job_requirement'].tolist())

@st.cache
def load_model():
    filename=r"./data/model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
    
@st.cache(suppress_st_warning=True)
def get_visualizations(df_data_jobs:pd.DataFrame):
    '''
    df_data_jobs = df_data_jobs[df_data_jobs['median_salary']<30000]

    df_jobs_by_state = df_data_jobs.groupby('state').size().sort_values()
    sns.set(rc={'figure.figsize':(20,20)})
    sns.set(font_scale=2)

    sns.set_palette('Reds')
    bar_states = sns.barplot(df_jobs_by_state.index, df_jobs_by_state.values)
    bar_states.set_title("Distribution of jobs by state", fontsize=40)
    bar_states.set_ylabel("Number of jobs", fontsize=20)
    bar_states.set_xlabel("States", fontsize=20)
    bar_states.bar_label(bar_states.containers[0])
    bar_states.tick_params(axis='x', rotation=90)

    df_jobs_by_industry = df_data_jobs.groupby('company_industry').size().sort_values()
    sns.set(rc={'figure.figsize':(20,20)})
    bar_industry = sns.barplot(df_jobs_by_industry.index, df_jobs_by_industry.values)
    bar_industry.set_title("Distribution of jobs by industry", fontsize=40)
    bar_industry.set_ylabel("Number of jobs", fontsize=20)
    bar_industry.set_xlabel("Industry", fontsize=20)
    bar_industry.tick_params(axis='x', rotation=90)
    bar_industry.bar_label(bar_industry.containers[0])

    df_median_salary_by_state = df_data_jobs.groupby('state')['median_salary'].median()
    sns.set(rc={'figure.figsize':(20,20)})
    sns.set(font_scale=2)
    ax_salary = sns.swarmplot(data=df_data_jobs, x="state", y="median_salary", color=".2")
    ax_salary = sns.boxplot(data= df_data_jobs, x="state", y="median_salary")
    ax_salary.set_title("Salary by state", fontsize=40)
    sns.set(font_scale=2)
    ax_salary.tick_params(axis='x', rotation=90)

    ax_salary.set_yticks(range(0,200000,4000))

    ax_company = sns.swarmplot(data=df_data_jobs, x="company_size", y="median_salary", color=".2")
    ax_company = sns.boxplot(data=df_data_jobs, x="company_size", y="median_salary")
    ax_company.tick_params(axis='x', rotation=90)
    ax_company.set_title("Salary by company size", fontsize=40)
    '''

    ''''''
    bar_states = Image.open(r"./images/distribution_of_jobs_by_state.png")
    bar_industry = Image.open(r"./images/distribution_of_jobs_by_industry.png")
    ax_salary = Image.open(r"./images/median_salary_by_state_swarm.png")
    ax_company = Image.open(r"./images/salary_by_company_size.png")

    return {'states':bar_states, 'industry':bar_industry, 'salary':ax_salary, 'company':ax_company}


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



def format_data(df_training:pd.DataFrame, df_description_tokenized:pd.DataFrame):
    df_training.drop(['job_id', 'job_title', 'job_description', 'job_specialization', 'company_name','company_registration', 'job_link', 'search_term', 'clean_job_description', 'job_requirement','start_salary_range','end_salary_range','median_salary'], inplace=True, axis=1)
    categories={'Not Specified':0,'Non-Executive':1,'Entry Level':2,'Junior Executive':3,'Senior Executive':4,'Manager':5, 'Senior Manager':6}
    df_training['job_level'] = df_training['job_level'].map(categories)
    job_types = ['Contract', 'Full-Time', 'Full-Time, Internship', 'Internship' ,'Part-Time', 'Temporary']
    df_job_type = pd.get_dummies(df_training['job_type'])
    df_training_en = df_training.join(df_job_type).drop('job_type',axis=1)
    company_sizes = {'1 - 50 Employees':0,'51 - 200 Employees':1,'201 - 500 Employees':2, '501 - 1000 Employees':3,'1001 - 2000 Employees':4,'2001 - 5000 Employees':5, 'More than 5000 Employees':6 }
    df_training_en['company_size'] = df_training_en['company_size'].map(company_sizes)
    df_states = pd.get_dummies(df_training_en['state'])
    df_training_en = df_training_en.drop('state',axis=1).join(df_states)
    df_industry = pd.get_dummies(df_training_en['company_industry'])
    df_training_en = df_training_en.drop('company_industry',axis=1).join(df_industry)
    df_training_full = df_training_en.join(df_description_tokenized)
    df_ready_training = df_training_full.drop(['qualifications', 'no_salary'], axis=1)
    df_ready_training.dropna(inplace=True)

    return df_ready_training
        



def main(data_model:pd.DataFrame, vec_1gram:CountVectorizer, vec_2gram:CountVectorizer, ML_model:sklearn.base.BaseEstimator, visualizations:dict):
    vec_1gram = vec_1gram
    vec_2gram = vec_2gram
    print("running")
    if 'job_locations' not in st.session_state:
        st.session_state.job_locations = data_model.state.unique().tolist()

    if "job_type" not in st.session_state:
        st.session_state.job_type = data_model.job_type.unique().tolist()

    if "company_size" not in st.session_state:
        st.session_state.company_size = data_model.company_size.unique().tolist()

    number = 0
    with header:
        st.title("How much should you be paid, working in Data Science in Malaysia?")
        st.text("Predicting how much you should be fairly paid.                         by Ru Sern")

    with inference:
        st.write("#")
        st.write("#")
        st.title("Predict your fair wage!")
        experience = st.number_input("Your years of experience", step=1, on_change=None, max_value=20)
        state = st.selectbox('Job location', st.session_state.job_locations, on_change=None)
        desired_job = st.selectbox("Preferred job", ['None','Data Scientist', 'Data Engineer', 'Data Analyst'], on_change=None)
        selected_company_size = st.selectbox("Preferred company size", st.session_state.company_size, on_change=None)

        sample_input = "I am a recent data science graduate from Forward School. I have experience building data analytics projects employing machine learning, deep learning, and big data. I have experience in scraping my own data and data mining. I can proficient in python. I also know pandas and sklearn."

        desc = st.text_input("Write a brief description of your skillsets here", on_change=None, placeholder=sample_input)
        
        
            
        
        
        def run_inference():
            print('inference')
            st.session_state.predicted_salary = experience
            categories = {'Not Specified':0,'Non-Executive':1,'Entry Level':2,'Junior Executive':3,'Senior Executive':4,'Manager':5, 'Senior Manager':6}
            job_level_categories = {x[1]:x[0] for x in categories.items()}

            job_description = cleanData(desc)
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

            data_job_type = {"is_data_scientist":0, "is_data_engineer":0, "is_data_analyst":0}
            if(desired_job == "Data Scientist"):
                data_job_type['is_data_scientist'] = 1
            elif(desired_job == "Data Engineer"):
                data_job_type['is_data_engineer'] = 1
            elif(desired_job == "Data Analyst"):
                data_job_type['is_data_analyst']=1

            count1_vector = vec_1gram.transform([job_description])
            count2_vector = vec_2gram.transform([job_description])

            df_count1 = pd.DataFrame(count1_vector.A, columns=vec_1gram.get_feature_names())
            df_count2 = pd.DataFrame(count2_vector.A, columns=vec_2gram.get_feature_names())
            df_description_tokenized = pd.concat([df_count1,df_count2], axis=1)
            df_description_tokenized = df_description_tokenized.applymap(lambda x: 1 if x>0 else 0)
            if selected_company_size == None:
                selected_company_size = '1 - 50 Employees'

            
            input_data = {'job_description':None,
            'job_level':job_level_cat,
            'experience':experience,
            'job_type' :'Full-Time',
            'qualifications':"Bachelor's Degree, Post Graduate Diploma, Professional Degree, Master's Degree",
            'job_specialization':"Computer/Information Technology, IT-Software",
            'company_name':None,
            'company_registration':None,
            'company_size':selected_company_size,
            'company_industry':'Computer / Information Technology (Software)',
            'job_link':None,
            'search_term':None,
            'state':state,
            'start_salary_range':0,
            'end_salary_range':0,
            'median_salary':0,
            'no_salary':None,
            'clean_job_description':None,
            'job_requirement':None}
            input_data.update(data_job_type)

            
            df_input =pd.DataFrame(input_data, index=[0])
            df_joined = pd.concat([data_model.copy(),df_input])
            
            #df_joined.drop(['job_id', 'job_title', 'job_description', 'job_specialization', 'company_name','company_registration', 'job_link', 'search_term', 'clean_job_description', 'job_requirement'], inplace=True, axis=1)
            #print(df_joined.tail(1))
            df_inference = format_data(df_joined, df_description_tokenized)
            print(df_inference.tail(1).values.tolist())
            prediction = ML_model.predict(df_inference.tail(1).values.tolist())
            print(prediction)
            st.session_state.predicted_salary = round(prediction[0])
        
            

            

        st.button("Calculate!", on_click=run_inference)
        st.write("#")
        st.title('Your recommended salary is: ')
        st.title(f'RM{st.session_state.predicted_salary}')


    #{'states':bar_states, 'industry':bar_industry, 'salary':ax_salary, 'company':ax_company}
    
    with visual:
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")

        st.title('Some facts to aid your job hunt.')
        st.write("#")
        st.write("#")
        
        st.title('Which state has the most data jobs?')
        st.image(visualizations['states'])

        st.write("#")
        st.write("#")

        st.title('which industry hires the most data science peeps?')
        st.image(visualizations['industry'])
        st.write("#")
        st.write("#")

        st.title('which state pays the highest?')
        st.image(visualizations['salary'])
        st.write("#")
        st.write("#")

        st.title('do bigger companies pay more?')
        st.image(visualizations['company'])
    

        




if __name__ =="__main__":
    data_model = get_data_model()
    vec_1gram = get_vec_count1(data_model)
    vec_2gram = get_vec_count2(data_model)
    ML_model = load_model()
    visualizations = get_visualizations(data_model)

    main(data_model, vec_1gram, vec_2gram, ML_model, visualizations)
