import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('final_data.csv')
df['combined']  = df['product_name']+" "+df['about_product']
display = pd.read_csv('display.csv')
features = ['rating', 'rating_count_log', 'price_num_log']

cat_list = df['main_category'].unique().tolist()

def get_vector(filter_df):
    vect = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vect.fit_transform(filter_df['combined'])
    return tfidf_matrix

def get_similarity_score(tfidf_matrix, filter_df, features=features):
    df_num = filter_df[features]
    combined = hstack([tfidf_matrix, df_num])
    cos_sim = cosine_similarity(combined, combined)
    return cos_sim

def get_recommendations(cos_sim, filter_df):
    recommendations = {}
    
    for ind, row in enumerate(cosine_sim):
        similar_prdts = np.argsort(row)[::-1][1:6]
        prdts_names = filter_df.iloc[similar_prdts]['product_name'].tolist()
        prdts_ind = filter_df.iloc[similar_prdts]['product_name'].index.tolist()
        return prdts_ind
    

# Application
st.set_page_config(page_title="Product Recommendation System", layout="wide", page_icon="🛒")
st.title("ShopMate🛒: Intelligent Product Suggestions🚚⭐")
st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)

selected_cat = st.selectbox("Category:", cat_list)

if selected_cat:
    sub_cat_list = df[df['main_category']==selected_cat]['sub-category'].unique().tolist()
    selected_subcat = st.selectbox("Select a Subcategory", sub_cat_list)
    
    if selected_subcat:
        display_df = display[(display['main_category']==selected_cat)&(display['sub-category']==selected_subcat)]
        filtered_df = df[(df['main_category']==selected_cat)&(df['sub-category']==selected_subcat)]
        
        tfidf = get_vector(filtered_df)
        cosine_sim = get_similarity_score(tfidf, filtered_df)
        
        recommendations = get_recommendations(cosine_sim, filtered_df)
        if not recommendations:
            recommendations = filtered_df.index.to_list()
            
            for i,ind in enumerate(recommendations):
                print(display_df.loc[ind])
        
        cols = st.columns(5)
        
        for i, ind in enumerate(recommendations):
            with cols[i % 5]:  
                st.write(f"**Name:** {display_df.loc[ind, 'product_name']}")
                st.write(f"**Price:** {display_df.loc[ind,'actual_price']}")
                st.write(f"**Link:** {display_df.loc[ind,'product_link']} ")
                st.image(display_df.loc[ind, 'img_link'], caption=display_df.loc[ind,'product_name'], use_container_width=False)
                st.write("---")

