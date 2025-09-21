import streamlit as st
from pathlib import Path
from datetime import datetime

from research_agent import ARISCore, ResearchQuery
from utils.file_readers import read_text_file
from utils.config import PAGE_TITLE, EMBEDDINGS_DIR
from utils.visualization import plot_length_vs_impact

st.set_page_config(page_title=PAGE_TITLE, layout='wide')

# Initialize ARIS in session
if 'aris' not in st.session_state:
    with st.spinner('Initializing ARIS...'):
        st.session_state.aris = ARISCore()
aris = st.session_state.aris

st.title('🧠 ARIS - Autonomous Research Intelligence System')

with st.sidebar:
    st.header('📚 Document Ingestion')
    uploaded_files = st.file_uploader('Upload documents', accept_multiple_files=True, type=['txt', 'md', 'json', 'csv', 'pdf', 'docx'])
    if uploaded_files:
        with st.form('ingest'):
            source_type = st.selectbox('Document Type', ['research_paper', 'review_article', 'technical_report'])
            citation_count = st.number_input('Citation Count', min_value=0, value=0)
            publication_year = st.number_input('Publication Year', min_value=1990, max_value=datetime.now().year, value=datetime.now().year)
            authors_input = st.text_input('Authors (comma-separated)')
            keywords_input = st.text_input('Keywords (comma-separated)')
            submit = st.form_submit_button('Process')
            if submit:
                for file in uploaded_files:
                    content = read_text_file(file)
                    metadata = {'filename': file.name, 'source_type': source_type, 'citation_count': int(citation_count), 'publication_date': datetime(publication_year,1,1), 'authors': [a.strip() for a in authors_input.split(',') if a.strip()], 'keywords': [k.strip() for k in keywords_input.split(',') if k.strip()]}
                    try:
                        aris.ingest_research_document(content, file.name, metadata)
                        st.success(f'Processed {file.name}')
                    except Exception as e:
                        st.error(f'Failed to process {file.name}: {e}')
                st.experimental_rerun()

st.header('📊 Research Database')
docs = aris.get_all_documents()
if docs:
    st.metric('Documents', len(docs))
    avg_impact = sum(d.impact_score for d in docs) / len(docs)
    st.metric('Avg Impact Score', f'{avg_impact:.3f}')
    if st.button('Show impact vs length'):
        lengths = [len(d.content) for d in docs]
        impacts = [d.impact_score for d in docs]
        fig = plot_length_vs_impact(lengths, impacts)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info('No documents - upload to begin')

st.header('🔎 Conduct Research Query')
with st.form('research'):
    qtext = st.text_input('Query')
    domain = st.selectbox('Domain', ['computer_science','medicine','engineering','social_sciences'])
    research_type = st.selectbox('Research Type', ['systematic_review','literature_survey','exploratory'])
    depth = st.selectbox('Depth', ['surface','deep','exhaustive'])
    start_year = st.number_input('Start Year', min_value=1900, max_value=datetime.now().year, value=2000)
    end_year = st.number_input('End Year', min_value=1900, max_value=datetime.now().year, value=datetime.now().year)
    inclusion = st.text_input('Inclusion (comma-separated)')
    exclusion = st.text_input('Exclusion (comma-separated)')
    quality = st.slider('Quality Threshold', 0.0, 1.0, 0.5)
    submit_q = st.form_submit_button('Run Research')
    if submit_q:
        query = ResearchQuery(query=qtext, domain=domain, research_type=research_type, depth_level=depth, time_range=(datetime(start_year,1,1), datetime(end_year,12,31)), inclusion_criteria=[s.strip() for s in inclusion.split(',') if s.strip()], exclusion_criteria=[s.strip() for s in exclusion.split(',') if s.strip()], methodology_focus=[], quality_threshold=quality)
        result = aris.conduct_research(query)
        st.write('Results:')
        st.json(result)