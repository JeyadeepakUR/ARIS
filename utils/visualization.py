import plotly.express as px
import pandas as pd
from typing import List, Dict, Any

def plot_length_vs_impact(lengths: List[int], impacts: List[float]):
    df = pd.DataFrame({'length': lengths, 'impact': impacts})
    fig = px.scatter(df, x='length', y='impact', trendline='ols', title='Document Length vs Impact')
    return fig

def plot_method_timeline(timeline_data: List[Dict[str, Any]]):
    df = pd.DataFrame(timeline_data)
    fig = px.line(df, x='Year', y='Count', color='Method', title='Research Methodology Trends Over Time')
    return fig

def plot_author_productivity(df_authors):
    # df_authors: pandas DataFrame with Author, Documents, Total Impact
    df = df_authors.copy()
    df['Avg Impact'] = df['Total Impact'] / df['Documents']
    fig = px.bar(df, x='Documents', y='Author', color='Avg Impact', orientation='h', title='Top Authors by Productivity')
    return fig