#!/usr/bin/env python3

'''
Plot the number of Github stars received over time for a user.

Steps:
1. Get user's repo names from Github API
2. Get repo's star counts from Github API
3. Plot stars over time
'''

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="GitHub Star History",
    page_icon="⭐",
    layout="wide"
)
import plotly.graph_objects as go
import concurrent.futures
from typing import List, Dict, Optional
from datetime import datetime, timedelta


MAX_NUM_REPOS = None


def check_rate_limit_exceeded(response) -> Optional[datetime]:
    """Returns reset time if rate limited, None otherwise"""
    if isinstance(response, dict):
        json_response = response
        headers = {}  # No headers for dict responses
    else:
        json_response = response.json()
        headers = response.headers
        
    if 'message' in json_response and 'rate limit exceeded' in json_response['message']:
        reset_timestamp = int(headers.get('x-ratelimit-reset', 0))
        reset_time = datetime.fromtimestamp(reset_timestamp)
        remaining = int(headers.get('x-ratelimit-remaining', 0))
        limit = int(headers.get('x-ratelimit-limit', 0))
        
        message = f"""
        GitHub API rate limit exceeded!
        - Remaining: {remaining}/{limit} requests
        - Resets at: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        st.warning(message)
        return reset_time
    return None

# check if ./streamlit/secrets.toml exists
if os.path.exists('./.streamlit/secrets.toml'):
    # read GITHUB_AUTH_TOKEN from secrets.toml
    GITHUB_AUTH_TOKEN = st.secrets['github_auth_token']
else:
    # read GITHUB_AUTH_TOKEN from environment variable
    if 'GITHUB_AUTH_TOKEN' in os.environ:
        GITHUB_AUTH_TOKEN = os.environ['GITHUB_AUTH_TOKEN']
    else:
        GITHUB_AUTH_TOKEN = None

query_params = st.query_params
print("query_params:", query_params)


# Get user's repo names from Github API
username_default = query_params.get('username', '') 

def parse_github_input(input_str: str) -> tuple[str, Optional[str]]:
    """Parse username or username/repo from input string"""
    input_str = input_str.strip()
    
    # Handle full GitHub URLs
    if input_str.startswith(('http://', 'https://')):
        parts = input_str.split('github.com/')
        if len(parts) > 1:
            input_str = parts[1].strip('/')
    
    # Split into username and optional repo
    parts = input_str.split('/')
    username = parts[0]
    repo = parts[1] if len(parts) > 1 else None
    
    return username, repo

github_input = st.sidebar.text_input('Enter Github username or username/repo:', username_default)
username, repo_from_input = parse_github_input(github_input)

# Get repo name from query params or input
repo_default = repo_from_input or query_params.get('repo', '')

# st.experimental_set_query_params(username=username)


datetime_tomorrow = datetime.now() + timedelta(days=1)
datetime_tomorrow_midnight = datetime(datetime_tomorrow.year,
                                    datetime_tomorrow.month,
                                    datetime_tomorrow.day)

num_days_default = query_params.get('num_days', '')

# Get x number of days to show.
num_days = st.sidebar.text_input('Enter number of days to show:', num_days_default)

datetime_today = datetime.now()
datetime_today_midnight = datetime(datetime_today.year,
                                 datetime_today.month,
                                 datetime_today.day)
if num_days:
    datetime_start = datetime_today_midnight - timedelta(days=int(num_days))
    datetime_end = datetime_today_midnight
else:

    datetime_start_str = query_params.get('datetime_start', '2008-01-01')
    try:
        datetime_start = datetime.strptime(datetime_start_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        datetime_start = datetime.strptime('2008-01-01', "%Y-%m-%d")

    datetime_end_str = query_params.get('datetime_end', '')
    if datetime_end_str:
        try:
            datetime_end = datetime.strptime(datetime_end_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            datetime_end = datetime_tomorrow_midnight
    else:
        datetime_end = datetime_tomorrow_midnight




# Add slider to sidebar that allows to select daterange.
st.sidebar.markdown('**Date range**')
date_range = st.sidebar.slider('Select date range:',
    value=(datetime_start,
           datetime_end),
    min_value=datetime.strptime('2008-01-01', "%Y-%m-%d"),
    max_value=datetime_tomorrow_midnight,
    format='YYYY-MM-DD')


datetime_start, datetime_end = date_range
datetime_start_str = datetime_start.strftime("%Y-%m-%d %H:%M:%S")
datetime_end_str = datetime_end.strftime("%Y-%m-%d %H:%M:%S")
st.query_params.update({
    "datetime_start": datetime_start_str,
    "datetime_end": datetime_end_str,
    "num_days": num_days,
    "username": username
})


# Prompt user to enter username if none is entered.
if username == '':
    st.sidebar.text('Enter Github username in sidebar.')
    st.sidebar.markdown('**Example**: tom-doerr')
    # stop
    st.warning('Please enter a Github username in the sidebar.')
    st.stop()


url = 'https://api.github.com/users/{}/repos'.format(username)
if GITHUB_AUTH_TOKEN:
    headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN)}
else:
    headers = {}


return_dict = {}
while True:
    response = requests.get(url, headers=headers)
    check_rate_limit_exceeded(response)
    data = response.json()
    for repo in data:
        return_dict[repo['name']] = repo
    if 'next' not in response.links.keys():
        break
    else:
        url = response.links['next']['url']

reponames = sorted(return_dict.keys())
selected_repo = st.sidebar.selectbox('Select repository (optional):', 
                                   ['All repositories'] + reponames,
                                   index=0 if not repo_default else reponames.index(repo_default) + 1)

if selected_repo != 'All repositories':
    reponames = [selected_repo]
    st.query_params['repo'] = selected_repo
else:
    st.query_params.pop('repo', None)


# Get repo's star counts from Github API with the starred_at attribute using the v3 API.
url = 'https://api.github.com/repos/{}/{}/stargazers'.format(username, reponames[0])
r = requests.get(url, headers=headers)
star_count = len(r.json())
print('star_count:', star_count)

def get_repo_stars_page(username: str, repo: str, page: int, headers_accept: Dict) -> List[str]:
    url = f'https://api.github.com/repos/{username}/{repo}/stargazers?page={page}&per_page=100'
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers_accept)
        
        try:
            data = r.json()
        except json.JSONDecodeError:
            st.error(f"Invalid JSON response for {repo} at page {page}")
            return []
            
        reset_time = check_rate_limit_exceeded(r)
        if reset_time is None:
            if isinstance(data, str):
                st.error(f"Unexpected string response for {repo} at page {page}: {data}")
                return []
            if not data:
                return []
            if not isinstance(data, list):
                st.error(f"Unexpected response type for {repo} at page {page}: {type(data)}")
                return []
                
            return [user['starred_at'] for user in data]
            
        wait_time = (reset_time - datetime.now()).total_seconds()
        if wait_time > 0:
            st.info(f"Waiting {wait_time:.0f} seconds for rate limit reset...")
            time.sleep(wait_time + 1)  # Add 1 second buffer
            continue
            
        retry_delay *= 2  # Exponential backoff
        if attempt < max_retries - 1:
            st.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            
    st.error("Max retries exceeded. Please try again later.")
    st.stop()

@st.cache_data(show_spinner=False, ttl=3600)
def get_repo_stars(username: str, repo: str) -> List[str]:
    starred_at = []
    headers_accept = headers.copy()
    headers_accept['Accept'] = 'application/vnd.github.v3.star+json'
    
    # Get first page and check total stars from API
    repo_url = f'https://api.github.com/repos/{username}/{repo}'
    r = requests.get(repo_url, headers=headers_accept)
    if r.status_code == 200:
        total_stars = r.json().get('stargazers_count', 0)
    else:
        st.error(f"Failed to get repo info for {repo}")
        return []
    
    first_page = get_repo_stars_page(username, repo, 1, headers_accept)
    if not first_page:
        return []
    
    starred_at.extend(first_page)
    
    if len(starred_at) >= total_stars:
        st.write(f"Collected all {total_stars} stars for {repo}")
        return starred_at
    
    # Calculate remaining pages
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        page = 2
        future_to_page = {}
        while len(starred_at) < total_stars:  # Continue until we have all stars
            future = executor.submit(get_repo_stars_page, username, repo, page, headers_accept)
            future_to_page[future] = page
            
            if len(future_to_page) >= 10:  # Max 10 concurrent requests
                done, _ = concurrent.futures.wait(future_to_page, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results = future.result()
                    if not results:  # No more results on this page
                        continue
                    starred_at.extend(results)
                    del future_to_page[future]
            
            page += 1
            time.sleep(0.1)  # Small delay between spawning requests
            
    return starred_at

# Plot stars over time
@st.cache_data(ttl=3600, show_spinner=True)
def get_stars_over_time(reponames, username, progress_text, progress_bar):
    '''
    Get the number of stars for each repo in reponames over time.
    '''
    stars_over_time = []
    repos_stared_at_lists = {}
    total_repos = len(reponames[:MAX_NUM_REPOS])
    for repo_num, reponame in enumerate(reponames[:MAX_NUM_REPOS]):
        progress = (repo_num + 1) / total_repos
        if progress_text and progress_bar:
            progress_text.text(f'Processing {reponame} ({repo_num + 1}/{total_repos})')
            progress_bar.progress(progress)
        
        repos_stared_at = []
        star_dates_and_times = get_repo_stars(username, reponame)
        repos_stared_at_lists[reponame] = star_dates_and_times
    return repos_stared_at_lists


# Plot stars over time


def plot_stars_over_time_plotly(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot stars over time using Plotly.
    '''
    fig = go.Figure()
    for reponame in repos_stared_at_lists.keys():
        dates = repos_stared_at_lists[reponame]
        dates_filtered = []
        for date in dates:
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                
        # Sort dates and generate y values after sorting
        dates_filtered.sort()
        y_filtered = [i + 1 for i in range(len(dates_filtered))]
        
        fig.add_trace(go.Scatter(x=dates_filtered, y=y_filtered, name=reponame))


    # Remove borders.
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Move legend below plot
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # fig.update_layout(template="plotly_dark")



    st.plotly_chart(fig, use_container_width=True, key="combined_plot")












def plot_stars_over_time_all(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot stars over time.
    '''

    dates_all = []
    for reponame in repos_stared_at_lists.keys():
        dates = repos_stared_at_lists[reponame]
        dates_all.extend(dates)

    dates_all = sorted(dates_all)
    y = [i + 1 for i, _ in enumerate(dates_all)]

    dates_filtered = []
    y_filtered = []
    for date, y_val in zip(dates_all, y):
        for reponame in repos_stared_at_lists.keys():
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                y_filtered.append(y_val)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_filtered, y=y_filtered))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True, key="all_stars_plot")





@st.cache_data(show_spinner=False, ttl=3600)
def filter_stared_list(repos_stared_at_lists, date_range):
    '''
    Filter the stared list with the date range.
    '''
    repos_stared_at_lists_filtered = {}
    for reponame, stared_list in repos_stared_at_lists.items():
        stared_list_filtered = [star_date for star_date in stared_list if star_date >= date_range[0] and star_date <= date_range[1]]
        repos_stared_at_lists_filtered[reponame] = stared_list_filtered
    return repos_stared_at_lists_filtered


@st.cache_data(show_spinner=False, ttl=3600)
def convert_to_datetime(repos_stared_at_lists):
    '''
    Convert the dates in the stared list to datetime objects.
    '''
    repos_stared_at_lists_datetime = {}
    for reponame, stared_list in repos_stared_at_lists.items():
        stared_list_datetime = [datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in stared_list]
        repos_stared_at_lists_datetime[reponame] = stared_list_datetime

    return repos_stared_at_lists_datetime


def num_stars_received_last_x_hours(repos_stared_at_lists, hours=24):
    '''
    Get the number of stars received in the last x hours.
    '''
    repos_stared_at_lists_datetime_filtered = {}
    now = datetime.now()
    for reponame, stared_list in repos_stared_at_lists.items():
        stared_list_filtered = [star_date for star_date in stared_list if now - star_date <= timedelta(hours=hours)]
        repos_stared_at_lists_datetime_filtered[reponame] = stared_list_filtered

    _sum = 0
    for reponame, stared_list in repos_stared_at_lists_datetime_filtered.items():
        _sum += len(stared_list)

    return repos_stared_at_lists_datetime_filtered, _sum


def plot_stars_repos_individually(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot the stars received in the last x hours for each repo using plotly.
    '''


    # Sort repos by number of stars.
    repo_stars_received_last_x_hours = []
    for reponame in repos_stared_at_lists.keys():
        stared_list_filtered, _sum = num_stars_received_last_x_hours(repos_stared_at_filtered, hours=24)
        repo_stars_received_last_x_hours.append((reponame, _sum))

    # Sort repos by number of stars.
    repos_ordered = sorted(repos_stared_at_lists.keys(), key=lambda x: len(repos_stared_at_lists[x]), reverse=True)

    for repo_num, reponame in enumerate(repos_ordered[:MAX_NUM_REPOS]):
        # Skip if no stars.
        num_stars = len(repos_stared_at_filtered[reponame])
        if num_stars == 0:
            continue
        fig = go.Figure()
        st.subheader(f'{reponame}')


        dates = repos_stared_at_lists[reponame]
        dates_filtered = []
        for date in dates:
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                
        # Sort dates and generate y values after sorting
        dates_filtered.sort()
        y_filtered = [i + 1 for i in range(len(dates_filtered))]

        # plot using a line plot

        # fig = px.line(x=repos_stared_at_lists_datetime_filtered[reponame], y=np.arange(num_stars), title='Stars over time')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_filtered, y=y_filtered, name=reponame))
        # fig.update_xaxes(nticks=20)
        # st.plotly_chart(fig)

        # Remove borders.
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Move legend below plot
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        st.plotly_chart(fig, use_container_width=True, key=f"repo_plot_{reponame}")



    # for reponame in repos_stared_at_lists_datetime_filtered_sorted:
        # # Check if repo has stars.
        # if len(repos_stared_at_lists_datetime_filtered[reponame]) == 0:
            # continue
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=[star_date.strftime("%d-%b-%Y %H:%M:%S") for star_date in repos_stared_at_lists_datetime_filtered[reponame]],
                                # y=[i for i, _ in enumerate(repos_stared_at_lists_datetime_filtered[reponame])],
                                # mode='markers',
                                # name=reponame))

        # # Show plot in streamlit.
        # fig.update_layout(
            # margin=dict(l=0, r=0, t=0, b=0),
            # paper_bgcolor="white",
        # )
        # fig.update_xaxes(nticks=20)

        # st.plotly_chart(fig)



# st.title('Github stars over time')
def main() -> None:
    st.title('GitHub Star History')
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Fetching repository data...")
    repos_stared_at_lists = get_stars_over_time(reponames, username, progress_text, progress_bar)
    
    progress_text.text("Processing dates...")
    repos_stared_at_lists = convert_to_datetime(repos_stared_at_lists)
    
    progress_text.empty()
    progress_bar.empty()
    # Show total num stars for user.
    total_stars = 0
    for reponame, stared_list in repos_stared_at_lists.items():
        total_stars += len(stared_list)
    st.subheader('Total stars: {}'.format(total_stars))

    repos_stared_at_filtered = filter_stared_list(repos_stared_at_lists, date_range)
    
    # Only show combined plots when viewing all repositories
    if selected_repo == 'All repositories':
        plot_stars_over_time_plotly(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)
        plot_stars_over_time_all(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)
        
    num_stars_last_24_hours_repos, num_stars_last_24_hours_sum = num_stars_received_last_x_hours(repos_stared_at_lists, hours=24)

    stars_last_7_days = num_stars_received_last_x_hours(repos_stared_at_lists, hours=7*24)
    stars_last_30_days = num_stars_received_last_x_hours(repos_stared_at_lists, hours=30*24)

    st.subheader('Number of stars received')
    st.text('last 24 hours: {}'.format(num_stars_last_24_hours_sum))
    st.text('last 7 days: {}'.format(stars_last_7_days[1]))
    st.text('last 30 days: {}'.format(stars_last_30_days[1]))

    plot_stars_repos_individually(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)


if __name__ == '__main__':
    main()

