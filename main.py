#!/usr/bin/env python3

'''
Plot the number of Github stars received over time for a user.

Steps:
1. Get user's repo names from Github API
2. Get repo's star counts from Github API
3. Plot stars over time
'''

import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta


MAX_NUM_REPOS = None


def check_rate_limit_exceeded(response) -> Optional[datetime]:
    """Returns reset time if rate limited, None otherwise"""
    if isinstance(response, dict):
        json_response = response
    else:
        json_response = response.json()
        
    if 'message' in json_response and 'rate limit exceeded' in json_response['message']:
        reset_timestamp = int(response.headers.get('x-ratelimit-reset', 0))
        reset_time = datetime.fromtimestamp(reset_timestamp)
        remaining = int(response.headers.get('x-ratelimit-remaining', 0))
        limit = int(response.headers.get('x-ratelimit-limit', 0))
        
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

username = st.sidebar.text_input('Enter Github username:', username_default)
username = username.strip()

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
    data = json.loads(response.text)
    check_rate_limit_exceeded(data)
    for repo in data:
        return_dict[repo['name']] = repo
    if 'next' not in response.links.keys():
        break
    else:
        url = response.links['next']['url']


reponames = list(return_dict.keys())


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
        reset_time = check_rate_limit_exceeded(r)
        
        if reset_time is None:
            return [user['starred_at'] for user in r.json()] if r.json() else []
            
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
    
    # Get first page to check total
    first_page = get_repo_stars_page(username, repo, 1, headers_accept)
    if not first_page:
        return []
    
    starred_at.extend(first_page)
    
    # Calculate remaining pages
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        page = 2
        future_to_page = {}
        while True:
            future = executor.submit(get_repo_stars_page, username, repo, page, headers_accept)
            future_to_page[future] = page
            
            if len(future_to_page) >= 10:  # Max 10 concurrent requests
                done, _ = concurrent.futures.wait(future_to_page, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results = future.result()
                    if not results:  # No more pages
                        executor.shutdown(wait=False)
                        return starred_at
                    starred_at.extend(results)
                    del future_to_page[future]
            
            page += 1
            time.sleep(0.1)  # Small delay between spawning requests
            
    return starred_at

# Plot stars over time
@st.cache_data(ttl=3600, show_spinner=True)
def get_stars_over_time(reponames, username):
    '''
    Get the number of stars for each repo in reponames over time.
    '''
    stars_over_time = []
    repos_stared_at_lists = {}
    counter = st.empty()
    repo_name_text = st.empty()
    for repo_num, reponame in enumerate(reponames[:MAX_NUM_REPOS]):
        counter.text(f'Loading data for repo {repo_num} of {len(reponames)}...')
        repo_name_text.text(f'Processing {reponame}')
        repos_stared_at = []
        star_dates_and_times = get_repo_stars(username, reponame)

        repos_stared_at_lists[reponame] = star_dates_and_times

    counter.text('')
    repo_name_text.text('')
    return repos_stared_at_lists


# Plot stars over time
def plot_stars_over_time(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot stars over time.
    '''

    # Plot stars over time
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    for reponame in repos_stared_at_lists.keys():
        dates = repos_stared_at_lists[reponame]
        y = [i + 1 for i, _ in enumerate(repos_stared_at_lists[reponame])]
        dates_filtered = []
        y_filtered = []
        for date, y_val in zip(dates, y):
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                y_filtered.append(y_val)

        ax.plot(dates_filtered, y_filtered, label=reponame)


    # Format plot
    date_fmt = mdates.DateFormatter('%m-%d-%Y')
    ax.xaxis.set_major_formatter(date_fmt)
    _ = plt.xticks(rotation=90)
    # _ = plt.legend()
    _ = plt.title('Github stars over time')
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Number of stars')

    # Show legend below plot.
    _ = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), shadow=True, ncol=2)

    # Show plot in streamlit.
    fig = ax.get_figure()
    st.pyplot(fig)
    # plt.show()


def plot_stars_over_time_plotly(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot stars over time using Plotly.
    '''

    # Plot stars over time
    fig = go.Figure()
    for reponame in repos_stared_at_lists.keys():
        dates = repos_stared_at_lists[reponame]
        y = [i + 1 for i, _ in enumerate(repos_stared_at_lists[reponame])]
        dates_filtered = []
        y_filtered = []
        for date, y_val in zip(dates, y):
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                y_filtered.append(y_val)

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



    st.plotly_chart(fig, use_container_width=True)












def plot_stars_over_time_all(reponames, username, repos_stared_at_lists, repos_stared_at_filtered):
    '''
    Plot stars over time.
    '''

    # Plot stars over time
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    dates_all = []
    for reponame in repos_stared_at_lists.keys():
        # dates = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in repos_stared_at_lists[reponame]]
        dates = repos_stared_at_lists[reponame]
        dates_all.extend(dates)


    # Sort the dates.
    dates_all = sorted(dates_all)

    y = [i + 1 for i, _ in enumerate(dates_all)]

    dates_filtered = []
    y_filtered = []
    for date, y_val in zip(dates_all, y):
        for reponame in repos_stared_at_lists.keys():
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                y_filtered.append(y_val)


    if False:
        ax.plot(dates_filtered, y_filtered, label=reponame)

        # Format plot
        date_fmt = mdates.DateFormatter('%m-%d-%Y')
        ax.xaxis.set_major_formatter(date_fmt)
        _ = plt.xticks(rotation=90)
        _ = plt.title('Github stars over time')
        _ = plt.xlabel('Date')
        _ = plt.ylabel('Number of stars')

        # Show plot in streamlit.
        fig = ax.get_figure()
        st.pyplot(fig)


    # Same plot as above but using plotly.
    # fig = px.line(x=    dates_filtered, y=y_filtered, title='Github stars over time')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_filtered, y=y_filtered))
    # fig.update_xaxes(nticks=20)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # Remove x and y text
    # fig.update_layout(
        # xaxis_title="",
        # yaxis_title="",
    # )
    # fig.update_layout(legend=dict(
        # yanchor="top",
        # y=0.99,
        # xanchor="left",
        # x=0.01
    # ))

    st.plotly_chart(fig, use_container_width=True)





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
        stared_list_filtered = [star_date for star_date in stared_list if now - star_date <= datetime.timedelta(hours=hours)]
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
        y = [i + 1 for i, _ in enumerate(repos_stared_at_lists[reponame])]
        dates_filtered = []
        y_filtered = []
        for date, y_val in zip(dates, y):
            if date in repos_stared_at_filtered[reponame]:
                dates_filtered.append(date)
                y_filtered.append(y_val)

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

        st.plotly_chart(fig, use_container_width=True)



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
def main():
    repos_stared_at_lists = get_stars_over_time(reponames, username)
    repos_stared_at_lists = convert_to_datetime(repos_stared_at_lists)
    # Show total num stars for user.
    total_stars = 0
    for reponame, stared_list in repos_stared_at_lists.items():
        total_stars += len(stared_list)
    st.subheader('Total stars: {}'.format(total_stars))

    repos_stared_at_filtered = filter_stared_list(repos_stared_at_lists, date_range)
    # plot_stars_over_time(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)
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

