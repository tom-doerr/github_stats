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
import datetime
import time
import pandas as pd
import streamlit as st
import json
import plotly.express as px


MAX_NUM_REPOS = None

# Read auth token GITHUB_AUTH_TOKEN
GITHUB_AUTH_TOKEN = st.secrets['github_auth_token']

query_params = st.experimental_get_query_params()
print("query_params:", query_params)


# Get user's repo names from Github API
if 'username' in query_params:
    username_default = query_params['username'][0]
else:
    username_default = ''

username = st.sidebar.text_input('Enter Github username:', username_default)
username = username.strip()

# st.experimental_set_query_params(username=username)


datetime_tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
datetime_tomorrow_midnight = datetime.datetime(datetime_tomorrow.year,
                                               datetime_tomorrow.month,
                                               datetime_tomorrow.day)

if 'num_days' in query_params:
    num_days_default = query_params['num_days'][0]
else:
    num_days_default = ''

# Get x number of days to show.
num_days = st.sidebar.text_input('Enter number of days to show:', num_days_default)

if num_days:
    datetime_start = datetime.datetime.now() - datetime.timedelta(days=int(num_days))
    datetime_end = datetime.datetime.now()
else:

    if 'datetime_start' in query_params:
        datetime_start_str = query_params['datetime_start'][0]
        datetime_start = datetime.datetime.strptime(datetime_start_str, "%Y-%m-%d %H:%M:%S")
    else:
        datetime_start = datetime.datetime.strptime('2008-01-01', "%Y-%m-%d")


    if 'datetime_end' in query_params:
        datetime_end_str = query_params['datetime_end'][0]
        print("datetime_end_str:", datetime_end_str)
        datetime_end = datetime.datetime.strptime(datetime_end_str, "%Y-%m-%d %H:%M:%S")
    else:
        datetime_end = datetime_tomorrow_midnight




# Add slider to sidebar that allows to select daterange.
st.sidebar.markdown('**Date range**')
date_range = st.sidebar.slider('Select date range:',
    value=(datetime_start,
           datetime_end),
    min_value=datetime.datetime.strptime('2008-01-01', "%Y-%m-%d"),
    max_value=datetime_tomorrow_midnight,
    format='YYYY-MM-DD')


datetime_start, datetime_end = date_range
datetime_start_str = datetime_start.strftime("%Y-%m-%d %H:%M:%S")
datetime_end_str = datetime_end.strftime("%Y-%m-%d %H:%M:%S")
st.experimental_set_query_params(datetime_start=datetime_start_str,
                                 datetime_end=datetime_end_str,
                                 num_days=num_days,
                                    username=username)


# Prompt user to enter username if none is entered.
if username == '':
    st.sidebar.text('Enter Github username in sidebar.')
    st.sidebar.markdown('**Example**: tom-doerr')
    # stop
    st.warning('Please enter a Github username in the sidebar.')
    st.stop()


# username = 'tom-doerr'
url = 'https://api.github.com/users/{}/repos'.format(username)
headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN)}


return_dict = {}
while True:
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)
    for repo in data:
        return_dict[repo['name']] = repo
    if 'next' not in response.links.keys():
        break
    else:
        url = response.links['next']['url']


# reponames = [repo['name'] for repo in data]
reponames = list(return_dict.keys())

# r = requests.get(url, headers=headers)
# reponames = [repo['name'] for repo in r.json()]
# print("reponames:", reponames)



# Get repo's star counts from Github API with the starred_at attribute using the v3 API.
# https://developer.github.com/v3/activity/starring/#list-repositories-being-starred
# https://api.github.com/repos/:owner/:repo/stargazers
# https://api.github.com/repos/tom-doerr/test_api_repo/stargazers
# https://api.github.com/users/tom-doerr/starred
url = 'https://api.github.com/repos/{}/{}/stargazers'.format(username, reponames[0])
headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN)}
r = requests.get(url, headers=headers)
star_count = len(r.json())
print('star_count:', star_count)

@st.cache(ttl=3600)
def get_repo_stars(username, repo):
    '''
    Get all starred_at datetime values of a GitHub repo.
    '''
    starred_at = []
    page = 1
    headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN),
            'Accept': 'application/vnd.github.v3.star+json'}
    while True:
        url = 'https://api.github.com/repos/' + username + '/' + repo + '/stargazers?page=' + str(page) 
        r = requests.get(url, headers=headers)
        page = page + 1
        if len(r.json()) == 0:
            break
        for user in r.json():
            starred_at.append(user['starred_at'])
        time.sleep(0.5)
    return starred_at

# Plot stars over time
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
        # dates = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in repos_stared_at_lists[reponame]]
        dates = repos_stared_at_lists[reponame]
        y = [i for i, _ in enumerate(repos_stared_at_lists[reponame])]
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
    import plotly.graph_objects as go

    # Plot stars over time
    fig = go.Figure()
    for reponame in repos_stared_at_lists.keys():
        # dates = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in repos_stared_at_lists[reponame]]
        dates = repos_stared_at_lists[reponame]
        y = [i for i, _ in enumerate(repos_stared_at_lists[reponame])]
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
        paper_bgcolor="white",
    )

    # Move legend below plot
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))




    st.plotly_chart(fig)












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

    y = [i for i, _ in enumerate(dates_all)]

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
    fig = px.line(x=    dates_filtered, y=y_filtered, title='Github stars over time')
    fig.update_xaxes(nticks=20)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig)





def filter_stared_list(repos_stared_at_lists, date_range):
    '''
    Filter the stared list with the date range.
    '''
    repos_stared_at_lists_filtered = {}
    for reponame, stared_list in repos_stared_at_lists.items():
        stared_list_filtered = [star_date for star_date in stared_list if star_date >= date_range[0] and star_date <= date_range[1]]
        repos_stared_at_lists_filtered[reponame] = stared_list_filtered
    return repos_stared_at_lists_filtered


def convert_to_datetime(repos_stared_at_lists):
    '''
    Convert the dates in the stared list to datetime objects.
    '''
    repos_stared_at_lists_datetime = {}
    for reponame, stared_list in repos_stared_at_lists.items():
        stared_list_datetime = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in stared_list]
        repos_stared_at_lists_datetime[reponame] = stared_list_datetime

    return repos_stared_at_lists_datetime


def num_stars_received_last_x_hours(repos_stared_at_lists, hours=24):
    '''
    Get the number of stars received in the last x hours.
    '''
    repos_stared_at_lists_datetime_filtered = {}
    now = datetime.datetime.now()
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
    import plotly.graph_objects as go


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
        print('repo_num:', repo_num)
        print('reponame:', reponame)
        st.subheader(f'{reponame}')


        dates = repos_stared_at_lists[reponame]
        y = [i for i, _ in enumerate(repos_stared_at_lists[reponame])]
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
        fig.update_xaxes(nticks=20)
        # st.plotly_chart(fig)

        # Remove borders.
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
        )

        # Move legend below plot
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        st.plotly_chart(fig)



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
    st.text("num_stars_last_24_hours_sum: {}".format(num_stars_last_24_hours_sum))

    stars_last_7_days = num_stars_received_last_x_hours(repos_stared_at_lists, hours=7*24)
    stars_last_30_days = num_stars_received_last_x_hours(repos_stared_at_lists, hours=30*24)

    st.text("stars_last_7_days: {}".format(stars_last_7_days[1]))
    st.text("stars_last_30_days: {}".format(stars_last_30_days[1]))

    plot_stars_repos_individually(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)


if __name__ == '__main__':
    main()

