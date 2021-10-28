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

MAX_NUM_REPOS = None

# Read auth token GITHUB_AUTH_TOKEN
GITHUB_AUTH_TOKEN = os.environ['GITHUB_AUTH_TOKEN']

# Get user's repo names from Github API
username = st.sidebar.text_input('Enter Github username:')

datetime_tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
datetime_tomorrow_midnight = datetime.datetime(datetime_tomorrow.year,
                                               datetime_tomorrow.month,
                                               datetime_tomorrow.day)

# Add slider to sidebar that allows to select daterange.
st.sidebar.markdown('**Date range**')
date_range = st.sidebar.slider('Select date range:',
    value=(datetime.datetime.strptime('2020-01-01', "%Y-%m-%d"),
           datetime_tomorrow_midnight),
    format='YYYY-MM-DD')


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
print("reponames:", reponames)

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

@st.cache
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
        print("r.json():", r.json())
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
        print("repos_stared_at_lists:", repos_stared_at_lists)

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
        if date in repos_stared_at_filtered[reponame]:
            dates_filtered.append(date)
            y_filtered.append(y_val)

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
    # plt.show()

def filter_stared_list(repos_stared_at_lists, date_range):
    '''
    Filter the stared list with the date range.
    '''
    print("date_range:", date_range)
    print(type(date_range))
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



def main():
    repos_stared_at_lists = get_stars_over_time(reponames, username)
    repos_stared_at_lists = convert_to_datetime(repos_stared_at_lists)
    print("repos_stared_at_lists:", repos_stared_at_lists)
    repos_stared_at_filtered = filter_stared_list(repos_stared_at_lists, date_range)
    plot_stars_over_time(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)
    plot_stars_over_time_all(reponames, username, repos_stared_at_lists, repos_stared_at_filtered)

if __name__ == '__main__':
    main()

