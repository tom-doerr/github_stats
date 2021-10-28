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
        url = 'https://api.github.com/repos/{}/{}/stargazers'.format(username, reponame)
        while True:
            headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN),
                    'Accept': 'application/vnd.github.v3.star+json'}
            r = requests.get(url, headers=headers)
            star_counts = len(r.json())
            print("r.json():", r.json())
            stars_over_time.append(star_counts)
            print('reponame:', reponame, 'star_count:', star_counts)
            time.sleep(1)

            # Parse json and get the time of when the repo was starred.
            for repo in r.json():
                repos_stared_at.append(repo['starred_at'])

            if 'next' not in response.links.keys():
                break
            else:
                url = response.links['next']['url']

        repos_stared_at_lists[reponame] = repos_stared_at
        print("repos_stared_at_lists:", repos_stared_at_lists)

    counter.text('')
    repo_name_text.text('')
    return repos_stared_at_lists


# Plot stars over time
def plot_stars_over_time(reponames, username, repos_stared_at_lists):
    '''
    Plot stars over time.
    '''

    # Plot stars over time
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    for reponame in repos_stared_at_lists.keys():
        dates = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in repos_stared_at_lists[reponame]]
        y = [i for i, _ in enumerate(repos_stared_at_lists[reponame])]
        ax.plot(dates, y, label=reponame)

    # Format plot
    date_fmt = mdates.DateFormatter('%m-%d-%Y')
    ax.xaxis.set_major_formatter(date_fmt)
    _ = plt.xticks(rotation=90)
    # _ = plt.legend()
    _ = plt.title('Github stars over time')
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Number of stars')

    # Show legend below plot.
    _ = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)

    # Show plot in streamlit.
    fig = ax.get_figure()
    st.pyplot(fig)
    # plt.show()

def plot_stars_over_time_all(reponames, username, repos_stared_at_lists):
    '''
    Plot stars over time.
    '''

    # Plot stars over time
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)
    dates_all = []
    for reponame in repos_stared_at_lists.keys():
        dates = [datetime.datetime.strptime(repo_stared_at, "%Y-%m-%dT%H:%M:%SZ") for repo_stared_at in repos_stared_at_lists[reponame]]
        dates_all.extend(dates)


    # Sort the dates.
    dates_all = sorted(dates_all)

    y = [i for i, _ in enumerate(dates_all)]
    ax.plot(dates_all, y)

    # Format plot
    date_fmt = mdates.DateFormatter('%m-%d-%Y')
    ax.xaxis.set_major_formatter(date_fmt)
    _ = plt.xticks(rotation=90)
    _ = plt.legend()
    _ = plt.title('Github stars over time')
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Number of stars')

    # Show plot in streamlit.
    fig = ax.get_figure()
    st.pyplot(fig)
    # plt.show()

def main():
    repos_stared_at_lists = get_stars_over_time(reponames, username)
    print("repos_stared_at_lists:", repos_stared_at_lists)
    plot_stars_over_time(reponames, username, repos_stared_at_lists)
    plot_stars_over_time_all(reponames, username, repos_stared_at_lists)

if __name__ == '__main__':
    main()

