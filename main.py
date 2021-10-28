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

# Read auth token GITHUB_AUTH_TOKEN
GITHUB_AUTH_TOKEN = os.environ['GITHUB_AUTH_TOKEN']

# Get user's repo names from Github API
username = st.sidebar.text_input('Enter Github username:')
# username = 'tom-doerr'
url = 'https://api.github.com/users/{}/repos'.format(username)
headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN)}
r = requests.get(url, headers=headers)
reponames = [repo['name'] for repo in r.json()]
print("reponames:", reponames)

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
    for reponame in reponames[:4]:
        url = 'https://api.github.com/repos/{}/{}/stargazers'.format(username, reponame)
        headers = {'Authorization': 'token {}'.format(GITHUB_AUTH_TOKEN),
                'Accept': 'application/vnd.github.v3.star+json'}
        r = requests.get(url, headers=headers)
        star_counts = len(r.json())
        print("r.json():", r.json())
        stars_over_time.append(star_counts)
        print('reponame:', reponame, 'star_count:', star_counts)
        time.sleep(1)

        # Parse json and get the time of when the repo was starred.
        repos_stared_at = []
        for repo in r.json():
            repos_stared_at.append(repo['starred_at'])
        repos_stared_at_lists[reponame] = repos_stared_at
        print("repos_stared_at_lists:", repos_stared_at_lists)

    return repos_stared_at_lists


# Plot stars over time
def plot_stars_over_time(reponames, username):
    '''
    Plot stars over time.
    '''
    repos_stared_at_lists = get_stars_over_time(reponames, username)
    print("repos_stared_at_lists:", repos_stared_at_lists)

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
    _ = plt.legend()
    _ = plt.title('Github stars over time')
    _ = plt.xlabel('Date')
    _ = plt.ylabel('Number of stars')

    # Show plot in streamlit.
    fig = ax.get_figure()
    st.pyplot(fig)
    # plt.show()


def main():
    plot_stars_over_time(reponames, username)


if __name__ == '__main__':
    main()

