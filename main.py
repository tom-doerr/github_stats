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
from functools import lru_cache
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class GitHubRateLimiter:
    def __init__(self, headers: Dict, safety_buffer: int = 1500):
        self.headers = headers
        self.safety_buffer = safety_buffer
        self._last_check = 0
        self._check_interval = 5  # Check rate limit every 5 seconds
        self.requests_this_hour = []
        
        # Calculate max requests per second to stay under limit
        self.max_requests_per_hour = 5000 - safety_buffer
        self.min_seconds_between_requests = 3600 / self.max_requests_per_hour

    def get_rate_limit_info(self) -> tuple[int, int, datetime]:
        """Get current rate limit info from GitHub"""
        now = time.time()
        if now - self._last_check > self._check_interval:
            r = requests.get('https://api.github.com/rate_limit', headers=self.headers)
            data = r.json()
        
            self.remaining = data['rate']['remaining']
            self.limit = data['rate']['limit']
            self.reset_time = datetime.fromtimestamp(data['rate']['reset'])
            self._last_check = now
        
            # Calculate stats
            reset_in = (self.reset_time - datetime.now()).total_seconds() / 60
            stats = {
                "remaining": self.remaining,
                "limit": self.limit,
                "reset_in_mins": reset_in,
                "requests_this_hour": len(self.requests_this_hour),
                "target_rate": self.max_requests_per_hour/3600,
                "min_seconds_between": self.min_seconds_between_requests
            }
        
            # Print to console
            print(f"\nGitHub API Rate Limit Stats:")
            print(f"  Remaining: {stats['remaining']}/{stats['limit']} requests")
            print(f"  Reset in: {stats['reset_in_mins']:.1f} minutes")
            print(f"  Requests this hour: {stats['requests_this_hour']}")
            print(f"  Target rate: {stats['target_rate']:.1f} requests/second")
            print(f"  Min seconds between requests: {stats['min_seconds_between']:.2f}")
        
            # Update session state
            if 'rate_limit_stats' not in st.session_state:
                st.session_state.rate_limit_stats = {}
            st.session_state.rate_limit_stats = stats
        
        return self.remaining, self.limit, self.reset_time

    def wait_if_needed(self) -> None:
        """Wait if we're approaching rate limits or need to throttle requests"""
        now = time.time()
        
        # Clean up old requests
        hour_ago = now - 3600
        self.requests_this_hour = [t for t in self.requests_this_hour if t > hour_ago]
        
        # Check if we need to wait based on rate limit
        remaining, limit, reset_time = self.get_rate_limit_info()
        if remaining <= self.safety_buffer:
            wait_time = (reset_time - datetime.now()).total_seconds()
            if wait_time > 0:
                st.warning(f"Rate limit approaching ({remaining} remaining). "
                          f"Waiting {wait_time:.1f} seconds for reset...")
                time.sleep(wait_time + 1)
                self.requests_this_hour = []
                return

        # Throttle requests to stay under max_requests_per_hour
        if self.requests_this_hour:
            time_since_last_request = now - self.requests_this_hour[-1]
            if time_since_last_request < self.min_seconds_between_requests:
                sleep_time = self.min_seconds_between_requests - time_since_last_request
                time.sleep(sleep_time)
        
        self.requests_this_hour.append(time.time())

def make_github_request(url: str, headers: Dict) -> requests.Response:
    """Make a GitHub API request with rate limiting"""
    rate_limiter.wait_if_needed()
    return requests.get(url, headers=headers)

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


@lru_cache(maxsize=1000)
def get_etag(url: str) -> str:
    """Get cached ETag for URL. ETags are unique identifiers returned by GitHub 
    that represent the current version of a resource. If we send the same ETag 
    back in a subsequent request with If-None-Match header, GitHub will return 
    304 Not Modified if the resource hasn't changed, saving our API quota."""
    return ""

def save_etag(url: str, etag: str) -> None:
    """Save ETag for URL to be used in future requests. Clears the cache to 
    ensure the new ETag is used."""
    get_etag.cache_clear()
    get_etag.cache_info()
    
def make_conditional_request(url: str, headers: Dict) -> requests.Response:
    """Make conditional request using ETags. This helps reduce API quota usage by:
    1. Including the ETag in If-None-Match header if we have one cached
    2. Getting a 304 Not Modified response (no data) if resource hasn't changed
    3. Getting new data and ETag only if resource has changed"""
    etag = get_etag(url)
    if etag:
        headers = headers.copy()
        headers['If-None-Match'] = etag
    
    response = make_github_request(url, headers=headers)
    
    if response.status_code == 304:  # Not Modified
        return response
        
    if 'ETag' in response.headers:
        save_etag(url, response.headers['ETag'])
        
    return response

def check_rate_limit_exceeded(response) -> Optional[datetime]:
    """Returns reset time if rate limited, None otherwise"""
    if isinstance(response, dict):
        json_response = response
        headers = {}
    else:
        json_response = response.json() if response.status_code != 304 else {}
        headers = response.headers
        
    # Always show current rate limit status
    remaining = int(headers.get('x-ratelimit-remaining', 0))
    limit = int(headers.get('x-ratelimit-limit', 0))
    reset_timestamp = int(headers.get('x-ratelimit-reset', 0))
    reset_time = datetime.fromtimestamp(reset_timestamp)
    
    if remaining < limit * 0.1:  # Warning at 10% remaining
        st.warning(f"GitHub API rate limit low: {remaining}/{limit} requests remaining")
        
    if 'message' in json_response and 'rate limit exceeded' in json_response['message']:
        message = f"""
        GitHub API rate limit exceeded!
        - Remaining: {remaining}/{limit} requests
        - Resets at: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        st.error(message)
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

# Create headers and rate limiter
headers = {'Authorization': f'token {GITHUB_AUTH_TOKEN}'} if GITHUB_AUTH_TOKEN else {}
rate_limiter = GitHubRateLimiter(headers=headers, safety_buffer=1500)
# Initialize rate limit stats
rate_limiter.get_rate_limit_info()

query_params = st.query_params
print("query_params:", query_params)


# Display rate limit stats in sidebar if available
if 'rate_limit_stats' in st.session_state:
    stats = st.session_state.rate_limit_stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### GitHub API Stats")
    st.sidebar.text(f"Remaining: {stats['remaining']}/{stats['limit']}")
    st.sidebar.text(f"Reset in: {stats['reset_in_mins']:.1f} min")
    st.sidebar.text(f"Requests this hour: {stats['requests_this_hour']}")
    st.sidebar.text(f"Target rate: {stats['target_rate']:.1f} req/s")
    st.sidebar.text(f"Min delay: {stats['min_seconds_between']:.2f}s")
    st.sidebar.markdown("---")

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
    reset_time = check_rate_limit_exceeded(response)
    if reset_time:
        wait_time = (reset_time - datetime.now()).total_seconds()
        if wait_time > 0:
            st.info(f"Waiting {wait_time:.0f} seconds for rate limit reset...")
            time.sleep(wait_time + 1)
            continue
            
    try:
        data = response.json()
        if isinstance(data, str):
            st.error(f"Unexpected string response: {data}")
            st.stop()
        if not isinstance(data, list):
            st.error(f"Unexpected response type: {type(data)}")
            st.stop()
            
        for repo in data:
            return_dict[repo['name']] = repo
            
        if 'next' not in response.links.keys():
            break
        url = response.links['next']['url']
            
    except json.JSONDecodeError:
        st.error("Invalid JSON response from GitHub API")
        st.stop()
    except KeyError as e:
        st.error(f"Missing expected field in response: {e}")
        st.stop()

reponames = sorted(return_dict.keys())
# Reset repo selection when username changes
if repo_default and repo_default not in reponames:
    repo_default = ''
    
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
        r = make_github_request(url, headers=headers_accept)
        
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
def get_stars_over_time(reponames, username):
    '''
    Get the number of stars for each repo in reponames over time.
    '''
    repos_stared_at_lists = {}
    for reponame in reponames[:MAX_NUM_REPOS]:
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
    
    total_repos = len(reponames[:MAX_NUM_REPOS])
    repos_stared_at_lists = {}
    
    for repo_num, reponame in enumerate(reponames[:MAX_NUM_REPOS]):
        progress = (repo_num + 1) / total_repos
        progress_text.text(f'Processing {reponame} ({repo_num + 1}/{total_repos})')
        progress_bar.progress(progress)
        
        repo_stars = get_stars_over_time([reponame], username)
        repos_stared_at_lists.update(repo_stars)
    
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

