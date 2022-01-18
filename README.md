<h1 align="center"> 📈 github_stats </h1>


<p align="center">
    <a href="https://github.com/tom-doerr/github_stats/stargazers"
        ><img
            src="https://img.shields.io/github/stars/tom-doerr/github_stats?colorA=2c2837&colorB=c9cbff&style=for-the-badge&logo=starship style=flat-square"
            alt="Repository's starts"
    /></a>
    <a href="https://github.com/tom-doerr/github_stats/issues"
        ><img
            src="https://img.shields.io/github/issues-raw/tom-doerr/github_stats?colorA=2c2837&colorB=f2cdcd&style=for-the-badge&logo=starship style=flat-square"
            alt="Issues"
    /></a>
    <a href="https://github.com/tom-doerr/github_stats/blob/main/LICENSE"
        ><img
            src="https://img.shields.io/github/license/tom-doerr/github_stats?colorA=2c2837&colorB=b5e8e0&style=for-the-badge&logo=starship style=flat-square"
            alt="License"
    /><br />
    <a href="https://github.com/tom-doerr/github_stats/commits/main"
		><img
			src="https://img.shields.io/github/last-commit/tom-doerr/github_stats/main?colorA=2c2837&colorB=ddb6f2&style=for-the-badge&logo=starship style=flat-square"
			alt="Latest commit"
    /></a>
    <a href="https://github.com/tom-doerr/github_stats"
        ><img
            src="https://img.shields.io/github/repo-size/tom-doerr/github_stats?colorA=2c2837&colorB=89DCEB&style=for-the-badge&logo=starship style=flat-square"
            alt="GitHub repository size"
    /></a>
</p>

<p align="center">
    <img src='screenshot.png'>
    <p align="center">
        Website: https://share.streamlit.io/tom-doerr/github_stats/main/main.py?username=tom-doerr
    </p>
</p>



## What is it?
This app allows you to plot the number of Github stars received over time for a user.
## How do I use it?
1. Enter a Github username in the sidebar.
2. Select a date range with the slider.
## How does it work?
1. Get user's repo names from Github API
2. Get repo's star counts from Github API
3. Plot stars over time
## How do I get the app?
1. Clone the repo:
```
git clone git@github.com:tom-doerr/github_stats.git
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run github_stats/main.py
```

## Rate Limits
You can increase your rate limit by creating a directory `.streamlit` in the project root and adding the file `.streamlit/secrets.toml` containing your github auth token:
```
github_auth_token = "..."
```
