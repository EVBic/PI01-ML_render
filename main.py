
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse


# Used Files:
df_reviews= pd.read_parquet('CleanDatasets\df_reviews_l.parquet')
df_funct_dev = pd.read_parquet('df_funct_dev.parquet')
df_expenses_items = pd.read_parquet('df_expenses_items.parquet')

# Home presentation : 
def presentation():
    '''
     Home Page Displaying a Presentation

    Returns:
    HTMLResponse: HTML response displaying the presentation.
    '''
    return '''
    <html>
        <head>
            <title>API Steam Find Your Fun</title>
            <style>
                body {
                    background:url(https://github.com/EVBic/PI-01-ML-SteamGames-FYF/blob/main/Images/FYF_Main.jpeg);
                    font-family: Georgia, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: white;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                    background-color: black;
                    padding: 10px;
                }
                .centered-button {
                background-color: black;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px; /* optional for rounded corners */
                display: block; /* Makes the button fill the available width */
                margin: 0 auto; /* Centers the button horizontally */
                cursor: pointer; /* Changes cursor to pointer on hover */
                }
            </style>
        </head>
        <body>
            <h1>FIND YOUR FUN</h1>
            <h1>Steam Video Game Queries API</h1>
            
            <p>Welcome to the Steam API, where you can make various queries related to the gaming platform.</p>
            <p><strong>INSTRUCTIONS:</strong></p>
            <p>Click the button below to interact with the API:</p>
            
            <button type="button" class="centered-button" onclick="window.location.href = window.location.href + 'docs'">API Docs</button>
            
            <p>Visit my profile on <a href="https://www.linkedin.com/in/maría-eva-bichi">&nbsp;<img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-blue?style=flat-square&logo=linkedin"></a></p>
            <p>The development of this project is hosted on <a href="https://github.com/EVBic">&nbsp;<img alt="GitHub" src="https://img.shields.io/badge/GitHub-black?style=flat-square&logo=github"></a></p>
        </body>
    </html>
    '''




#Developer Function:
def developer(developer_name: str):
    filtered_developer = df_funct_dev[df_funct_dev['developer'] == developer_name]
    game_count_by_year = filtered_developer.groupby('release_year')['item_id'].count()
    free_games_percentage = (filtered_developer[filtered_developer['price'] == 0.0]
                             .groupby('release_year')['item_id']
                             .count() / game_count_by_year * 100).fillna(0).astype(int)
    results = [] 
    for year, num_games, free_games in zip(game_count_by_year.index, game_count_by_year.values, free_games_percentage.values):
        results.append({
            'Year': int(year),
            'Number of games': int(num_games),
            '% Free games': int(free_games)
        })
    return results



#User Data Function:
def user_data(user_id):
    user = df_reviews[df_reviews['user_id'] == user_id]
    amount_money = df_expenses_items[df_expenses_items['user_id']== user_id]['price'].sum()
    count_items = df_expenses_items[df_expenses_items['user_id']== user_id]['items_count'].iloc[0]
    total_recommendations = user['recommend'].sum()
    total_reviews = len(df_reviews['user_id'].unique())
    percentage_recommendations = (total_recommendations / total_reviews) * 100
    return {
        'user_id': user_id,
        'amount_money': float(amount_money),
        'percentage_recommendation': round(float(percentage_recommendations), 2),
        'total_items': int(count_items)
    }


# UserForGenre Function:

def User_For_Genre(genre: str) -> dict:
    genre_df = df_userfg[df_userfg['genres'] == genre]
    genre_df['playtime_hours'] = genre_df['playtime_hours'] / 60
    user_playtime = genre_df.groupby('user_id')['playtime_hours'].sum()
    top_user = user_playtime.idxmax()
    top_user_genre_df = genre_df[genre_df['user_id'] == top_user]
    playtime_by_year = top_user_genre_df.groupby('release_year')['playtime_hours'].sum()
    playtime_list = []
    for year, playtime in playtime_by_year.items():
        try:
            year_int = int(year)
            playtime_float = float(playtime)
            playtime_list.append({"Year": year_int, "Hours": playtime_float})
        except ValueError:
            continue
    return {
        "User with most playtime for Genre {}".format(genre): str(top_user),
        "Playtime": playtime_list
    }

# best_developer_year Function:
def best_developer_year(year: int):
    df_reviews['release_year'] = pd.to_numeric(df_reviews['release_year'], errors='coerce')
    df_filtered = df_reviews[(df_reviews['release_year'] == year) & (df_reviews['recommend'] == True) & (df_reviews['sentiment_analysis'] == 2)]
    df_grouped = df_filtered.groupby('developer').size()
    top_developers = df_grouped.nlargest(3).index.tolist()
    result = [{"Rank {}".format(i+1): dev} for i, dev in enumerate(top_developers)]
    return result

# dev_reviews_analysis Function:

def dev_reviews_analysis(developer):
    reviews2 = df_reviews[df_reviews['developer'] == str(developer)]  # Convert developer name to a string
    sentiment_counts = {'Negative': 0, 'Positive': 0}
    for index, row in reviews2.iterrows():
        sentiment = row['sentiment_analysis']
        sentiment_category = ''
        if sentiment == 0:
            sentiment_category = 'Negative'
        elif sentiment == 2:
            sentiment_category = 'Positive'
        else:
            continue
        sentiment_counts[sentiment_category] += 1
    return {"developer": developer, "sentiment_counts": sentiment_counts}

# game_recommendation Function:
def game_recommendation(item_id: int):
    filtered_game = df_recommendation[df_recommendation['item_id'] == item_id]
    game_genres = set(filtered_game['genres'].str.split(',').explode())
    recommended_games = df_recommendation[df_recommendation['genres'].apply(lambda x: len(set(x.split(',')).intersection(game_genres)) >= 1)]
    recommended_games.loc[:, 'genres_vector'] = recommended_games['genres'].apply(lambda x: np.array([1 if genre in x else 0 for genre in game_genres]))
    filtered_game.loc[:, 'genres_vector'] = filtered_game['genres'].apply(lambda x: np.array([1 if genre in x else 0 for genre in game_genres]))
    recommended_games.loc[:, 'similarity'] = recommended_games.apply(lambda row: cosine_similarity([row['genres_vector']], [filtered_game['genres_vector'].iloc[0]])[0][0], axis=1)
    recommended_games = recommended_games.sort_values(['similarity'], ascending=[False])
    top_recommended_games = recommended_games.head(5)
    recommended_games_dict = {}
    recommended_games_dict['Because you liked ' + filtered_game['item_name'].iloc[0] + ', you might also enjoy...'] = top_recommended_games[['item_name']].to_dict(orient='records')
    return recommended_games_dict


app = FastAPI()
# Functions
@app.get(path="/", response_class=HTMLResponse,
         tags=["Home"])
def home():
    '''
    Home Page Presentation

    Returns:
    Hellow Gamer!!
    '''
    return presentation()


# Developer FastAPI
@app.get(path='/developer',
          description="""<font color="purple">
                         1. Click on "Try it out".<br>
                         2. Enter the developer's name in the box below.<br>
                         3. Scroll to "Resposes" to see the number of items and percentage of Free content per year from that developer.
                       </font>""",
          tags=["General Inquiries"])
def developer_handler(developer_name: str = Query(..., description="Developer", example="Laush Dmitriy Sergeevich")):
    return developer(developer_name)


# User Data FastAPI
@app.get(path = '/userdata',
          description = """ <font color="purple">
                        INSTRUCTIONS<br>
                        1. Click on "Try it out".<br>
                        2. Enter the user_id in the box below.<br>
                        3. Scroll to "Resposes" to see the amount of money spent by the user, the percentage of recommendations made by the user and the number of items the user has.
                        </font>
                        """,
         tags=["General Inquiries"])
def userdata(user_id: str = Query(..., description="user_id", example="js41637")):
    return user_data(user_id)


# UserForGenre FastAPI
@app.get('/UserForGenre', response_class=JSONResponse,
         description = """ <font color="purple">
                        1. Click on "Try it out".<br>
                        2. Enter the genre in the box below.<br>
                        3. Scroll to "Responses" to see the user with the most playtime for the given genre and their playtime by year.
                        </font>
                        """,
        tags=["General Inquiries"])
def UserForGenre(genre: str = Query(..., description="Game´s genre", example='Indie')):
    return User_For_Genre(genre)


# best_developer_year FastAPI
@app.get("/best_developer_year/{year}", 
         description = """ <font color="purple">
                        INSTRUCTIONS<br>
                        1. Click on "Try it out".<br>
                        2. Enter the year in the box below.<br>
                        3. Scroll to "Responses" to see the result of the classification.
                        </font>
                        """,
         tags=["General Inquiries"])
def bestdeveloperyear(year: int):
    return best_developer_year(year)


# dev_reviews_analysis FastAPI
@app.get('/dev_reviews_analysis',
         description="""<font color="purple">
                    INSTRUCTIONS<br>
                    1. Click on "Try it out".<br>
                    2. Enter the dev in the box below.<br>
                    3. Scroll down to "Responses" to view the number of user review records categorized with sentiment analysis.
                    </font>
                    """,
         tags=["General Inquiries"])
def devreviewsanalysis(developer: str = Query(..., description="Returns a dictionary with the developer's name", example="Trion Worlds, Inc.")):
    return dev_reviews_analysis(developer)                                      
                                         
# game_recommendation FastAPI
@app.get('/game_recommendation',
         description=""" <font color="purple">
                    INSTRUCTIONS<br>
                    1. Click on "Try it out".<br>
                    2. Enter the name of a game in box below.<br>
                    3. Scroll to "Resposes" to see recommended games.
                    </font>
                    """,
         tags=["Recommendation"])
def gamerecommendation(item_id:int = Query(..., description="Game from which the recommendation of other games is made", example="70")):
    return game_recommendation(item_id)                                                         
    
