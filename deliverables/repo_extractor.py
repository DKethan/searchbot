import pandas as pd
import requests
from io import StringIO


def download_and_combine_csv(urls: list) -> pd.DataFrame:
    """
    Download CSV files from provided URLs and combine them row-wise.

    The function ensures compatibility by checking column names or count,
    handling encoding issues, and converting numerical columns to integers.

    Args:
        urls (list): List of URLs pointing to CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame containing data from valid CSV files.
    """
    expected_columns = {'user_prompt', 'url_to_check', 'func_rating', 'custom_rating'}
    combined_df = pd.DataFrame()

    for url in urls:
        try:
            raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob', '')
            response = requests.get(raw_url)

            if response.status_code == 200:
                try:
                    content = response.content.decode('utf-8')
                except UnicodeDecodeError:
                    content = response.content.decode('latin1')

                df = pd.read_csv(StringIO(content))

                if set(df.columns) == expected_columns:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                elif len(df.columns) >= 4:
                    df = df.iloc[:, :4]
                    df.columns = ['user_prompt', 'url_to_check', 'func_rating', 'custom_rating']
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                else:
                    print(f"Skipping file with insufficient columns: {url}")
            else:
                print(f"Failed to download from {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    if not combined_df.empty:
        combined_df[['func_rating', 'custom_rating']] = combined_df[['func_rating', 'custom_rating']].round().astype(
            int)

    return combined_df


# Example usage:
urls = [
    "https://github.com/vrutika-prajapati/Credibility-Score-for-Articles/blob/main/projects/deliverable%202/Url_validation.csv",
    "https://github.com/Saikumar08-sk/URL-Validation/blob/main/Deliverable.csv",
    "https://github.com/aditya19111/Project-1-Credibility-Score-for-Articles-Sources-References/blob/main/Deliverable_2/Deliverables_rating_comparison%20-%20Sheet1.csv",
    "https://github.com/SAIKUMAR500/algorithums-for-data-science/blob/main/deliverable2/dataset(Sheet1).csv",
    "https://github.com/drashti-sanghani/Credibility_score_articles/blob/main/Project/Deliverable2/user_ratings.csv",
    "https://github.com/anchalrai101/CREDIBILITY-SCORE/blob/main/urltesting_scores.csv",
    "https://github.com/AliSInamdar/CS676-Ali-Inamdar/blob/Project_1_Credibilty_Score/Deliverable_2/Deliverable_2.csv",
    "https://github.com/bhavnaa22/Algorithms-For-Data-Science/blob/main/Deliverable%202/Deliverable2.csv",
    "https://github.com/bhatthardik4/AlgorithmDS/blob/main/Deliverable_2/HardikBhattD2.csv",
    "https://github.com/Rakeshkanche/Algorithms-for-Data-Science/blob/main/queries_urls_ratings.csv",
    "https://github.com/kristinakim-code/Credibility-Check/blob/main/deliverable2.csv",
    "https://github.com/kamaldomandula/Algorithms-of-data-science-Project/blob/main/Project/project1/Deliverable2/deliverable.csv",
    "https://github.com/ChinmayShetye26/Algo-for-DS-delivery2/blob/main/Sample.csv",
    "https://github.com/krishnam229/Project1/blob/main/deliverable2/sample.csv",
    "https://github.com/drona23/Deliverable2/blob/main/output.csv",
    "https://github.com/JoshuaGottlieb/Chatbot-Credibility-Scorer/blob/main/src/deliverable-02/testing/sample.csv",
    "https://github.com/DKethan/searchbot/blob/dev-01/deliverables/samples/url_validation_results.csv"
]

result_df = download_and_combine_csv(urls)
print(result_df)
