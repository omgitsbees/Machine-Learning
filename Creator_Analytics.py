import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_channel_stats(channel_url):
    """
    Fetches basic channel statistics (subscribers, views, videos) from a YouTube channel page.
    Note: This relies on web scraping and might break if YouTube's page structure changes.
    Use the official YouTube Data API for a more robust solution in a production environment.
    """
    try:
        response = requests.get(channel_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        subscriber_count_element = soup.find('yt-formatted-string', {'id': 'subscriber-count'})
        view_count_element = soup.find('div', {'id': 'view-count'})
        video_count_element = soup.find('span', {'class': 'yt-simple-endpoint style-scope yt-formatted-string'})

        subscribers = subscriber_count_element.text.strip() if subscriber_count_element else "N/A"
        views = view_count_element.text.strip().replace(' views', '') if view_count_element else "N/A"
        videos = video_count_element.text.strip().split(' ')[0] if video_count_element else "N/A"

        return {'Subscribers': subscribers, 'Views': views, 'Videos': videos}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except AttributeError:
        print("Could not find the required elements on the page. YouTube's structure might have changed.")
        return None

def analyze_multiple_channels(channel_urls):
    """
    Analyzes statistics for a list of YouTube channel URLs.
    """
    all_channel_data = []
    for url in channel_urls:
        print(f"Fetching data for: {url}")
        data = fetch_channel_stats(url)
        if data:
            all_channel_data.append(data)
    return pd.DataFrame(all_channel_data)

def visualize_channel_comparison(df):
    """
    Visualizes the comparison of channel statistics.
    """
    if df.empty:
        print("No channel data to visualize.")
        return

    # Clean up numerical columns for plotting
    for col in ['Subscribers', 'Views', 'Videos']:
        if col in df.columns:
            # Remove non-numeric characters and convert to numeric
            df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    plt.figure(figsize=(15, 5 * len(df)))

    if 'Subscribers' in df.columns:
        plt.subplot(3, 1, 1)
        sns.barplot(x=df.index, y='Subscribers', data=df)
        plt.title('Subscriber Count Comparison')
        plt.ylabel('Subscribers')
        plt.xticks(rotation=45, ha='right')

    if 'Views' in df.columns:
        plt.subplot(3, 1, 2)
        sns.barplot(x=df.index, y='Views', data=df)
        plt.title('Total View Count Comparison')
        plt.ylabel('Total Views')
        plt.xticks(rotation=45, ha='right')

    if 'Videos' in df.columns:
        plt.subplot(3, 1, 3)
        sns.barplot(x=df.index, y='Videos', data=df)
        plt.title('Number of Videos Comparison')
        plt.ylabel('Number of Videos')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    channel_list = [
        "https://www.youtube.com/@LinusTechTips",
        "https://www.youtube.com/@MrBeast",
        "https://www.youtube.com/@mkbhd"
        # Add more channel URLs here
    ]

    channel_data_df = analyze_multiple_channels(channel_list)

    if not channel_data_df.empty:
        print("\nChannel Statistics:")
        print(channel_data_df)

        # Set channel URLs as index for better visualization labels
        channel_data_df.index = [url.split('@')[-1] for url in channel_list]
        visualize_channel_comparison(channel_data_df)